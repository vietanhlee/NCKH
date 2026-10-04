"""
=============================================================================
 Hướng 4: Spatio-Temporal DINO for Continuous Road Space Occupancy 
          and Congestion Level of Service (LoS) Estimation
 Module: Train Pipeline (Quy trình huấn luyện và đánh giá chỉ số ùn tắc đô thị)
 Hỗ trợ Multi-GPU, Mixed Precision (AMP), Resume Training chuẩn Production
=============================================================================
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import project root
_dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)

from common.backbone_loader import get_dino_backbone
from common.gpu_utils import setup_multi_gpu, unwrap_model, save_checkpoint, load_checkpoint
from direction4_temporal_density.dataset import TemporalTrafficDataset
from direction4_temporal_density.models import SpatioTemporalDensityNet
from direction4_temporal_density.losses import SpatioTemporalDensityLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Spatio-Temporal Traffic Density & LoS Estimation")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục background")
    parser.add_argument("--origin_dir", "--data_dir", dest="origin_dir", type=str, default="output", help="Thư mục origin images")
    parser.add_argument("--csv_file", type=str, default=None, help="File CSV nhãn số lượng xe (nếu có)")
    parser.add_argument("--save_dir", "--output_dir", dest="save_dir", type=str, default="checkpoints/direction4_temporal_density", help="Thư mục lưu mô hình")
    parser.add_argument("--backbone", "--model_name", dest="backbone", type=str, default="dinov3_vits16", help="Tên backbone DINO")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn trọng số backbone ban đầu")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn file checkpoint (.pth) để tiếp tục huấn luyện")
    parser.add_argument("--window_size", type=int, default=4, help="Số khung hình trong một cửa sổ thời gian")
    parser.add_argument("--img_size", type=int, default=224, help="Kích thước ảnh đầu vào")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size mỗi GPU")
    parser.add_argument("--epochs", type=int, default=20, help="Số epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Đóng băng ViT backbone")
    parser.add_argument("--max_sequences", type=int, default=None, help="Giới hạn số chuỗi kiểm thử nhanh")
    parser.add_argument("--num_workers", type=int, default=0, help="Số luồng nạp dữ liệu")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán ('cuda' hoặc 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face user access token cho các mô hình có quyền truy cập đóng (Meta DINOv3)")
    parsed, unknown = parser.parse_known_args()
    if unknown:
        print(f"⚠️ [CLI Warning] Bỏ qua các đối số chưa khai báo: {unknown}")
    if parsed.hf_token:
        os.environ["HF_TOKEN"] = parsed.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = parsed.hf_token
    return parsed


def train_temporal_density(args):
    """Quy trình huấn luyện toàn diện mạng SpatioTemporalDensityNet."""
    os.makedirs(args.save_dir, exist_ok=True)
    device_arg = args.device

    print("=" * 80)
    print(" 🚀 [Direction 4] KHỞI CHẠY HUẤN LUYỆN SPATIO-TEMPORAL DENSITY & LoS ESTIMATION")
    print("=" * 80)

    # 1. Dataset & DataLoader
    dataset = TemporalTrafficDataset(
        bg_dir=args.bg_dir,
        origin_dir=args.origin_dir,
        window_size=args.window_size,
        img_size=args.img_size,
        csv_file=args.csv_file,
        is_train=True,
        max_sequences=args.max_sequences,
    )

    if len(dataset) == 0:
        print("❌ [Lỗi] Không tìm thấy dữ liệu hợp lệ trong thư mục chỉ định.")
        return

    # 2. Khởi tạo Backbone DINO & Mô hình Không-Thời gian
    backbone, embed_dim, _ = get_dino_backbone(
        model_name=args.backbone,
        pretrained=True,
        device=device_arg,
    )

    base_model = SpatioTemporalDensityNet(
        backbone=backbone,
        embed_dim=embed_dim,
        delta_dim=128,
        temporal_dim=256,
        num_los_classes=4,
        freeze_backbone=args.freeze_backbone,
    )

    model, device, num_gpus, effective_batch_size, effective_lr = setup_multi_gpu(
        model=base_model,
        batch_size_per_gpu=args.batch_size,
        base_lr=args.lr,
        device_arg=device_arg,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(dataset) > effective_batch_size),
    )

    # 3. Criterion, Optimizer & Scheduler
    criterion = SpatioTemporalDensityLoss().to(device)

    raw_model = unwrap_model(model)
    optimizer = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=effective_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device_type, enabled=(device.type == "cuda"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 4. Khôi phục từ checkpoint nếu có cờ --resume hoặc nạp --weights
    start_epoch = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Không tìm thấy file checkpoint resume: {args.resume}")
        print(f"\n🔄 [Resume] Khôi phục toàn bộ trạng thái huấn luyện từ checkpoint: {args.resume}")
        ckpt_data = load_checkpoint(
            load_path=args.resume,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            strict=False,
            verbose=True,
        )
        if "epoch" in ckpt_data and ckpt_data["epoch"] is not None:
            start_epoch = int(ckpt_data["epoch"])
            print(f"   ⏱️ [Epoch] Khôi phục tại epoch {start_epoch}. Sẽ tiếp tục chạy từ epoch {start_epoch + 1}.")
            if args.epochs <= start_epoch:
                target_epochs = start_epoch + args.epochs
                print(f"   💡 [Gia hạn Epochs] Số epochs cài đặt ({args.epochs}) <= epoch checkpoint ({start_epoch}).")
                print(f"      -> Tự động huấn luyện thêm {args.epochs} epochs (Tổng mới: {target_epochs} epochs).")
                args.epochs = target_epochs
    elif args.weights:
        if os.path.isfile(args.weights):
            print(f"\n📦 [Weights] Nạp trọng số khởi tạo ban đầu: {args.weights}")
            load_checkpoint(load_path=args.weights, model=raw_model, device=device, strict=False, verbose=True)

    # 5. Huấn luyện
    best_mae = float("inf")
    start_time = time.time()
    history = {
        "epochs": [],
        "loss": [],
        "mae_occupancy": [],
        "los_acc": [],
    }
    print(f"\n🏁 [Train] Bắt đầu huấn luyện từ Epoch [{start_epoch+1}/{args.epochs}] trên {max(1, num_gpus)} thiết bị...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        correct_los = 0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for batch in pbar:
            rgb_seq = batch["rgb_seq"].to(device, non_blocking=True)
            delta_seq = batch["delta_seq"].to(device, non_blocking=True)
            targets = {
                "occupancy_seq": batch["occupancy_seq"].to(device, non_blocking=True),
                "current_occupancy": batch["current_occupancy"].to(device, non_blocking=True),
                "current_los": batch["current_los"].to(device, non_blocking=True),
                "trend": batch["trend"].to(device, non_blocking=True),
            }

            optimizer.zero_grad()

            if scaler.is_enabled():
                with torch.amp.autocast(device_type=device_type):
                    preds = model(rgb_seq, delta_seq)
                    loss_dict = criterion(preds, targets)
                    loss = loss_dict["loss_total"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(rgb_seq, delta_seq)
                loss_dict = criterion(preds, targets)
                loss = loss_dict["loss_total"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=2.0)
                optimizer.step()

            # Thống kê độ chính xác
            bs = rgb_seq.size(0)
            mae = torch.abs(preds["pred_occupancy"] - targets["current_occupancy"]).mean().item()
            pred_los = torch.argmax(preds["logits_los"], dim=-1)
            correct = (pred_los == targets["current_los"]).sum().item()

            total_loss += loss.item() * bs
            total_mae += mae * bs
            correct_los += correct
            total_samples += bs

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mae_occ": f"{mae:.3f}",
                "los_acc": f"{(correct / bs) * 100:.1f}%",
            })

        scheduler.step()

        epoch_loss = total_loss / max(1, total_samples)
        epoch_mae = total_mae / max(1, total_samples)
        epoch_acc = (correct_los / max(1, total_samples)) * 100.0

        history["epochs"].append(epoch + 1)
        history["loss"].append(float(epoch_loss))
        history["mae_occupancy"].append(float(epoch_mae))
        history["los_acc"].append(float(epoch_acc))

        print(
            f"📊 Epoch [{epoch+1:02d}/{args.epochs:02d}] "
            f"Loss: {epoch_loss:.4f} | "
            f"Occupancy MAE: {epoch_mae:.4f} | "
            f"LoS Accuracy: {epoch_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Lưu checkpoint tốt nhất theo MAE Occupancy
        if epoch_mae < best_mae:
            best_mae = epoch_mae
            for ckpt_name in ["best_temporal_model.pth", "best_checkpoint.pth"]:
                ckpt_path = os.path.join(args.save_dir, ckpt_name)
                save_checkpoint(
                    save_path=ckpt_path,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    metrics={"loss": epoch_loss, "mae_occupancy": epoch_mae, "los_accuracy": epoch_acc},
                    extra_dict={"best_mae": best_mae, "num_gpus": num_gpus, "args": vars(args)},
                    verbose=(ckpt_name == "best_temporal_model.pth"),
                )

        # Lưu epoch cuối cùng phục vụ resume liên tục
        last_ckpt_path = os.path.join(args.save_dir, "last_checkpoint.pth")
        save_checkpoint(
            save_path=last_ckpt_path,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            metrics={"loss": epoch_loss, "mae_occupancy": epoch_mae, "los_accuracy": epoch_acc},
            extra_dict={"best_mae": best_mae, "num_gpus": num_gpus, "args": vars(args)},
            verbose=False,
        )

    # Lưu metrics JSON
    metrics_path = os.path.join(args.save_dir, "training_metrics.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "args": vars(args),
                "history": history,
                "best_mae": float(best_mae),
            }, f, indent=2, ensure_ascii=False)
        print(f"📊 [Metrics] Đã lưu lịch sử huấn luyện tại: {metrics_path}")
    except Exception as e_m:
        print(f"⚠️ [Metrics Warning] {e_m}")

    # Vẽ biểu đồ Loss & MAE Curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
        ax1.plot(history["epochs"], history["loss"], "b-o", linewidth=2, label="Train Loss (Smooth L1 + CE)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Hàm mất mát Spatio-Temporal", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        ax2.plot(history["epochs"], history["mae_occupancy"], "r-s", linewidth=2, label="Occupancy MAE")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Occupancy MAE", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.grid(True, linestyle="--", alpha=0.6)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(history["epochs"], history["los_acc"], "g--^", linewidth=2, label="LoS Accuracy (%)")
        ax2_twin.set_ylabel("LoS Accuracy (%)", color="g")
        ax2_twin.tick_params(axis="y", labelcolor="g")

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax2.set_title("Chỉ số MAE Chiếm dụng & Độ chính xác LoS", fontweight="bold")

        plt.tight_layout()
        loss_curve_path = os.path.join(args.save_dir, "loss_curve.png")
        plt.savefig(loss_curve_path, bbox_inches="tight")
        plt.close()
        print(f"📈 [Charts] Đã lưu biểu đồ học tập tại: {loss_curve_path}")
    except Exception as e_plot:
        print(f"⚠️ [Chart Warning] {e_plot}")

    # Xuất ảnh trực quan hóa PCA Feature Map & Kết quả ước lượng không-thời gian
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print("\n🎨 [Visualization] Đang xuất ảnh trực quan hóa Spatio-Temporal Density & LoS...")
        sample_batch = next(iter(dataloader))
        val_rgb_seq = sample_batch["rgb_seq"][:1].to(device)
        val_delta_seq = sample_batch["delta_seq"][:1].to(device)
        val_occ_seq = sample_batch["occupancy_seq"][0].cpu().numpy()
        val_los_gt = int(sample_batch["current_los"][0].item())
        cam_id = sample_batch["cam_id"][0]

        model.eval()
        with torch.no_grad():
            preds_demo = model(val_rgb_seq, val_delta_seq)
            pred_occ = float(preds_demo["pred_occupancy"][0].item())
            pred_los = int(torch.argmax(preds_demo["logits_los"][0]).item())

        last_frame_tensor = val_rgb_seq[0, -1]
        img_np = last_frame_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0.0, 1.0)

        delta_np = val_delta_seq[0, -1, 0].cpu().numpy()
        delta_np = (delta_np * 0.5) + 0.5
        delta_np = np.clip(delta_np, 0.0, 1.0)

        H, W = img_np.shape[:2]
        road_mask = dataset._get_road_mask(cam_id, H, W)

        backbone_m = raw_model.backbone
        backbone_m.eval()
        with torch.no_grad():
            feat_out = backbone_m(val_rgb_seq[:, -1])
            if isinstance(feat_out, dict):
                tokens = feat_out.get("x_norm_patchtokens", feat_out.get("patch_tokens"))
            elif hasattr(backbone_m, "get_intermediate_layers"):
                layers = backbone_m.get_intermediate_layers(val_rgb_seq[:, -1], n=1, return_class_token=True)
                tokens = layers[0][0] if isinstance(layers[0], tuple) else layers[0]
            else:
                tokens = feat_out

        if tokens is not None:
            tokens_np = tokens[0].detach().cpu().numpy()
            tokens_centered = tokens_np - tokens_np.mean(axis=0)
            u, s, vt = np.linalg.svd(tokens_centered, full_matrices=False)
            pca3 = u[:, :3]
            pca3 = (pca3 - pca3.min(axis=0)) / (pca3.max(axis=0) - pca3.min(axis=0) + 1e-6)
            h_p = int(np.sqrt(len(tokens_np)))
            pca_rgb = pca3.reshape(h_p, h_p, 3)
        else:
            pca_rgb = np.zeros((14, 14, 3))

        fig, axs = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)

        overlay = img_np.copy()
        overlay[road_mask, 1] = np.clip(overlay[road_mask, 1] * 0.6 + 0.4, 0.0, 1.0)
        axs[0].imshow(overlay)
        axs[0].set_title(f"(a) Frame $x_t$ + Road Mask\nCam: {cam_id}", fontsize=11, fontweight="bold")
        axs[0].axis("off")

        im2 = axs[1].imshow(delta_np, cmap="inferno")
        axs[1].set_title("(b) Optical Motion $\\Delta_t$\n(Road Subtraction)", fontsize=11, fontweight="bold")
        axs[1].axis("off")
        plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

        axs[2].imshow(pca_rgb)
        axs[2].set_title("(c) DINOv3 PCA Feature Map\n(Semantic Clustering)", fontsize=11, fontweight="bold")
        axs[2].axis("off")

        los_names = ["0: Free-Flow", "1: Moderate", "2: Slow", "3: Gridlock"]
        time_steps = list(range(1, len(val_occ_seq) + 1))
        axs[3].plot(time_steps, val_occ_seq, "b-o", linewidth=2, label="GT Proxy $\\rho$")
        axs[3].plot([time_steps[-1]], [pred_occ], "r*", markersize=14, label=f"Pred $\\hat{{\\rho}}$: {pred_occ:.3f}")
        axs[3].axhline(y=0.15, color="green", linestyle=":", alpha=0.7, label="LoS Thresh 1 (0.15)")
        axs[3].axhline(y=0.35, color="orange", linestyle=":", alpha=0.7, label="LoS Thresh 2 (0.35)")
        axs[3].axhline(y=0.60, color="red", linestyle=":", alpha=0.7, label="LoS Thresh 3 (0.60)")
        axs[3].set_ylim(-0.05, 1.05)
        axs[3].set_xlabel("Time step in Window (t)")
        axs[3].set_ylabel("Occupancy Ratio $\\rho$")
        
        gt_los_str = los_names[val_los_gt] if 0 <= val_los_gt < 4 else str(val_los_gt)
        pred_los_str = los_names[pred_los] if 0 <= pred_los < 4 else str(pred_los)
        color_badge = "green" if pred_los == val_los_gt else "red"
        axs[3].set_title(f"(d) Occupancy Sequence & LoS\nGT: {gt_los_str} | Pred: {pred_los_str}", 
                          fontsize=10, fontweight="bold", color=color_badge)
        axs[3].grid(True, linestyle="--", alpha=0.5)
        axs[3].legend(fontsize=8, loc="upper left")

        plt.tight_layout()
        prog_path = os.path.join(args.save_dir, "temporal_density_progress.png")
        plt.savefig(prog_path, bbox_inches="tight")
        plt.close()
        print(f"🖼️ [Visualization] Đã lưu ảnh trực quan hóa tại: {prog_path}")
    except Exception as e_vis:
        print(f"⚠️ [Visualization Warning] {e_vis}")

    elapsed = time.time() - start_time
    print(f"\n🎉 [Complete] Huấn luyện Direction 4 hoàn tất sau {elapsed/60:.2f} phút! Kỷ lục MAE Occupancy: {best_mae:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train_temporal_density(args)
