"""
=============================================================================
 Hướng 4: Foreground-Enhanced Counting — Training Pipeline
 Huấn luyện mô hình ước lượng số lượng xe với kênh đầu vào tăng cường Foreground (RGB + Δ)
 Hỗ trợ giao thức đánh giá phân lập Spatial Disjoint Camera Split và Few-Shot
=============================================================================
"""

import argparse
import os
import re
import sys

# Chống xung đột OpenMP trên Windows Anaconda và đảm bảo hiển thị ký tự tiếng Việt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.backbone_loader import get_dino_backbone
from common.gpu_utils import setup_multi_gpu, unwrap_model, save_checkpoint
from dataset import FGCountingDataset
from models import DINOv3FGCountingModel


def evaluate_mae(model, loader, device, mode="4channel"):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            targets = batch["counts"].to(device)
            rgb = batch["rgb"].to(device)
            delta = batch.get("delta")
            if delta is not None:
                delta = delta.to(device)
            preds = model(rgb, delta=delta)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # MAE per class: [xe_may, o_to, tong]
    mae_per_class = np.mean(np.abs(all_preds - all_targets), axis=0)
    rmse_per_class = np.sqrt(np.mean((all_preds - all_targets) ** 2, axis=0))

    return {
        "mae_bike": float(mae_per_class[0]),
        "mae_car": float(mae_per_class[1]),
        "mae_total": float(mae_per_class[2]),
        "rmse_total": float(rmse_per_class[2]),
    }


def train_fg_counting(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(" 🚦 TRAINING FOREGROUND-ENHANCED VEHICLE COUNTING (STAGE 1 UPGRADE)")
    print("=" * 78)
    print(f" Backbone Architecture : {args.backbone}")
    print(f" Mode                  : {args.mode} (4channel: RGB+Δ | spatial_attention)")
    print(f" Few-Shot Ratio        : {args.few_shot_ratio * 100:.1f}%")
    print(f" Epochs                : {args.epochs}")
    print(f" Batch Size            : {args.batch_size}")
    print(f" Learning Rate         : {args.lr}")
    print(f" Device                : {device}")
    print("=" * 78)

    # 1. Phân vùng camera Train / Test Disjoint
    if not os.path.isfile(args.csv_file):
        raise FileNotFoundError(f"Không tìm thấy file nhãn CSV tại: {args.csv_file}")

    df_full = pd.read_csv(args.csv_file)
    def get_cam(fn):
        m = re.search(r"^(\d+)_", str(fn))
        return str(int(m.group(1))) if m else "0"
    all_cams = list(sorted(df_full["filename"].apply(get_cam).unique()))
    np.random.seed(args.seed)
    np.random.shuffle(all_cams)

    n_test_cams = max(1, int(len(all_cams) * 0.2))
    test_cams = all_cams[:n_test_cams]
    train_cams = all_cams[n_test_cams:]

    print(f"📍 Tổng số trạm camera: {len(all_cams)} (Train: {len(train_cams)}, Test: {len(test_cams)})")

    # 2. Datasets & Loaders
    train_ds = FGCountingDataset(
        csv_file=args.csv_file,
        origin_dir=args.origin_dir,
        bg_dir=args.bg_dir,
        match_strategy=args.match_strategy,
        img_size=args.img_size,
        is_train=True,
        camera_id_list=train_cams,
        few_shot_ratio=args.few_shot_ratio,
        seed=args.seed,
    )
    test_ds = FGCountingDataset(
        csv_file=args.csv_file,
        origin_dir=args.origin_dir,
        bg_dir=args.bg_dir,
        match_strategy=args.match_strategy,
        img_size=args.img_size,
        is_train=False,
        camera_id_list=test_cams,
        few_shot_ratio=1.0,
    )

    # 2. Khởi tạo Model & Tự động cấu hình Toàn bộ GPU (Multi-GPU Engine)
    backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        pretrained=True,
        weights_path=args.weights,
        device=args.device,
        hf_token=args.hf_token,
    )
    base_model = DINOv3FGCountingModel(
        backbone=backbone,
        embed_dim=embed_dim,
        mode=args.mode,
        freeze_backbone=args.freeze_backbone,
    )

    # Tự động nhận diện toàn bộ GPU và cấu hình DataParallel song song
    model, device, num_gpus, effective_batch_size, effective_lr = setup_multi_gpu(
        model=base_model,
        batch_size_per_gpu=args.batch_size,
        base_lr=args.lr,
        device_arg=args.device,
    )

    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=effective_batch_size, shuffle=False)
    print(f"✅ Đã nạp {len(train_ds)} mẫu train ({args.few_shot_ratio*100:.0f}%) và {len(test_ds)} mẫu test.")

    criterion = nn.SmoothL1Loss(beta=1.0)
    raw_model = unwrap_model(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, raw_model.parameters()), lr=effective_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Khôi phục từ checkpoint nếu có cờ --resume
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

    # 5. Training Loop
    best_mae = float("inf")
    history = {"epochs": [], "loss": [], "mae_total": [], "mae_bike": [], "mae_car": []}
    print(f"\n🏁 [Train] Bắt đầu huấn luyện từ Epoch [{start_epoch+1}/{args.epochs}]...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for batch in pbar:
            targets = batch["counts"].to(device)
            rgb = batch["rgb"].to(device)
            delta = batch.get("delta")
            if delta is not None:
                delta = delta.to(device)
            preds = model(rgb, delta=delta)

            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # Đánh giá sau mỗi epoch
        metrics = evaluate_mae(model, test_loader, device, mode=args.mode)
        print(f"📊 Epoch [{epoch+1}/{args.epochs}] — Train Loss: {avg_train_loss:.4f} | Test MAE [Bike: {metrics['mae_bike']:.2f}, Car: {metrics['mae_car']:.2f}, Total: {metrics['mae_total']:.2f}]")

        history["epochs"].append(epoch + 1)
        history["loss"].append(float(avg_train_loss))
        history["mae_total"].append(float(metrics["mae_total"]))
        history["mae_bike"].append(float(metrics["mae_bike"]))
        history["mae_car"].append(float(metrics["mae_car"]))

        if metrics["mae_total"] < best_mae:
            best_mae = metrics["mae_total"]
            ckpt_path = os.path.join(args.save_dir, f"best_counting_model_{args.mode}.pth")
            save_checkpoint(
                save_path=ckpt_path,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics=metrics,
                extra_dict={"num_gpus": num_gpus, "args": vars(args)},
                verbose=False,
            )

        # Lưu Last Checkpoint mỗi epoch
        last_ckpt_path = os.path.join(args.save_dir, "last_checkpoint.pth")
        save_checkpoint(
            save_path=last_ckpt_path,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            metrics=metrics,
            extra_dict={"num_gpus": num_gpus, "args": vars(args)},
            verbose=False,
        )

    # Lưu metrics JSON
    metrics_path = os.path.join(args.save_dir, "training_metrics.json")
    try:
        import json
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
        ax1.plot(history["epochs"], history["loss"], "b-o", linewidth=2, label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Hàm mất mát huấn luyện (Smooth L1)", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        ax2.plot(history["epochs"], history["mae_total"], "r-s", linewidth=2, label="Total MAE")
        ax2.plot(history["epochs"], history["mae_bike"], "g--^", linewidth=1.5, label="Bike MAE")
        ax2.plot(history["epochs"], history["mae_car"], "m-.d", linewidth=1.5, label="Car MAE")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE (Sai số xe)")
        ax2.set_title(f"Sai số đếm xe trên tập kiểm thử ({args.mode})", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        loss_curve_path = os.path.join(args.save_dir, "loss_curve.png")
        plt.savefig(loss_curve_path, bbox_inches="tight")
        plt.close()
        print(f"📈 [Charts] Đã lưu biểu đồ sai số tại: {loss_curve_path}")
    except Exception as e_plot:
        print(f"⚠️ [Chart Warning] {e_plot}")

    # Xuất ảnh trực quan hóa PCA Feature Map & Kết quả đếm
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print("\n🎨 [Visualization] Đang xuất ảnh trực quan hóa dự báo số lượng xe...")
        test_batch = next(iter(test_loader))
        val_rgb = test_batch["rgb"][:1].to(device)
        val_delta = test_batch.get("delta")
        if val_delta is not None:
            val_delta = val_delta[:1].to(device)
        with torch.no_grad():
            val_preds = model(val_rgb, delta=val_delta)[0].cpu().numpy()
        val_gt = test_batch["counts"][0].cpu().numpy()

        img_np = val_rgb[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0.0, 1.0)

        # Trích xuất PCA tokens từ backbone
        backbone_m = raw_model.backbone
        backbone_m.eval()
        with torch.no_grad():
            feat_out = backbone_m(val_rgb)
            if isinstance(feat_out, dict):
                tokens = feat_out.get("x_norm_patchtokens", feat_out.get("patch_tokens"))
            elif hasattr(backbone_m, "get_intermediate_layers"):
                layers = backbone_m.get_intermediate_layers(val_rgb, n=1, return_class_token=True)
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

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)
        axes[0].imshow(img_np)
        axes[0].set_title("1. Ảnh Giao Thông Đầu Vào", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        if val_delta is not None:
            delta_np = val_delta[0, 0].cpu().numpy()
            axes[1].imshow(delta_np, cmap="inferno")
            axes[1].set_title("2. Kênh Sai Khác Tiền Cảnh Δ", fontsize=11, fontweight="bold")
        else:
            axes[1].imshow(img_np)
            axes[1].set_title("2. RGB Không Δ", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(pca_rgb)
        axes[2].set_title("3. DINOv3 PCA Feature Map", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        categories = ["Xe Máy", "Ô Tô", "Tổng"]
        x_pos = np.arange(len(categories))
        width = 0.35
        axes[3].bar(x_pos - width/2, val_gt, width, label="Thực tế (GT)", color="steelblue")
        axes[3].bar(x_pos + width/2, val_preds, width, label="AI Dự Báo", color="coral")
        axes[3].set_xticks(x_pos)
        axes[3].set_xticklabels(categories, fontsize=10, fontweight="bold")
        axes[3].set_ylabel("Số lượng xe")
        axes[3].set_title("4. Đối Chiếu Số Lượng Đếm", fontsize=11, fontweight="bold")
        axes[3].grid(True, linestyle="--", alpha=0.5, axis="y")
        axes[3].legend()

        plt.suptitle(f"Tiến Trình Đếm Xe Hướng 3 ({args.mode}) — Best MAE: {best_mae:.2f}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        vis_path = os.path.join(args.save_dir, "counting_progress.png")
        plt.savefig(vis_path, bbox_inches="tight")
        plt.close()
        print(f"🖼️ [Visualization] Đã lưu ảnh đối chiếu đếm xe tại: {vis_path}")
    except Exception as e_vis:
        print(f"💡 [Visualization Notice] {e_vis}")

    print(f"\n🎉 [Complete] Huấn luyện hoàn tất! Kỷ lục MAE Total: {best_mae:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Foreground-Enhanced Vehicle Counting")
    parser.add_argument("--csv_file", type=str, default="stage1_perception/counting_labels_5012.csv")
    parser.add_argument("--origin_dir", "--data_dir", dest="origin_dir", type=str, default="output")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds")
    parser.add_argument("--save_dir", "--output_dir", dest="save_dir", type=str, default="checkpoints/direction3_fg_counting")
    parser.add_argument("--match_strategy", type=str, default="route_hourly")
    parser.add_argument("--backbone", "--model_name", dest="backbone", type=str, default="dinov3_vits16")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn trọng số khởi tạo backbone ban đầu")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn file checkpoint (.pth) để tiếp tục huấn luyện")
    parser.add_argument("--mode", type=str, default="4channel", choices=["4channel", "spatial_attention"])
    parser.add_argument("--few_shot_ratio", type=float, default=1.0, help="Tỷ lệ nhãn huấn luyện (0.05, 0.1, 0.2, 1.0)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face user access token cho các mô hình có quyền truy cập đóng (Meta DINOv3)")
    parsed, unknown = parser.parse_known_args()
    if unknown:
        print(f"⚠️ [CLI Warning] Bỏ qua các đối số chưa khai báo: {unknown}")
    if parsed.hf_token:
        os.environ["HF_TOKEN"] = parsed.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = parsed.hf_token
    return parsed


if __name__ == "__main__":
    args = parse_args()
    train_fg_counting(args)
