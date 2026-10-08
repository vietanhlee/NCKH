"""
=============================================================================
 Hướng 2: Scene Decomposition — Training Script
 Huấn luyện mạng nơ-ron phân rã cảnh giao thông tự giám sát (Traffic-Decompose)
=============================================================================
"""

import argparse
import os
import sys

# Chống xung đột OpenMP trên Windows Anaconda và đảm bảo hiển thị ký tự tiếng Việt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.backbone_loader import get_dino_backbone
from common.gpu_utils import setup_multi_gpu, unwrap_model, save_checkpoint, load_checkpoint
from dataset import DecompositionDataset
from models import TrafficDecompositionNet
from losses import DecompositionLoss


def save_visual_sample(origin, bg, pred_bg, pred_fg, pred_mask, recon, save_path, epoch):
    """Xuất lưới 6 ảnh kiểm tra tiến độ học tách lớp."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)

    def to_img(t):
        arr = t.squeeze().detach().cpu().permute(1, 2, 0).numpy()
        return np.clip(arr, 0.0, 1.0)

    def to_mask(t):
        arr = t.squeeze().detach().cpu().numpy()
        return np.clip(arr, 0.0, 1.0)

    axes[0, 0].imshow(to_img(origin[0]))
    axes[0, 0].set_title("Input Origin Image", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(to_img(bg[0]))
    axes[0, 1].set_title("Ground-Truth Background", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(to_img(recon[0]))
    axes[0, 2].set_title("Reconstructed Origin", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(to_img(pred_bg[0]))
    axes[1, 0].set_title("Decomposed Background Layer", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(to_img(pred_fg[0]))
    axes[1, 1].set_title("Decomposed Foreground Layer", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(to_mask(pred_mask[0]), cmap="inferno")
    axes[1, 2].set_title("Predicted Vehicle Alpha Mask", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")

    plt.suptitle(f"Traffic-Decompose Layer Separation — Epoch {epoch}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def train_decomposition(args):
    torch.manual_seed(args.seed)
    if "cuda" in args.device.lower() and not torch.cuda.is_available():
        print("⚠️ [Cảnh Báo] CUDA không khả dụng trên môi trường hiện tại, tự động chuyển sang CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    vis_dir = os.path.join(args.save_dir, "visual_progress")
    os.makedirs(vis_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(" 🏙️ TRAINING TRAFFIC SCENE DECOMPOSITION NETWORK")
    print("=" * 78)
    print(f" Backbone Architecture : {args.backbone}")
    print(f" Image Resolution      : {args.img_size}x{args.img_size}")
    print(f" Epochs                : {args.epochs}")
    print(f" Batch Size            : {args.batch_size}")
    print(f" Learning Rate         : {args.lr}")
    print(f" Device                : {device}")
    print("=" * 78)

    # 1. Khởi tạo Dataset (Tự động phát hiện pseudo_bgs nếu chưa có bg_dir)
    bg_dir_to_use = args.bg_dir
    if not os.path.exists(bg_dir_to_use):
        pseudo_candidate = os.path.join("checkpoints", "direction2_scene_fit", "pseudo_bgs")
        if os.path.exists(pseudo_candidate):
            print(f"💡 [Data] Tự động phát hiện và liên kết ảnh nền pseudo-background từ: {pseudo_candidate}")
            bg_dir_to_use = pseudo_candidate

    dataset = DecompositionDataset(
        bg_dir=bg_dir_to_use,
        origin_dir=args.origin_dir,
        match_strategy=args.match_strategy,
        img_size=args.img_size,
        is_train=True,
        group_by_camera_slot=getattr(args, "group_by_camera_slot", False),
        max_samples=args.max_samples,
    )
    print(f"✅ [Data] Đã nạp {len(dataset)} mẫu huấn luyện.")

    # 2. Khởi tạo Mô hình & Tự động cấu hình Toàn bộ GPU (Multi-GPU Engine)
    backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        pretrained=True,
        device=args.device,
        weights_path=args.weights,
        hf_token=args.hf_token,
    )
    base_model = TrafficDecompositionNet(
        backbone=backbone,
        embed_dim=embed_dim,
        patch_size=patch_size,
        freeze_backbone=args.freeze_backbone,
    )

    # Tự động nhận diện toàn bộ GPU và cấu hình DataParallel
    model, device, num_gpus, effective_batch_size, effective_lr = setup_multi_gpu(
        model=base_model,
        batch_size_per_gpu=args.batch_size,
        base_lr=args.lr,
        device_arg=args.device,
    )

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # 3. Hàm mất mát & Optimizer
    loss_fn = DecompositionLoss(
        lambda_prior=getattr(args, "lambda_prior", getattr(args, "lambda_bg", 1.0)),
        lambda_shared=getattr(args, "lambda_shared", 1.0),
        lambda_excl=getattr(args, "lambda_excl", 0.5),
        lambda_tv=args.lambda_tv,
        lambda_sparse=args.lambda_sparse,
        lambda_bin=getattr(args, "lambda_bin", 0.05),
    )
    raw_model = unwrap_model(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, raw_model.parameters()),
        lr=effective_lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Khôi phục từ checkpoint nếu có cờ --resume hoặc nạp trọng số --weights
    start_epoch = 0
    best_loss = float("inf")
    history = {"epochs": [], "loss": [], "loss_rec": [], "loss_prior": []}

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Không tìm thấy file hoặc thư mục checkpoint resume: {args.resume}")
        print(f"\n🔄 [Resume] Khôi phục toàn bộ trạng thái huấn luyện từ checkpoint: {args.resume}")
        # Chú ý quan trọng: Truyền scheduler=None vào load_checkpoint để ngăn chặn bug của PyTorch
        # CosineAnnealingLR khi last_epoch >= T_max cũ làm mẫu số tiến về 0, khiến LR nổ tung (tăng gấp hàng nghìn lần).
        ckpt_data = load_checkpoint(
            load_path=args.resume,
            model=raw_model,
            optimizer=optimizer,
            scheduler=None,
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

        # Tái cấu hình Scheduler mượt mà cho số epochs còn lại
        remaining_epochs = max(1, args.epochs - start_epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        # Nếu LR cũ đã chạm đáy cực tiểu (1e-6) ở cuối session trước, thực hiện Cosine Warm-Restart mượt mà
        if current_lr <= 5e-6:
            restart_lr = effective_lr * 0.5  # Bắt đầu chu kỳ gia hạn với 50% base lr để tiếp tục hội tụ sâu
            for pg in optimizer.param_groups:
                pg["lr"] = restart_lr
            current_lr = restart_lr
            print(f"   🔄 [LR Warm-Restart] LR trước đó đã chạm đáy ({current_lr:.2e}), tự động khởi động mềm với LR = {current_lr:.6e} cho {remaining_epochs} epochs tiếp theo.")
        else:
            print(f"   🎯 [LR Continuity] Tiếp tục tốc độ học hiện tại: LR = {current_lr:.6e} cho {remaining_epochs} epochs còn lại.")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_epochs,
            eta_min=1e-6,
        )

        # Khôi phục kỷ lục best_loss trước đó (nếu có)
        if "metrics" in ckpt_data and isinstance(ckpt_data["metrics"], dict):
            saved_loss = ckpt_data["metrics"].get("loss", None)
            if saved_loss is not None:
                best_loss = float(saved_loss)
                print(f"   🏆 [Best Loss] Khôi phục kỷ lục loss tốt nhất trước đó: {best_loss:.4f}")

        # Khôi phục lịch sử huấn luyện từ metrics JSON nếu có để vẽ biểu đồ liền mạch
        metrics_candidates = [
            os.path.join(args.save_dir, "training_metrics.json"),
            os.path.join(os.path.dirname(args.resume) if os.path.isfile(args.resume) else args.resume, "training_metrics.json"),
        ]
        for mc in metrics_candidates:
            if os.path.isfile(mc):
                try:
                    with open(mc, "r", encoding="utf-8") as f_m:
                        old_data = json.load(f_m)
                        if "history" in old_data and isinstance(old_data["history"], dict):
                            history = old_data["history"]
                            print(f"   📈 [History] Đã khôi phục {len(history.get('epochs', []))} epochs lịch sử để tiếp nối biểu đồ.")
                            break
                except Exception:
                    pass
    elif args.weights:
        if os.path.isfile(args.weights):
            print(f"\n📦 [Weights] Nạp trọng số khởi tạo ban đầu: {args.weights}")
            load_checkpoint(load_path=args.weights, model=raw_model, device=device, strict=False, verbose=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Vòng lặp huấn luyện
    print(f"\n🏁 [Train] Bắt đầu huấn luyện từ Epoch [{start_epoch+1}/{args.epochs}]...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_rec = 0.0
        total_prior = 0.0
        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for batch in pbar:
            if "origins" in batch:
                origins = batch["origins"].to(device)
                bgs = batch["prior_bgs"].to(device)
                B, K, C, H, W = origins.shape
                origin = origins.view(B * K, C, H, W)
                bg = bgs.view(B * K, C, H, W)
            else:
                origin = batch["origin"].to(device)
                bg = batch["bg"].to(device)

            preds = model(origin, prior=bg)
            loss, loss_dict = loss_fn(
                preds, origin, bg,
                epoch=epoch,
                warmup_bin_epoch=getattr(args, "warmup_bin_epoch", 10),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_rec += loss_dict.get("loss_rec", 0.0)
            total_prior += loss_dict.get("loss_prior", 0.0)

            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "rec": f"{loss_dict.get('loss_rec', 0.0):.4f}",
                "prior": f"{loss_dict.get('loss_prior', 0.0):.4f}",
                "lr": f"{cur_lr:.2e}",
                "sigma": f"{loss_dict.get('mean_sigma', 0.0):.3f}",
            })

        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        avg_loss = total_loss / max(1, len(loader))
        avg_rec = total_rec / max(1, len(loader))
        avg_prior = total_prior / max(1, len(loader))

        history["epochs"].append(epoch + 1)
        history["loss"].append(float(avg_loss))
        history["loss_rec"].append(float(avg_rec))
        history["loss_prior"].append(float(avg_prior))

        print(f"📊 Epoch [{epoch+1}/{args.epochs}] — LR: {cur_lr:.6e} — Loss TB: {avg_loss:.4f} (Recon: {avg_rec:.4f}, Prior: {avg_prior:.4f})")

        # Xuất ảnh trực quan kiểm tra
        with torch.no_grad():
            sample = next(iter(loader))
            s_origin = sample["origin"][:1].to(device) if "origin" in sample else sample["origins"][:1, 0].to(device)
            s_bg = sample["bg"][:1].to(device) if "bg" in sample else sample["prior_bgs"][:1, 0].to(device)
            s_preds = model(s_origin, prior=s_bg)
            s_mask = s_preds.get("alpha_mask", s_preds.get("pred_mask"))
            progress_path = os.path.join(args.save_dir, "decomposition_progress.png")
            epoch_vis_path = os.path.join(vis_dir, f"epoch_{epoch+1:03d}.png")
            save_visual_sample(
                s_origin, s_bg,
                s_preds["pred_bg"], s_preds["pred_fg"], s_mask, s_preds["recon_origin"],
                save_path=epoch_vis_path,
                epoch=epoch + 1,
            )
            # Đồng thời cập nhật decomposition_progress.png mới nhất ở thư mục gốc
            save_visual_sample(
                s_origin, s_bg,
                s_preds["pred_bg"], s_preds["pred_fg"], s_mask, s_preds["recon_origin"],
                save_path=progress_path,
                epoch=epoch + 1,
            )

        # Lưu Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            for b_name in ["best_decomposition_model.pth", "best_checkpoint.pth"]:
                ckpt_path = os.path.join(args.save_dir, b_name)
                save_checkpoint(
                    save_path=ckpt_path,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    metrics={"loss": avg_loss, "loss_rec": avg_rec, "loss_prior": avg_prior},
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
            metrics={"loss": avg_loss, "loss_rec": avg_rec, "loss_prior": avg_prior},
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
                "best_loss": float(best_loss),
            }, f, indent=2, ensure_ascii=False)
        print(f"📊 [Metrics] Đã lưu lịch sử huấn luyện tại: {metrics_path}")
    except Exception as e_m:
        print(f"⚠️ [Metrics Warning] Không thể lưu JSON: {e_m}")

    # Vẽ biểu đồ Loss Curves
    try:
        plt.figure(figsize=(10, 4.5), dpi=150)
        plt.plot(history["epochs"], history["loss"], "b-o", linewidth=2, label="Total Loss")
        plt.plot(history["epochs"], history["loss_rec"], "r--s", linewidth=1.5, label="Reconstruction Loss")
        plt.plot(history["epochs"], history["loss_prior"], "g-.^", linewidth=1.5, label="Prior BG Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Tiến trình huấn luyện Traffic Scene Decomposition ({args.backbone})", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        loss_curve_path = os.path.join(args.save_dir, "loss_curve.png")
        plt.savefig(loss_curve_path, bbox_inches="tight")
        plt.close()
        print(f"📈 [Charts] Đã lưu biểu đồ hàm mất mát tại: {loss_curve_path}")
    except Exception as e_plot:
        print(f"⚠️ [Chart Warning] {e_plot}")

    print("\n🎉 [Complete] Huấn luyện Scene Decomposition thành công!")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Traffic Scene Decomposition")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục background")
    parser.add_argument("--origin_dir", "--data_dir", dest="origin_dir", type=str, default="output", help="Thư mục origin")
    parser.add_argument("--save_dir", "--output_dir", dest="save_dir", type=str, default="checkpoints/direction2_scene_decomp", help="Thư mục lưu")
    parser.add_argument("--match_strategy", type=str, default="route_hourly")
    parser.add_argument("--backbone", "--model_name", dest="backbone", type=str, default="dinov3_vits16")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn file trọng số khởi tạo ban đầu")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn file checkpoint (.pth) để tiếp tục huấn luyện")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_prior", "--lambda_bg", dest="lambda_prior", type=float, default=1.0, help="Trọng số giám sát nền prior (L_prior / lambda_bg)")
    parser.add_argument("--lambda_shared", type=float, default=1.0, help="Trọng số ràng buộc nền dùng chung (L_shared)")
    parser.add_argument("--lambda_excl", type=float, default=0.5, help="Trọng số loại trừ nền - tiền cảnh (L_excl)")
    parser.add_argument("--lambda_sparse", type=float, default=0.001, help="Trọng số mask sparsity (L_sparse)")
    parser.add_argument("--lambda_tv", type=float, default=0.01, help="Trọng số Total Variation (L_tv)")
    parser.add_argument("--lambda_bin", type=float, default=0.05, help="Trọng số nhị phân hóa mặt nạ (L_bin)")
    parser.add_argument("--warmup_bin_epoch", type=int, default=10, help="Epoch bắt đầu kích hoạt loss nhị phân hóa mặt nạ L_bin (mặc định: 10)")
    parser.add_argument("--group_by_camera_slot", action="store_true", default=False, help="Nhóm K-frame cùng trạm khác ngày để tính L_shared")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face user access token cho các mô hình có quyền truy cập đóng (Meta DINOv3)")
    parsed, unknown = parser.parse_known_args()
    if unknown:
        print(f"⚠️ [CLI Warning] Bỏ qua các đối số chưa khai báo: {unknown}")
    if parsed.hf_token:
        os.environ["HF_TOKEN"] = parsed.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = parsed.hf_token
    parsed.lambda_bg = parsed.lambda_prior  # Giữ alias cho thuộc tính
    return parsed


if __name__ == "__main__":
    args = parse_args()
    train_decomposition(args)
