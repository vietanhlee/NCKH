"""
=============================================================================
 Hướng 3: Scene Decomposition — Training Script
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
from common.gpu_utils import setup_multi_gpu, unwrap_model, save_checkpoint
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
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
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

    # 1. Khởi tạo Dataset
    dataset = DecompositionDataset(
        bg_dir=args.bg_dir,
        origin_dir=args.origin_dir,
        match_strategy=args.match_strategy,
        img_size=args.img_size,
        is_train=True,
        max_samples=args.max_samples,
    )
    print(f"✅ [Data] Đã nạp {len(dataset)} mẫu huấn luyện.")

    # 2. Khởi tạo Mô hình & Tự động cấu hình Toàn bộ GPU (Multi-GPU Engine)
    backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        pretrained=True,
        device=args.device,
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
        lambda_bg=args.lambda_bg,
        lambda_sparse=args.lambda_sparse,
        lambda_tv=args.lambda_tv,
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
    elif args.weights:
        if os.path.isfile(args.weights):
            print(f"\n📦 [Weights] Nạp trọng số khởi tạo ban đầu: {args.weights}")
            load_checkpoint(load_path=args.weights, model=raw_model, device=device, strict=False, verbose=True)

    # 5. Vòng lặp huấn luyện
    best_loss = float("inf")
    print(f"\n🏁 [Train] Bắt đầu huấn luyện từ Epoch [{start_epoch+1}/{args.epochs}]...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for batch in pbar:
            origin = batch["origin"].to(device)
            bg = batch["bg"].to(device)

            preds = model(origin)
            targets = {"origin": origin, "bg": bg}

            loss, loss_dict = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{loss_dict['loss_recon']:.4f}",
                "bg": f"{loss_dict['loss_bg']:.4f}",
            })

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"📊 Epoch [{epoch+1}/{args.epochs}] — Loss TB: {avg_loss:.4f}")

        # Xuất ảnh trực quan kiểm tra
        with torch.no_grad():
            sample = next(iter(loader))
            s_origin = sample["origin"][:1].to(device)
            s_bg = sample["bg"][:1].to(device)
            s_preds = model(s_origin)
            save_visual_sample(
                s_origin, s_bg,
                s_preds["pred_bg"], s_preds["pred_fg"], s_preds["pred_mask"], s_preds["recon_origin"],
                save_path=os.path.join(vis_dir, f"epoch_{epoch+1:03d}.png"),
                epoch=epoch + 1,
            )

        # Lưu Checkpoint
        if avg_loss < best_loss or (epoch + 1) == args.epochs:
            best_loss = avg_loss
            ckpt_path = os.path.join(args.save_dir, "best_decomposition_model.pth")
            save_checkpoint(
                save_path=ckpt_path,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics={"loss": avg_loss},
                extra_dict={"num_gpus": num_gpus, "args": vars(args)},
                verbose=True,
            )

    print("\n🎉 [Complete] Huấn luyện Scene Decomposition thành công!")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Traffic Scene Decomposition")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục background")
    parser.add_argument("--origin_dir", type=str, default="output", help="Thư mục origin")
    parser.add_argument("--save_dir", type=str, default="checkpoints/direction2_scene_decomp", help="Thư mục lưu")
    parser.add_argument("--match_strategy", type=str, default="route_hourly")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn file trọng số khởi tạo ban đầu")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn file checkpoint (.pth) để tiếp tục huấn luyện")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_bg", type=float, default=1.5, help="Trọng số giám sát nền")
    parser.add_argument("--lambda_sparse", type=float, default=0.05, help="Trọng số mask sparsity")
    parser.add_argument("--lambda_tv", type=float, default=0.1, help="Trọng số Total Variation")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_decomposition(args)
