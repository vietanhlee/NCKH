"""
=============================================================================
 Hướng 5: Temporal Contrastive Learning for Traffic Density Estimation
 Module: Train Pipeline (Quy trình huấn luyện tương phản chuỗi thời gian)
=============================================================================
"""

import argparse
import os
import sys
import time
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import project root
_dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)

from common.backbone_loader import get_dino_backbone
from common.gpu_utils import setup_device, print_gpu_summary, wrap_model_distributed, unwrap_model, save_checkpoint, load_checkpoint
from dataset import TemporalTrafficDataset
from models import TemporalTrafficEncoder
from losses import TemporalContrastiveLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Temporal Contrastive Learning cho Thị Giác Giao Thông")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục background")
    parser.add_argument("--origin_dir", type=str, default="output", help="Thư mục origin images")
    parser.add_argument("--csv_file", type=str, default=None, help="File CSV nhãn số lượng xe (nếu có)")
    parser.add_argument("--save_dir", type=str, default="checkpoints/direction5_temporal_density", help="Thư mục lưu mô hình")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone DINO")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn trọng số backbone ban đầu")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn file checkpoint (.pth) để tiếp tục huấn luyện")
    parser.add_argument("--window_size", type=int, default=4, help="Số khung hình trong một cửa sổ thời gian")
    parser.add_argument("--img_size", type=int, default=224, help="Kích thước ảnh đầu vào")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (số cửa sổ thời gian)")
    parser.add_argument("--epochs", type=int, default=20, help="Số epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, help="Nhiệt độ InfoNCE")
    parser.add_argument("--lambda_density", type=float, default=0.5, help="Trọng số loss mật độ")
    parser.add_argument("--max_sequences", type=int, default=None, help="Giới hạn số chuỗi kiểm thử nhanh")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_density = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for batch in pbar:
        rgb_a = batch["rgb_seq"].to(device, non_blocking=True)           # (B, K, 3, H, W)
        delta_a = batch["delta_seq"].to(device, non_blocking=True)       # (B, K, 1, H, W)
        rgb_b = batch["rgb_seq_pos"].to(device, non_blocking=True)       # (B, K, 3, H, W)
        delta_b = batch["delta_seq_pos"].to(device, non_blocking=True)   # (B, K, 1, H, W)
        targets = batch["density"].to(device, non_blocking=True)         # (B,)

        optimizer.zero_grad()

        use_amp = device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Forward cửa sổ A
            out_a = model(rgb_a, delta_a)
            # Forward cửa sổ B (cặp dương)
            out_b = model(rgb_b, delta_b)

            z_a = out_a["proj_contrastive"]
            z_b = out_b["proj_contrastive"]
            pred_density = out_a["pred_density"]

            loss_dict = criterion(
                z_a=z_a,
                z_b=z_b,
                pred_density=pred_density,
                target_density=targets,
            )
            loss = loss_dict["loss_total"]

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_contrastive += loss_dict["loss_contrastive"].item()
        total_density += loss_dict["loss_density"].item()
        num_batches += 1

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Contrast": f"{loss_dict['loss_contrastive'].item():.4f}",
        })

    avg_loss = total_loss / max(1, num_batches)
    avg_contrast = total_contrastive / max(1, num_batches)
    avg_density = total_density / max(1, num_batches)
    return {
        "loss_total": avg_loss,
        "loss_contrastive": avg_contrast,
        "loss_density": avg_density,
    }


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print_gpu_summary()

    print("=" * 80)
    print(" 🚀 [Direction 5] KHỞI CHẠY HUẤN LUYỆN TEMPORAL CONTRASTIVE TRAFFIC DENSITY")
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

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if sys.platform == "win32" else 2,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(dataset) > args.batch_size),
    )

    # 2. Tải ViT Backbone
    print(f"\n📦 Tải Backbone DINO: {args.backbone}...")
    backbone, embed_dim, _ = get_dino_backbone(
        model_name=args.backbone,
        pretrained=True,
        device=device,
    )

    # 3. Khởi tạo mô hình
    model = TemporalTrafficEncoder(
        backbone=backbone,
        embed_dim=embed_dim,
        delta_dim=128,
        temporal_dim=256,
        proj_dim=128,
        freeze_backbone=True,
    ).to(device)

    # Đa GPU nếu có
    model = wrap_model_distributed(model, device)

    # 4. Criterion, Optimizer & Scheduler
    criterion = TemporalContrastiveLoss(
        temperature=args.temperature,
        lambda_density=args.lambda_density,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device_type, enabled=(device.type == "cuda"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 5. Khôi phục từ checkpoint nếu có cờ --resume hoặc nạp --weights
    start_epoch = 0
    raw_model = unwrap_model(model)

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

    # 6. Huấn luyện
    best_loss = float("inf")
    start_time = time.time()
    print(f"\n🏁 [Train] Bắt đầu huấn luyện từ Epoch [{start_epoch+1}/{args.epochs}]...")

    for epoch in range(start_epoch, args.epochs):
        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )
        scheduler.step()

        print(
            f"Epoch [{epoch+1:02d}/{args.epochs:02d}] "
            f"Loss: {metrics['loss_total']:.4f} | "
            f"Contrastive: {metrics['loss_contrastive']:.4f} | "
            f"Density: {metrics['loss_density']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Lưu checkpoint tốt nhất và checkpoint định kỳ
        if metrics["loss_total"] < best_loss or (epoch + 1) == args.epochs:
            best_loss = min(best_loss, metrics["loss_total"])
            ckpt_path = os.path.join(args.save_dir, "best_temporal_model.pth")
            save_checkpoint(
                save_path=ckpt_path,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                metrics=metrics,
                extra_dict={"best_loss": best_loss, "args": vars(args)},
                verbose=True,
            )

    elapsed = time.time() - start_time
    print(f"\n🎉 Hoàn tất huấn luyện Direction 5 sau {elapsed/60:.2f} phút! Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
