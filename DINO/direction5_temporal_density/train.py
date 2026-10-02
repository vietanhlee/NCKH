"""
=============================================================================
 Hướng 5: Spatio-Temporal DINO for Continuous Road Space Occupancy 
          and Congestion Level of Service (LoS) Estimation
 Module: Train Pipeline (Quy trình huấn luyện và đánh giá chỉ số ùn tắc đô thị)
 Hỗ trợ Multi-GPU, Mixed Precision (AMP), Resume Training chuẩn Production
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
from common.gpu_utils import setup_multi_gpu, unwrap_model, save_checkpoint, load_checkpoint
from direction5_temporal_density.dataset import TemporalTrafficDataset
from direction5_temporal_density.models import SpatioTemporalDensityNet
from direction5_temporal_density.losses import SpatioTemporalDensityLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Spatio-Temporal Traffic Density & LoS Estimation")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục background")
    parser.add_argument("--origin_dir", type=str, default="output", help="Thư mục origin images")
    parser.add_argument("--csv_file", type=str, default=None, help="File CSV nhãn số lượng xe (nếu có)")
    parser.add_argument("--save_dir", type=str, default="checkpoints/direction5_temporal_density", help="Thư mục lưu mô hình")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone DINO")
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
    return parser.parse_args()


def train_temporal_density(args):
    """Quy trình huấn luyện toàn diện mạng SpatioTemporalDensityNet."""
    os.makedirs(args.save_dir, exist_ok=True)
    device_arg = args.device

    print("=" * 80)
    print(" 🚀 [Direction 5] KHỞI CHẠY HUẤN LUYỆN SPATIO-TEMPORAL DENSITY & LoS ESTIMATION")
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

        print(
            f"📊 Epoch [{epoch+1:02d}/{args.epochs:02d}] "
            f"Loss: {epoch_loss:.4f} | "
            f"Occupancy MAE: {epoch_mae:.4f} | "
            f"LoS Accuracy: {epoch_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Lưu checkpoint tốt nhất theo MAE Occupancy và lưu epoch cuối
        if epoch_mae < best_mae or (epoch + 1) == args.epochs:
            best_mae = min(best_mae, epoch_mae)
            ckpt_path = os.path.join(args.save_dir, "best_temporal_model.pth")
            save_checkpoint(
                save_path=ckpt_path,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                metrics={"loss": epoch_loss, "mae_occupancy": epoch_mae, "los_accuracy": epoch_acc},
                extra_dict={"best_mae": best_mae, "num_gpus": num_gpus, "args": vars(args)},
                verbose=True,
            )

    elapsed = time.time() - start_time
    print(f"\n🎉 [Complete] Huấn luyện Direction 5 hoàn tất sau {elapsed/60:.2f} phút! Kỷ lục MAE Occupancy: {best_mae:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train_temporal_density(args)
