"""
=============================================================================
 Hướng 8: Self-Supervised Road Surface Condition Estimation
 Module: Train Pipeline (Quy trình huấn luyện ước lượng điều kiện mặt đường)
 Hỗ trợ Multi-GPU, Mixed Precision (AMP), Resume Training chuẩn Production
=============================================================================
"""

import os
import sys
import argparse
import time
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Đảm bảo nạp đúng thư mục gốc DINO
_dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)

from common.backbone_loader import get_dino_backbone
from common.gpu_utils import setup_multi_gpu, unwrap_model, save_checkpoint, load_checkpoint
from direction8_road_condition.dataset import RoadSurfaceDataset
from direction8_road_condition.models import RoadConditionClassifier
from direction8_road_condition.losses import SurfaceConsistencyLoss


def train_road_condition(args):
    """Quy trình huấn luyện mạng RoadConditionClassifier."""
    os.makedirs(args.save_dir, exist_ok=True)
    device_arg = args.device

    print("=" * 80)
    print(" 🚀 [Direction 8] KHỞI CHẠY HUẤN LUYỆN ROAD SURFACE CONDITION ESTIMATION")
    print("=" * 80)

    # 1. Khởi tạo Dataset & DataLoader
    dataset = RoadSurfaceDataset(
        bg_dir=args.bg_dir,
        img_size=args.img_size,
        max_samples=args.max_samples,
    )

    if len(dataset) == 0:
        print("❌ [Lỗi] Không tìm thấy ảnh nền nào trong thư mục chỉ định.")
        return

    # 2. Khởi tạo Mô hình & Multi-GPU
    backbone, embed_dim, _ = get_dino_backbone(
        model_name=args.backbone,
        pretrained=True,
        device=device_arg,
    )

    base_model = RoadConditionClassifier(
        backbone=backbone,
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        freeze_backbone=args.freeze_backbone,
    )

    model, device, num_gpus, effective_batch_size, effective_lr = setup_multi_gpu(
        model=base_model,
        batch_size_per_gpu=args.batch_size,
        base_lr=args.lr,
        device_arg=device_arg,
    )

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(dataset) > effective_batch_size),
    )

    # 3. Hàm mất mát, Optimizer & Scheduler
    loss_fn = SurfaceConsistencyLoss(
        weight_illum=args.weight_illum,
        weight_wetness=args.weight_wetness,
        weight_degradation=args.weight_degradation,
    ).to(device)

    raw_model = unwrap_model(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, raw_model.parameters()),
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

    # 5. Vòng lặp huấn luyện
    best_loss = float("inf")
    start_time = time.time()
    print(f"\n🏁 [Train] Bắt đầu huấn luyện từ Epoch [{start_epoch+1}/{args.epochs}] trên {max(1, num_gpus)} thiết bị...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            targets = {
                "illum_class": batch["illum_class"].to(device, non_blocking=True),
                "specular_ratio": batch["specular_ratio"].to(device, non_blocking=True),
                "roughness": batch["roughness"].to(device, non_blocking=True),
            }

            optimizer.zero_grad()

            if scaler.is_enabled():
                with torch.amp.autocast(device_type=device_type):
                    preds = model(images)
                    loss_dict = loss_fn(preds, targets)
                    loss = loss_dict["loss_total"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(images)
                loss_dict = loss_fn(preds, targets)
                loss = loss_dict["loss_total"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=2.0)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "illum": f"{loss_dict['loss_illum'].item():.4f}",
                "wet": f"{loss_dict['loss_wetness'].item():.4f}",
                "deg": f"{loss_dict['loss_degradation'].item():.4f}",
            })

        scheduler.step()
        avg_loss = total_loss / max(1, len(loader))
        print(f"📊 Epoch [{epoch+1}/{args.epochs}] — Loss TB: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Lưu checkpoint tốt nhất và checkpoint định kỳ
        if avg_loss < best_loss or (epoch + 1) == args.epochs:
            best_loss = min(best_loss, avg_loss)
            ckpt_path = os.path.join(args.save_dir, "best_road_condition_model.pth")
            save_checkpoint(
                save_path=ckpt_path,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                metrics={"loss": avg_loss},
                extra_dict={"best_loss": best_loss, "num_gpus": num_gpus, "args": vars(args)},
                verbose=True,
            )

    elapsed = time.time() - start_time
    print(f"\n🎉 [Complete] Huấn luyện Road Condition hoàn tất sau {elapsed/60:.2f} phút! Best Loss: {best_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Road Surface Condition Estimation")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục background")
    parser.add_argument("--save_dir", type=str, default="checkpoints/direction8_road_condition", help="Thư mục lưu checkpoint")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone DINO")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn trọng số khởi tạo ban đầu")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn file checkpoint (.pth) để tiếp tục huấn luyện")
    parser.add_argument("--img_size", type=int, default=224, help="Kích thước ảnh")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Số chiều lớp ẩn các head")
    parser.add_argument("--epochs", type=int, default=15, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size mỗi GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--weight_illum", type=float, default=1.0, help="Trọng số loss chiếu sáng")
    parser.add_argument("--weight_wetness", type=float, default=2.0, help="Trọng số loss độ ẩm/ngập nước")
    parser.add_argument("--weight_degradation", type=float, default=1.0, help="Trọng số loss suy giảm mặt đường")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Đóng băng backbone ViT")
    parser.add_argument("--num_workers", type=int, default=0, help="Số worker nạp dữ liệu")
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn số mẫu thử nghiệm nhanh")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị ('cuda' hoặc 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_road_condition(args)
