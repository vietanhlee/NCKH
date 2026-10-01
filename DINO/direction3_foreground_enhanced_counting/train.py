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
            if mode == "4channel":
                inputs = batch["input_4ch"].to(device)
                preds = model(inputs)
            else:
                inputs = batch["rgb"].to(device)
                delta = batch["delta"].to(device)
                preds = model(inputs, delta=delta)

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

    # 4. Training Loop
    best_mae = float("inf")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for batch in pbar:
            targets = batch["counts"].to(device)
            if args.mode == "4channel":
                inputs = batch["input_4ch"].to(device)
                preds = model(inputs)
            else:
                inputs = batch["rgb"].to(device)
                delta = batch["delta"].to(device)
                preds = model(inputs, delta=delta)

            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Đánh giá sau mỗi epoch
        metrics = evaluate_mae(model, test_loader, device, mode=args.mode)
        print(f"📊 Epoch [{epoch+1}/{args.epochs}] — Test MAE [Bike: {metrics['mae_bike']:.2f}, Car: {metrics['mae_car']:.2f}, Total: {metrics['mae_total']:.2f}]")

        if metrics["mae_total"] < best_mae:
            best_mae = metrics["mae_total"]
            ckpt_path = os.path.join(args.save_dir, f"best_counting_model_{args.mode}.pth")
            save_checkpoint(
                save_path=ckpt_path,
                model=raw_model,
                epoch=epoch + 1,
                metrics=metrics,
                extra_dict={"num_gpus": num_gpus, "args": vars(args)},
                verbose=True,
            )

    print(f"\n🎉 [Complete] Huấn luyện hoàn tất! Kỷ lục MAE Total: {best_mae:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Foreground-Enhanced Vehicle Counting")
    parser.add_argument("--csv_file", type=str, default="stage1_perception/counting_labels_5012.csv")
    parser.add_argument("--origin_dir", type=str, default="output")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds")
    parser.add_argument("--save_dir", type=str, default="checkpoints/direction3_fg_counting")
    parser.add_argument("--match_strategy", type=str, default="route_hourly")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--mode", type=str, default="4channel", choices=["4channel", "spatial_attention"])
    parser.add_argument("--few_shot_ratio", type=float, default=1.0, help="Tỷ lệ nhãn huấn luyện (0.05, 0.1, 0.2, 1.0)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fg_counting(args)
