"""
=============================================================================
 Downstream Vehicle Counting Fine-Tuning with DINOv2
 Supports:
   1. Official Meta DINOv2 (Off-the-shelf Foundation Baseline)
   2. Traffic SSL Domain-Adapted DINOv2 (Pre-trained via train_ssl_dino.py)
   3. Few-Shot Evaluation Protocol (1%, 5%, 10%, 20%, 100% labelled data)
   4. Linear Probing vs Full Backbone Fine-Tuning
=============================================================================
"""

import argparse
import copy
import math
import os
import random
import sys
import time
import warnings
from typing import Dict, List, Tuple

# Suppress harmless third-party notices
warnings.filterwarnings("ignore", category=UserWarning, message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =====================================================================
# 1. DATASET LOADER FOR VEHICLE COUNTING (AUTO-COLUMN DISCOVERY)
# =====================================================================

class VehicleCountingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_name = str(row["filename"])
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.exists(img_path):
            # Try recursive search if not in root
            alt = glob.glob(os.path.join(self.image_dir, "**", img_name), recursive=True)
            if alt:
                img_path = alt[0]
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform:
                x = self.transform(img)
            else:
                x = transforms.ToTensor()(img)

        # Targets: [car, motorcycle]
        target = torch.tensor([float(row["car"]), float(row["motorcycle"])], dtype=torch.float32)
        return x, target


# =====================================================================
# 2. DINOv2 COUNTING MODEL ARCHITECTURE
# =====================================================================

class DINOv2CountingModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        pretrained_weights: str = None,
        freeze_backbone: bool = False,
        head_hidden_dim: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone

        print(f"📥 Loading DINOv2 backbone: '{backbone_name}'...")
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name, pretrained=True)
        embed_dim = getattr(self.backbone, "embed_dim", 384)

        # Apply SSL Domain-Adapted Weights if provided
        if pretrained_weights and os.path.exists(pretrained_weights):
            print(f"🔄 Loading Traffic-Adapted SSL weights from: {pretrained_weights}")
            state = torch.load(pretrained_weights, map_location="cpu")
            if "student" in state:
                state = {k.replace("0.", ""): v for k, v in state["student"].items() if k.startswith("0.")}
            self.backbone.load_state_dict(state, strict=False)
            print("✅ Domain-adapted SSL weights loaded successfully!")

        if freeze_backbone:
            print("🧊 Freezing DINOv2 Backbone (Linear Probing Mode)...")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2-layer MLP Counting Regression Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(head_hidden_dim, num_classes),
            nn.ReLU(),  # Vehicle counts are strictly non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        out = self.head(feat)
        return out


# =====================================================================
# 3. TRAINING & EVALUATION FUNCTIONS
# =====================================================================

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Car Metrics
    car_mae = mean_absolute_error(targets[:, 0], preds[:, 0])
    car_rmse = np.sqrt(mean_squared_error(targets[:, 0], preds[:, 0]))
    car_r2 = r2_score(targets[:, 0], preds[:, 0])

    # Motorcycle Metrics
    moto_mae = mean_absolute_error(targets[:, 1], preds[:, 1])
    moto_rmse = np.sqrt(mean_squared_error(targets[:, 1], preds[:, 1]))
    moto_r2 = r2_score(targets[:, 1], preds[:, 1])

    # Overall Combined Metrics
    tot_mae = mean_absolute_error(targets.sum(axis=1), preds.sum(axis=1))
    tot_rmse = np.sqrt(mean_squared_error(targets.sum(axis=1), preds.sum(axis=1)))

    return {
        "car_mae": car_mae,
        "car_rmse": car_rmse,
        "car_r2": car_r2,
        "moto_mae": moto_mae,
        "moto_rmse": moto_rmse,
        "moto_r2": moto_r2,
        "tot_mae": tot_mae,
        "tot_rmse": tot_rmse,
    }


def train_counting_dinov2(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    print("\n" + "=" * 70)
    print(" 🚗 DINOv2 VEHICLE COUNTING DOWNSTREAM FINE-TUNING")
    print("=" * 70)
    print(f" Backbone            : {args.backbone}")
    print(f" SSL Adapted Weights : {args.ssl_weights if args.ssl_weights else 'None (Off-the-shelf Meta DINOv2)'}")
    print(f" Freeze Backbone     : {args.freeze_backbone} ({'Linear Probe' if args.freeze_backbone else 'End-to-End Fine-Tune'})")
    print(f" Few-Shot Ratio      : {args.few_shot_ratio * 100:.1f}%")
    print(f" Batch Size          : {args.batch_size}")
    print(f" Epochs              : {args.epochs}")
    print(f" Learning Rate       : {args.lr}")
    print("=" * 70)

    # 1. Load Annotations
    df = pd.read_csv(args.csv_file)
    print(f"   Loaded total dataset: {len(df)} labelled instances.")

    # Standardize column names
    col_map = {}
    for c in df.columns:
        if c.lower() in ["filename", "file_name", "image", "image_name", "img"]:
            col_map[c] = "filename"
        elif c.lower() in ["car", "cars", "oto", "car_count"]:
            col_map[c] = "car"
        elif c.lower() in ["motorcycle", "motorcycles", "motorbike", "moto", "bike", "xemay"]:
            col_map[c] = "motorcycle"
    df = df.rename(columns=col_map)

    # Train / Val / Test split (70 / 15 / 15)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=args.seed)
    train_df, val_df = train_test_split(train_df, test_size=0.176, random_state=args.seed)

    # Few-shot subsampling if specified
    if args.few_shot_ratio < 1.0:
        n_samples = max(10, int(len(train_df) * args.few_shot_ratio))
        train_df = train_df.sample(n=n_samples, random_state=args.seed).reset_index(drop=True)
        print(f"   [Few-Shot Protocol] Using {len(train_df)} training samples ({args.few_shot_ratio*100:.1f}% of train split).")
    else:
        print(f"   [Full Protocol] Using {len(train_df)} training samples (100% of train split).")

    print(f"   Validation: {len(val_df)} | Test: {len(test_df)}")

    # 2. Transforms (Divisible by 14: 224x224)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        VehicleCountingDataset(train_df, args.image_dir, transform=train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        VehicleCountingDataset(val_df, args.image_dir, transform=val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        VehicleCountingDataset(test_df, args.image_dir, transform=val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 3. Model & Optimizer
    model = DINOv2CountingModel(
        backbone_name=args.backbone,
        pretrained_weights=args.ssl_weights,
        freeze_backbone=args.freeze_backbone,
        head_hidden_dim=256,
        num_classes=2,
    ).to(device)

    # Optimizer with differential LR if fine-tuning end-to-end
    if args.freeze_backbone:
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr * 0.1, "weight_decay": 1e-4},
            {"params": model.head.parameters(), "lr": args.lr, "weight_decay": 1e-4},
        ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.SmoothL1Loss()  # Huber loss for robust counting regression

    best_val_mae = float("inf")
    best_weights = None

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        curr_val_mae = val_metrics["tot_mae"]

        if curr_val_mae < best_val_mae:
            best_val_mae = curr_val_mae
            best_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            print(
                f"Ep {epoch+1:03d}/{args.epochs:03d} | Train Loss: {train_loss:.4f} | "
                f"Val Car MAE: {val_metrics['car_mae']:.3f} | Moto MAE: {val_metrics['moto_mae']:.3f} | "
                f"Best Tot MAE: {best_val_mae:.3f}"
            )

    # 5. Final Evaluation on Unseen Test Split
    print("\n" + "=" * 70)
    print(" 🏁 FINAL TEST EVALUATION (BEST CHECKPOINT)")
    print("=" * 70)
    model.load_state_dict(best_weights)
    test_metrics = evaluate(model, test_loader, device)

    print(f" 🚗 CARS       : MAE = {test_metrics['car_mae']:.4f} | RMSE = {test_metrics['car_rmse']:.4f} | R2 = {test_metrics['car_r2']:.4f}")
    print(f" 🛵 MOTORCYCLES: MAE = {test_metrics['moto_mae']:.4f} | RMSE = {test_metrics['moto_rmse']:.4f} | R2 = {test_metrics['moto_r2']:.4f}")
    print(f" 📊 TOTAL FLOW : MAE = {test_metrics['tot_mae']:.4f} | RMSE = {test_metrics['tot_rmse']:.4f}")
    print("=" * 70 + "\n")

    # Save Best Model Checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"dinov2_counting_ratio_{int(args.few_shot_ratio*100)}.pth")
    torch.save(best_weights, save_path)
    print(f"💾 Checkpoint saved to: {save_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Downstream Vehicle Counting with DINOv2")
    parser.add_argument("--csv_file", type=str, default="/workspace/traffic_update.csv", help="Path to counting CSV")
    parser.add_argument("--image_dir", type=str, default="/workspace/images", help="Path to camera images")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14", help="DINOv2 backbone (dinov2_vits14, dinov2_vitb14)")
    parser.add_argument("--ssl_weights", type=str, default=None, help="Path to SSL domain-adapted weights (from train_ssl_dino.py)")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone for linear probing")
    parser.add_argument("--few_shot_ratio", type=float, default=1.0, help="Ratio of labelled training data (0.01 to 1.0)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints/dinov2_counting", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_counting_dinov2(args)


if __name__ == "__main__":
    main()
