"""
=============================================================================
 Hướng 8: Self-Supervised Road Surface Condition Estimation
 Module: Evaluation & Analysis Pipeline (Đánh giá & Gom cụm mặt đường toàn đô thị)
=============================================================================
"""

import argparse
import os
import sys
import logging
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader

_dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)

from common.backbone_loader import get_dino_backbone
from dataset import RoadSurfaceDataset
from models import RoadConditionClassifier


def setup_logger():
    logger = logging.getLogger("RoadCondition")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    return logger


def plot_road_condition_pca(
    embeddings_np: np.ndarray,
    illum_labels: np.ndarray,
    wetness_scores: np.ndarray,
    save_path: str,
):
    """Vẽ biểu đồ phân bố không gian đặc trưng mặt đường DINO qua PCA 2D."""
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings_np)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Tô màu theo điều kiện chiếu sáng
    scatter1 = axes[0].scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=illum_labels, cmap="viridis", alpha=0.8, edgecolors="none", s=35
    )
    axes[0].set_title(f"Phân Tách Chiếu Sáng (PCA 2D)\nGiải thích: {pca.explained_variance_ratio_.sum()*100:.1f}% phương sai", fontsize=11)
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_ticks([0, 1, 2])
    cbar1.set_ticklabels(["Đêm", "Chạng vạng", "Ngày"])

    # 2. Tô màu theo chỉ số ẩm ướt / đọng nước (Wetness Index)
    scatter2 = axes[1].scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=wetness_scores, cmap="coolwarm", alpha=0.8, edgecolors="none", s=35
    )
    axes[1].set_title("Chỉ Số Ẩm Ướt Mặt Đường (Wetness Score)", fontsize=11)
    axes[1].set_xlabel("Principal Component 1")
    axes[1].set_ylabel("Principal Component 2")
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label("Wetness Index [0 (Khô) -> 1 (Ướt)]")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Citywide Road Surface Condition Evaluation")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục background")
    parser.add_argument("--output_dir", type=str, default="checkpoints/direction5_road_condition", help="Thư mục xuất kết quả")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone DINO")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn số mẫu đánh giá")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán")
    args = parser.parse_args()

    logger = setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    logger.info(f"Khởi chạy đánh giá tình trạng mặt đường trên thiết bị: {device}")

    # 1. Dataset & DataLoader
    dataset = RoadSurfaceDataset(bg_dir=args.bg_dir, max_samples=args.max_samples)
    if len(dataset) == 0:
        logger.error("Không tìm thấy ảnh background trong thư mục chỉ định.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if sys.platform == "win32" else 2,
    )

    # 2. Tải ViT Backbone & Model
    logger.info(f"Tải ViT Backbone: {args.backbone}...")
    backbone, embed_dim, _ = get_dino_backbone(model_name=args.backbone, pretrained=True, device=device)
    model = RoadConditionClassifier(backbone=backbone, embed_dim=embed_dim, freeze_backbone=True).to(device)
    model.eval()

    # 3. Chạy suy luận trên toàn bộ ảnh nền
    logger.info(f"Đang phân tích {len(dataset)} ảnh nền từ mạng lưới camera...")

    all_embeddings = []
    all_wetness = []
    all_illum_preds = []
    all_degradation = []
    all_records = []

    illum_names = {0: "Đêm", 1: "Chạng vạng", 2: "Ngày"}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Road Surfaces"):
            imgs = batch["image"].to(device)
            out = model(imgs)

            emb = out["embedding_norm"].cpu().numpy()
            wet = out["pred_wetness"].cpu().numpy()
            illum_logits = out["logits_illum"]
            illum_pred = torch.argmax(illum_logits, dim=-1).cpu().numpy()
            deg = out["pred_degradation"].cpu().numpy()

            all_embeddings.append(emb)
            all_wetness.append(wet)
            all_illum_preds.append(illum_pred)
            all_degradation.append(deg)

            for i in range(len(imgs)):
                all_records.append({
                    "cam_id": batch["cam_id"][i],
                    "slot_h": int(batch["slot_h"][i]),
                    "wetness_score": float(wet[i]),
                    "is_wet": bool(wet[i] > 0.4),
                    "illumination": illum_names.get(int(illum_pred[i]), "Khác"),
                    "degradation_score": float(deg[i]),
                    "path": batch["path"][i],
                })

    all_embeddings_np = np.concatenate(all_embeddings, axis=0)
    all_wetness_np = np.concatenate(all_wetness, axis=0)
    all_illum_np = np.concatenate(all_illum_preds, axis=0)

    # 4. Vẽ biểu đồ gom cụm PCA
    chart_path = os.path.join(args.output_dir, "road_condition_clustering.png")
    plot_road_condition_pca(all_embeddings_np, all_illum_np, all_wetness_np, chart_path)
    logger.info(f"Đã lưu biểu đồ gom cụm không gian tại: {chart_path}")

    # 5. Xuất báo cáo CSV
    csv_path = os.path.join(args.output_dir, "road_surface_citywide_report.csv")
    df = pd.DataFrame(all_records)
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ Hoàn tất! Báo cáo tình trạng mặt đường thành phố đã được lưu tại: {csv_path}")


if __name__ == "__main__":
    main()
