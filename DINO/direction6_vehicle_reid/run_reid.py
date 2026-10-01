"""
=============================================================================
 Hướng 6: Delta-Guided Unsupervised Vehicle Re-Identification Across Cameras
 Module: Execution Pipeline (Quy trình trích xuất và tìm kiếm xe liên camera)
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
from tqdm.auto import tqdm
import torch
from torchvision import transforms

# Nạp thư mục gốc
_dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)

from common.matcher import TrafficPairMatcher
from common.backbone_loader import get_dino_backbone
from roi_extractor import DeltaRoIExtractor
from models import VehicleReIDModel
from matcher import VehicleReIDMatcher


def setup_logger():
    logger = logging.getLogger("VehicleReID")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    return logger


def save_retrieval_visualization(query_info: Dict[str, Any], top_results: List[Dict[str, Any]], save_path: str):
    """Vẽ ảnh trực quan hóa kết quả truy vấn phương tiện Top-5."""
    num_matches = len(top_results)
    fig, axes = plt.subplots(1, 1 + num_matches, figsize=(3 * (1 + num_matches), 4))

    # 1. Vẽ Query
    axes[0].imshow(query_info["crop_pil"])
    axes[0].set_title(f"Query\nCam: {query_info['cam_id']}", color="blue", fontsize=10)
    axes[0].axis("off")

    # 2. Vẽ Top Matches
    for idx, res in enumerate(top_results):
        ax = axes[1 + idx]
        crop = res["meta"]["crop_pil"]
        sim = res["similarity"]
        cam = res["meta"]["cam_id"]
        ax.imshow(crop)
        ax.set_title(f"Rank {res['rank']}\nSim: {sim:.3f}\nCam: {cam}", color="green" if sim > 0.6 else "black", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Delta-Guided Vehicle Re-Identification Pipeline")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục ảnh nền")
    parser.add_argument("--origin_dir", type=str, default="output", help="Thư mục ảnh origin")
    parser.add_argument("--output_dir", type=str, default="checkpoints/direction6_vehicle_reid", help="Thư mục xuất kết quả")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone DINO")
    parser.add_argument("--top_k", type=int, default=5, help="Số lượng kết quả xếp hạng cần truy vấn")
    parser.add_argument("--min_area", type=int, default=500, help="Diện tích tối thiểu để phát hiện xe")
    parser.add_argument("--max_pairs", type=int, default=50, help="Số lượng ảnh tối đa để trích xuất gallery")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán")
    args = parser.parse_args()

    logger = setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Khởi chạy Vehicle Re-ID trên thiết bị: {device}")

    # 1. Ghép cặp và trích xuất RoI phương tiện
    matcher = TrafficPairMatcher(bg_dir=args.bg_dir, origin_dir=args.origin_dir, match_strategy="route_hourly")
    all_pairs = matcher.discover_pairs()
    if not all_pairs:
        logger.error("Không tìm thấy cặp ảnh giao thông hợp lệ.")
        return

    pairs = all_pairs[:args.max_pairs]
    logger.info(f"Đang phân tích {len(pairs)} khung hình để trích xuất phương tiện bằng Delta-RoI...")

    roi_extractor = DeltaRoIExtractor(min_area=args.min_area, target_size=(256, 128))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    extracted_vehicles: List[Dict[str, Any]] = []
    for p in tqdm(pairs, desc="Extracting RoIs"):
        try:
            orig = Image.open(p["origin_path"]).convert("RGB")
            bg = Image.open(p["bg_path"]).convert("RGB")
            rois = roi_extractor.extract_rois(orig, bg)

            cam_id = p["origin_name"].split("_")[0] if "_" in p["origin_name"] else "unknown"
            for roi in rois:
                extracted_vehicles.append({
                    "cam_id": cam_id,
                    "origin_name": p["origin_name"],
                    "bbox": roi["bbox"],
                    "crop_pil": roi["crop_pil"],
                    "area": roi["area"],
                    "confidence": roi["confidence"],
                })
        except Exception as e:
            logger.warning(f"Lỗi khi xử lý {p['origin_name']}: {e}")

    logger.info(f"Đã trích xuất thành công {len(extracted_vehicles)} phương tiện từ các camera.")
    if len(extracted_vehicles) < 2:
        logger.warning("Không đủ số lượng phương tiện để thực hiện so khớp Re-ID.")
        return

    # 2. Tải ViT Backbone và tạo model Re-ID
    logger.info(f"Tải ViT Backbone: {args.backbone}...")
    backbone, embed_dim, _ = get_dino_backbone(model_name=args.backbone, pretrained=True, device=device)
    model = VehicleReIDModel(backbone=backbone, embed_dim=embed_dim, reid_dim=256, freeze_backbone=True).to(device)
    model.eval()

    # 3. Trích xuất Embeddings cho tất cả xe
    logger.info("Đang trích xuất vector đặc trưng danh tính (Re-ID Embeddings)...")
    embeddings_list = []
    with torch.no_grad():
        for veh in extracted_vehicles:
            tensor = transform(veh["crop_pil"]).unsqueeze(0).to(device)
            emb = model.extract_embedding(tensor)  # (1, 256)
            embeddings_list.append(emb.cpu())

    all_embeddings = torch.cat(embeddings_list, dim=0)  # (N, 256)

    # 4. Thiết lập Gallery và chạy truy vấn thử nghiệm
    reid_matcher = VehicleReIDMatcher(gallery_embeddings=all_embeddings, gallery_meta=extracted_vehicles)

    # Chọn tối đa 5 xe ngẫu nhiên làm Query để kiểm tra truy vấn
    num_queries = min(5, len(extracted_vehicles))
    query_indices = np.random.choice(len(extracted_vehicles), size=num_queries, replace=False)

    results_table = []
    for q_idx in query_indices:
        q_item = extracted_vehicles[q_idx]
        q_emb = all_embeddings[q_idx]

        # Truy vấn tìm xe ở các camera KHÁC
        top_matches = reid_matcher.query(
            query_embedding=q_emb,
            query_cam_id=q_item["cam_id"],
            top_k=args.top_k,
            filter_same_camera=False,  # Hiển thị cả cùng và khác camera
        )

        vis_path = os.path.join(vis_dir, f"query_cam{q_item['cam_id']}_veh{q_idx}.png")
        save_retrieval_visualization(q_item, top_matches, vis_path)

        for match in top_matches:
            results_table.append({
                "query_cam": q_item["cam_id"],
                "query_file": q_item["origin_name"],
                "rank": match["rank"],
                "sim": match["similarity"],
                "match_cam": match["meta"]["cam_id"],
                "match_file": match["meta"]["origin_name"],
            })

    # Lưu bảng kết quả CSV
    csv_path = os.path.join(args.output_dir, "reid_retrieval_results.csv")
    pd.DataFrame(results_table).to_csv(csv_path, index=False)
    logger.info(f"✅ Hoàn tất! Đã lưu kết quả tại {csv_path} và các ảnh trực quan hóa trong {vis_dir}.")


if __name__ == "__main__":
    main()
