"""
=============================================================================
 Hướng 6: Delta-Guided Unsupervised Vehicle Re-Identification Across Cameras
          and Corridor Travel Time Estimation
 Module: Pipeline Execution (Quy trình nhận dạng lại xe và đo vận tốc hành trình)
=============================================================================
"""

import argparse
import os
import sys
import logging
import re
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
from direction6_vehicle_reid.roi_extractor import DeltaRoIExtractor
from direction6_vehicle_reid.models import VehicleReIDModel
from direction6_vehicle_reid.matcher import VehicleReIDMatcher


def setup_logger():
    logger = logging.getLogger("CorridorVehicleReID")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    return logger


def save_retrieval_visualization(
    query_info: Dict[str, Any],
    top_results: List[Dict[str, Any]],
    save_path: str,
):
    """Vẽ ảnh đối chiếu trực quan hóa xe Query và Top-k ứng viên xe tìm thấy ở các camera khác."""
    num_matches = len(top_results)
    if num_matches == 0:
        return

    fig, axes = plt.subplots(1, 1 + num_matches, figsize=(3.2 * (1 + num_matches), 4), dpi=150)
    if num_matches == 0:
        axes = [axes]

    # 1. Vẽ ảnh xe Query
    axes[0].imshow(query_info["crop_pil"])
    axes[0].set_title(
        f"Query Vehicle\nCam: {query_info['cam_id']}\nFile: {query_info['origin_name'][:15]}...",
        color="blue",
        fontsize=9,
        fontweight="bold",
    )
    axes[0].axis("off")

    # 2. Vẽ Top Matches
    for idx, res in enumerate(top_results):
        ax = axes[1 + idx]
        crop = res["meta"]["crop_pil"]
        v_sim = res["visual_sim"]
        st_w = res.get("st_weight", 1.0)
        final_sc = res["final_score"]
        cam = res["meta"]["cam_id"]

        is_high_conf = final_sc >= 0.65
        title_color = "darkgreen" if is_high_conf else "black"
        ax.imshow(crop)
        ax.set_title(
            f"Rank {res['rank']} ({final_sc:.2f})\nCam: {cam}\nVis:{v_sim:.2f} | ST:{st_w:.2f}",
            color=title_color,
            fontsize=8,
        )
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Corridor-Based Vehicle Re-ID & Travel Time Estimation")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục ảnh nền")
    parser.add_argument("--origin_dir", type=str, default="output", help="Thư mục ảnh origin")
    parser.add_argument("--output_dir", type=str, default="checkpoints/direction6_vehicle_reid", help="Thư mục xuất kết quả")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone DINO")
    parser.add_argument("--distance_meters", type=float, default=1200.0, help="Khoảng cách giả định giữa các camera trên tuyến (mét)")
    parser.add_argument("--top_k", type=int, default=5, help="Số lượng ứng viên xe cần truy vấn")
    parser.add_argument("--min_area", type=int, default=400, help="Diện tích tối thiểu để phát hiện xe")
    parser.add_argument("--max_pairs", type=int, default=50, help="Số lượng ảnh tối đa để trích xuất gallery")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán ('cuda' hoặc 'cpu')")
    args = parser.parse_args()

    logger = setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Khởi chạy Corridor Vehicle Re-ID trên thiết bị: {device}")

    # 1. Quét dữ liệu và phát hiện phương tiện qua Delta-RoI
    matcher = TrafficPairMatcher(bg_dir=args.bg_dir, origin_dir=args.origin_dir, match_strategy="route_hourly")
    all_pairs = matcher.discover_pairs()
    if not all_pairs:
        logger.error("Không tìm thấy cặp ảnh giao thông hợp lệ trong thư mục.")
        return

    pairs = all_pairs[:args.max_pairs]
    logger.info(f"Đang phân tích {len(pairs)} khung hình bằng Delta-RoI...")

    roi_extractor = DeltaRoIExtractor(min_area=args.min_area, target_size=(256, 128))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    extracted_vehicles: List[Dict[str, Any]] = []
    for p in tqdm(pairs, desc="Extracting Delta-RoIs"):
        try:
            orig = Image.open(p["origin_path"]).convert("RGB")
            bg = Image.open(p["bg_path"]).convert("RGB")
            rois = roi_extractor.extract_rois(orig, bg)

            m = re.match(r"^(\d+)_(.+)\.(jpg|jpeg|png)$", p["origin_name"], re.IGNORECASE)
            cam_id = m.group(1) if m else "unknown"
            try:
                ts = float(m.group(2)) if m else 0.0
            except ValueError:
                ts = 0.0

            for roi in rois:
                extracted_vehicles.append({
                    "cam_id": cam_id,
                    "origin_name": p["origin_name"],
                    "timestamp": ts,
                    "bbox": roi["bbox"],
                    "crop_pil": roi["crop_pil"],
                    "area": roi["area"],
                    "confidence": roi["confidence"],
                })
        except Exception as e:
            logger.warning(f"Lỗi khi xử lý {p['origin_name']}: {e}")

    logger.info(f"✅ Đã trích xuất thành công {len(extracted_vehicles)} phương tiện từ các camera.")
    if len(extracted_vehicles) < 2:
        logger.warning("Không đủ số lượng phương tiện để thực hiện so khớp liên camera.")
        return

    # 2. Tải ViT Backbone và Mô hình VehicleReIDModel
    logger.info(f"Tải ViT Backbone: {args.backbone}...")
    backbone, embed_dim, _ = get_dino_backbone(model_name=args.backbone, pretrained=True, device=device)
    model = VehicleReIDModel(backbone=backbone, embed_dim=embed_dim, reid_dim=256, freeze_backbone=True).to(device)
    model.eval()

    # 3. Trích xuất đặc trưng L2-normalized 256-D
    logger.info("Trích xuất vector nhúng nhận dạng (L2-normalized 256-D Re-ID Embeddings)...")
    embeddings_list = []
    with torch.no_grad():
        for veh in extracted_vehicles:
            tensor = transform(veh["crop_pil"]).unsqueeze(0).to(device)
            emb = model.extract_embedding(tensor)
            embeddings_list.append(emb.cpu())

    all_embeddings = torch.cat(embeddings_list, dim=0)

    # 4. So khớp liên camera dọc hành lang giao thông
    reid_matcher = VehicleReIDMatcher(gallery_embeddings=all_embeddings, gallery_meta=extracted_vehicles)

    # Lấy danh sách camera khác nhau
    cam_set = sorted(list(set(v["cam_id"] for v in extracted_vehicles)))
    logger.info(f"Phát hiện {len(cam_set)} camera trên tuyến: {cam_set}")

    # Thực hiện truy vấn kiểm thử
    num_queries = min(10, len(extracted_vehicles))
    query_indices = np.random.choice(len(extracted_vehicles), size=num_queries, replace=False)

    results_table = []
    matched_pairs_for_speed = []

    for q_idx in query_indices:
        q_item = extracted_vehicles[q_idx]
        q_emb = all_embeddings[q_idx]

        top_matches = reid_matcher.query_with_spatio_temporal(
            query_embedding=q_emb,
            query_meta=q_item,
            top_k=args.top_k,
            filter_same_camera=True,  # Bắt buộc tìm xe ở camera KHÁC
            use_temporal_filter=True,
            distance_meters=args.distance_meters,
        )

        if top_matches:
            vis_path = os.path.join(vis_dir, f"query_cam{q_item['cam_id']}_idx{q_idx}.png")
            save_retrieval_visualization(q_item, top_matches, vis_path)

            for m in top_matches:
                m["query_ts"] = q_item["timestamp"]
                matched_pairs_for_speed.append(m)
                results_table.append({
                    "query_cam": q_item["cam_id"],
                    "query_file": q_item["origin_name"],
                    "query_time": q_item["timestamp"],
                    "rank": m["rank"],
                    "visual_sim": m["visual_sim"],
                    "st_weight": m["st_weight"],
                    "final_score": m["final_score"],
                    "match_cam": m["meta"]["cam_id"],
                    "match_file": m["meta"]["origin_name"],
                    "match_time": m["meta"]["timestamp"],
                })

    # 5. Ước lượng vận tốc hành trình và thời gian di chuyển
    speed_report = VehicleReIDMatcher.estimate_corridor_speed(
        matched_pairs_for_speed,
        distance_meters=args.distance_meters,
        min_sim_threshold=0.60,
    )

    logger.info("=" * 60)
    logger.info(" 📊 BÁO CÁO THỜI GIAN HÀNH TRÌNH TUYẾN ĐƯỜNG (CORRIDOR REPORT):")
    logger.info(f"   • Khoảng cách giữa 2 nút camera: {args.distance_meters} m")
    logger.info(f"   • Số cặp xe trùng khớp tin cậy: {speed_report['num_matched_pairs']}")
    logger.info(f"   • Thời gian di chuyển trung bình: {speed_report['mean_travel_time_sec']:.1f} giây")
    logger.info(f"   • Vận tốc hành trình trung bình: {speed_report['mean_speed_kmh']:.1f} km/h")
    logger.info("=" * 60)

    # Xuất file CSV
    csv_results_path = os.path.join(args.output_dir, "reid_retrieval_results.csv")
    pd.DataFrame(results_table).to_csv(csv_results_path, index=False)

    csv_speed_path = os.path.join(args.output_dir, "corridor_speed_report.csv")
    pd.DataFrame([speed_report]).to_csv(csv_speed_path, index=False)

    logger.info(f"✅ Hoàn tất! Báo cáo kết quả đã được lưu tại {args.output_dir}.")


if __name__ == "__main__":
    main()
