"""
=============================================================================
 Hướng 7: Open-Vocabulary Traffic Scene Understanding via Delta Proposals
 Module: Execution Pipeline (Quy trình nhận diện giao thông từ vựng mở)
=============================================================================
"""

import argparse
import os
import sys
import logging
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
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
from proposal_engine import DeltaProposalEngine
from text_prompts import TrafficPromptVocabulary
from models import OpenVocabTrafficDetector


def setup_logger():
    logger = logging.getLogger("OpenVocabTraffic")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    return logger


def draw_open_vocab_detections(
    origin_pil: Image.Image,
    detections: List[Dict[str, Any]],
    save_path: str,
):
    """Vẽ bounding box và nhãn văn bản mở lên ảnh hiện trường."""
    img_draw = origin_pil.copy()
    draw = ImageDraw.Draw(img_draw)

    # Bảng màu cho các lớp
    colors = ["#FF3838", "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#00A878", "#7D82B8"]

    for det in detections:
        x, y, w, h = det["bbox"]
        label = det["class_name"]
        prob = det["confidence"]
        color = colors[hash(label) % len(colors)]

        # Vẽ hình chữ nhật
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        # Vẽ nhãn chữ
        text_str = f"{label} ({prob:.2f})"
        draw.rectangle([x, max(0, y - 20), x + len(text_str) * 8, y], fill=color)
        draw.text((x + 2, max(0, y - 18)), text_str, fill="white")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_draw.save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Open-Vocabulary Traffic Scene Understanding")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục background")
    parser.add_argument("--origin_dir", type=str, default="output", help="Thư mục origin")
    parser.add_argument("--output_dir", type=str, default="checkpoints/direction7_open_vocabulary", help="Thư mục xuất kết quả")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone DINO")
    parser.add_argument("--language", type=str, default="vi", choices=["vi", "en"], help="Ngôn ngữ danh mục ('vi' hoặc 'en')")
    parser.add_argument("--custom_classes", type=str, default=None, help="Danh sách danh mục tùy biến, cách nhau bởi dấu phẩy")
    parser.add_argument("--max_images", type=int, default=30, help="Số khung hình tối đa cần phân tích")
    parser.add_argument("--min_confidence", type=float, default=0.20, help="Ngưỡng xác suất tối thiểu để giữ lại nhãn")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán")
    args = parser.parse_args()

    logger = setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Khởi chạy Open-Vocabulary Traffic Understanding trên: {device}")

    # 1. Khởi tạo danh mục từ vựng
    custom_list = [c.strip() for c in args.custom_classes.split(",")] if args.custom_classes else None
    vocab = TrafficPromptVocabulary(classes=custom_list, language=args.language, text_dim=512)
    logger.info(f"Danh mục từ vựng mở ({len(vocab.classes)} lớp): {vocab.classes}")

    text_embeddings = vocab.get_text_embeddings(device=device)

    # 2. Khởi tạo Proposal Engine và ViT Model
    proposal_engine = DeltaProposalEngine(min_area=350, max_proposals=20)

    logger.info(f"Tải ViT Backbone: {args.backbone}...")
    backbone, embed_dim, _ = get_dino_backbone(model_name=args.backbone, pretrained=True, device=device)
    model = OpenVocabTrafficDetector(backbone=backbone, embed_dim=embed_dim, clip_dim=512, freeze_backbone=True).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Quét cặp ảnh
    matcher = TrafficPairMatcher(bg_dir=args.bg_dir, origin_dir=args.origin_dir, match_strategy="route_hourly")
    pairs = matcher.discover_pairs()[:args.max_images]
    if not pairs:
        logger.error("Không tìm thấy cặp ảnh giao thông hợp lệ.")
        return

    logger.info(f"Đang thực hiện nhận diện trên {len(pairs)} khung hình...")

    all_detections_record = []
    for p in tqdm(pairs, desc="Open-Vocab Detection"):
        try:
            orig_pil = Image.open(p["origin_path"]).convert("RGB")
            bg_pil = Image.open(p["bg_path"]).convert("RGB")

            # Sinh đề xuất từ trường sai khác Delta
            proposals = proposal_engine.generate_proposals(orig_pil, bg_pil)
            if not proposals:
                continue

            # Chuẩn bị batch tensor các vùng crop
            crop_tensors = torch.stack([transform(prop["crop_pil"]) for prop in proposals], dim=0).to(device)

            with torch.no_grad():
                out = model(crop_tensors, text_embeddings)
                probs = out["probabilities"]  # (B, N_classes)
                pred_ids = out["predicted_class_ids"].cpu().numpy()

            frame_detections = []
            for i, prop in enumerate(proposals):
                cid = pred_ids[i]
                cname = vocab.classes[cid]
                conf = float(probs[i, cid].item())

                if conf >= args.min_confidence:
                    det_item = {
                        "bbox": prop["bbox"],
                        "class_name": cname,
                        "confidence": conf,
                        "area": prop["area"],
                    }
                    frame_detections.append(det_item)

                    all_detections_record.append({
                        "filename": p["origin_name"],
                        "x": prop["bbox"][0],
                        "y": prop["bbox"][1],
                        "w": prop["bbox"][2],
                        "h": prop["bbox"][3],
                        "class_name": cname,
                        "confidence": conf,
                    })

            # Vẽ ảnh trực quan hóa
            save_vis = os.path.join(vis_dir, f"{os.path.splitext(p['origin_name'])[0]}_openvocab.jpg")
            draw_open_vocab_detections(orig_pil, frame_detections, save_vis)

        except Exception as e:
            logger.warning(f"Lỗi khi xử lý {p['origin_name']}: {e}")

    # Xuất báo cáo CSV
    csv_path = os.path.join(args.output_dir, "open_vocab_detections.csv")
    pd.DataFrame(all_detections_record).to_csv(csv_path, index=False)
    logger.info(f"✅ Hoàn tất nhận diện từ vựng mở! Bảng kết quả lưu tại: {csv_path}")


if __name__ == "__main__":
    main()
