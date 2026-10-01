"""
=============================================================================
 Hướng 5: Unsupervised Traffic Anomaly Detection
 Execution Pipeline
=============================================================================
"""

import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.matcher import TrafficPairMatcher
from feature_extractor import DeltaConditionedExtractor
from memory_bank import AnomalyMemoryBank
from detector import TrafficAnomalyDetector


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def save_visualization(origin_np, bg_np, delta_mask, score, is_anomaly, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(bg_np)
    axes[0].set_title("Background")
    axes[0].axis('off')
    
    axes[1].imshow(origin_np)
    title = f"Origin - Score: {score:.2f} - {'ANOMALY' if is_anomaly else 'NORMAL'}"
    axes[1].set_title(title, color='red' if is_anomaly else 'green')
    axes[1].axis('off')
    
    axes[2].imshow(delta_mask, cmap='gray')
    axes[2].set_title("Delta Mask")
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Unsupervised Traffic Anomaly Detection Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Thư mục chứa ảnh Origin giao thông")
    parser.add_argument("--bg_dir", type=str, required=True, help="Thư mục chứa ảnh Background")
    parser.add_argument("--output_dir", type=str, default="output_anomaly", help="Thư mục xuất kết quả")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14", help="Tên mô hình DINO (vd: dinov2_vits14)")
    parser.add_argument("--threshold", type=float, default=2.0, help="Ngưỡng phân loại bất thường")
    parser.add_argument("--bank_size", type=int, default=1000, help="Kích thước Memory Bank")
    parser.add_argument("--fit_samples", type=int, default=100, help="Số lượng ảnh ban đầu dùng để fit memory bank")
    args = parser.parse_args()

    logger = setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    logger.info("Khởi tạo hệ thống Anomaly Detection...")
    
    # Init Matcher
    matcher = TrafficPairMatcher(bg_dir=args.bg_dir, origin_dir=args.input_dir, match_strategy="route_hourly")
    pairs_info = matcher.discover_pairs()
    
    if not pairs_info:
        logger.error("Không tìm thấy cặp ảnh hợp lệ. Vui lòng kiểm tra lại cấu trúc thư mục.")
        return
        
    logger.info(f"Tìm thấy {len(pairs_info)} cặp ảnh.")
    
    # Init modules
    extractor = DeltaConditionedExtractor(backbone_name=args.backbone, device='cuda')
    feature_dim = extractor.embed_dim + 3 # DINO [CLS] + 3 Delta stats
    memory_bank = AnomalyMemoryBank(feature_dim=feature_dim, bank_size=args.bank_size)
    detector = TrafficAnomalyDetector(extractor=extractor, memory_bank=memory_bank, threshold=args.threshold)
    
    # Chuẩn bị dữ liệu
    all_pairs = []
    for p_info in pairs_info:
        try:
            origin_np = np.array(Image.open(p_info["origin_path"]).convert("RGB"))
            bg_np = np.array(Image.open(p_info["bg_path"]).convert("RGB"))
            
            if origin_np.shape[:2] != bg_np.shape[:2]:
                import cv2
                bg_np = cv2.resize(bg_np, (origin_np.shape[1], origin_np.shape[0]))
                
            all_pairs.append({
                "origin": origin_np,
                "bg": bg_np,
                "name": p_info["origin_name"],
                "origin_path": p_info["origin_path"]
            })
        except Exception as e:
            logger.warning(f"Lỗi khi đọc {p_info['origin_name']}: {e}")
            
    if len(all_pairs) < args.fit_samples:
        logger.warning("Số lượng mẫu tìm thấy ít hơn fit_samples. Sử dụng toàn bộ mẫu để fit.")
        fit_samples = len(all_pairs)
    else:
        fit_samples = args.fit_samples
        
    # Phase 1: Build memory bank (Sử dụng các mẫu đầu tiên làm normal data)
    fit_data = [(p["origin"], p["bg"]) for p in all_pairs[:fit_samples]]
    detector.fit(fit_data)
    
    # Phase 2: Detect trên các mẫu còn lại (hoặc toàn bộ)
    eval_data = all_pairs[fit_samples:]
    if not eval_data:
        eval_data = all_pairs # Fallback đánh giá lại trên tập fit nếu thiếu dữ liệu
        
    logger.info(f"Bắt đầu phát hiện bất thường trên {len(eval_data)} mẫu...")
    
    results = []
    for p in tqdm(eval_data, desc="Detecting"):
        is_anomaly, score = detector.detect(p["origin"], p["bg"])
        results.append({
            "filename": p["name"],
            "score": score,
            "is_anomaly": is_anomaly
        })
        
        # Visualization cho ảnh bất thường hoặc một số ảnh bình thường ngẫu nhiên
        if is_anomaly or np.random.rand() < 0.1:
            delta_norm, _ = extractor.subtractor.compute_delta(p["origin"], p["bg"])
            delta_mask = extractor.subtractor.extract_binary_mask(delta_norm)
            save_path = os.path.join(vis_dir, f"{os.path.splitext(p['name'])[0]}_anomaly.png")
            save_visualization(p["origin"], p["bg"], delta_mask, score, is_anomaly, save_path)
            
    # Lưu CSV
    csv_path = os.path.join(args.output_dir, "anomaly_scores.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Đã lưu kết quả tại {csv_path}")

if __name__ == "__main__":
    main()
