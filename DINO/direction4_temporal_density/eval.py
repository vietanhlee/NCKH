"""
=============================================================================
 Hướng 4: Spatio-Temporal Density & HCM LoS Estimation
 Module: Evaluation & Metrics Reporting (Production-Ready)
 Đánh giá độ chính xác ước lượng tỷ lệ chiếm dụng lòng đường (MAE, RMSE)
 và phân loại cấp độ dịch vụ giao thông đô thị (HCM Level of Service)
=============================================================================
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any

# Chống xung đột OpenMP và hỗ trợ tiếng Việt trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import project root
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dino_dir = os.path.dirname(_current_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)

from common.backbone_loader import get_dino_backbone
from common.gpu_utils import load_checkpoint
from direction4_temporal_density.dataset import TemporalTrafficDataset
from direction4_temporal_density.models import SpatioTemporalDensityNet


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Spatio-Temporal Density & HCM LoS")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục background")
    parser.add_argument("--origin_dir", type=str, default="output", help="Thư mục origin images")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/direction4_temporal_density/eval", help="Thư mục lưu báo cáo")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone DINO")
    parser.add_argument("--window_size", type=int, default=4, help="Kích thước cửa sổ thời gian")
    parser.add_argument("--img_size", type=int, default=224, help="Kích thước ảnh đầu vào")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size đánh giá")
    parser.add_argument("--max_sequences", type=int, default=None, help="Giới hạn số chuỗi kiểm tra nhanh")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị ('cuda' hoặc 'cpu')")
    return parser.parse_args()


def evaluate_temporal_model(args):
    """Đánh giá toàn diện mô hình Spatio-Temporal Density & LoS."""
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"🚀 [Eval] Bắt đầu đánh giá trên thiết bị: {device}")

    # 1. Khởi tạo Dataset chế độ đánh giá (is_train=False)
    dataset = TemporalTrafficDataset(
        bg_dir=args.bg_dir,
        origin_dir=args.origin_dir,
        window_size=args.window_size,
        img_size=args.img_size,
        is_train=False,
        max_sequences=args.max_sequences,
    )

    if len(dataset) == 0:
        print("⚠️ [Eval] Không tìm thấy dữ liệu hợp lệ để đánh giá.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Khởi tạo mô hình & nạp checkpoint
    backbone_vit, _, _ = get_dino_backbone(model_name=args.backbone, pretrained=False, device=device)
    model = SpatioTemporalDensityNet(backbone=backbone_vit, freeze_backbone=True)
    model.to(device)

    print(f"📦 [Checkpoint] Nạp trọng số từ: {args.weights}")
    load_checkpoint(args.weights, model=model, device=device, strict=False, verbose=True)
    model.eval()

    # 3. Vòng lặp suy luận
    pred_occ_all = []
    target_occ_all = []
    pred_los_all = []
    target_los_all = []

    los_names = ["0: Free-Flow", "1: Moderate", "2: Slow", "3: Congested"]
    confusion_mat = np.zeros((4, 4), dtype=int)

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Đang đánh giá"):
            rgb_seq = batch["rgb_seq"].to(device, non_blocking=True)
            delta_seq = batch["delta_seq"].to(device, non_blocking=True)
            target_occ = batch["current_occupancy"].cpu().numpy()
            target_los = batch["current_los"].cpu().numpy()

            outputs = model(rgb_seq, delta_seq)

            pred_occ = outputs["pred_occupancy"].cpu().numpy()
            pred_los = torch.argmax(outputs["logits_los"], dim=1).cpu().numpy()

            pred_occ_all.extend(pred_occ)
            target_occ_all.extend(target_occ)
            pred_los_all.extend(pred_los)
            target_los_all.extend(target_los)

            for t, p in zip(target_los, pred_los):
                if 0 <= t < 4 and 0 <= p < 4:
                    confusion_mat[t, p] += 1

    elapsed = time.time() - start_time
    pred_occ_all = np.array(pred_occ_all)
    target_occ_all = np.array(target_occ_all)
    pred_los_all = np.array(pred_los_all)
    target_los_all = np.array(target_los_all)

    # 4. Tính toán các chỉ số
    mae = float(np.mean(np.abs(pred_occ_all - target_occ_all)))
    rmse = float(np.sqrt(np.mean((pred_occ_all - target_occ_all) ** 2)))
    los_acc = float(np.mean(pred_los_all == target_los_all)) * 100.0

    print("\n" + "=" * 65)
    print(" 📊 KẾT QUẢ ĐÁNH GIÁ SPATIO-TEMPORAL DENSITY & HCM LoS")
    print("=" * 65)
    print(f" - Tổng số mẫu đánh giá: {len(pred_occ_all)}")
    print(f" - Thời gian suy luận: {elapsed:.2f}s ({len(pred_occ_all)/max(0.001, elapsed):.1f} samples/s)")
    print(f" - Sai số Mật độ trung bình (MAE Occupancy): {mae:.4f}")
    print(f" - Sai số Căn bậc hai (RMSE Occupancy): {rmse:.4f}")
    print(f" - Độ chính xác phân loại LoS (Accuracy): {los_acc:.2f}%")
    print("\n [Ma trận Nhầm lẫn Cấp độ Dịch vụ LoS (Hàng: Thực tế, Cột: Dự đoán)]")
    header = "       " + " ".join([f"[{i}]" for i in range(4)])
    print(header)
    for i in range(4):
        row_str = f"[{i}]    " + "   ".join([f"{confusion_mat[i, j]:4d}" for j in range(4)])
        print(f"{row_str}  -> {los_names[i]}")
    print("=" * 65)

    # 5. Xuất báo cáo ra file JSON
    report = {
        "num_samples": len(pred_occ_all),
        "mae_occupancy": mae,
        "rmse_occupancy": rmse,
        "los_accuracy_percent": los_acc,
        "confusion_matrix": confusion_mat.tolist(),
        "los_labels": los_names,
    }
    report_path = os.path.join(args.output_dir, "temporal_density_eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📁 Báo cáo chi tiết đã lưu tại: {report_path}")


if __name__ == "__main__":
    args = parse_args()
    evaluate_temporal_model(args)
