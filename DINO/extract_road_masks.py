"""
=============================================================================
 Production Tool: Drivable Road Mask Extractor (extract_road_masks.py)
 Tự động trích xuất mặt nạ làn đường / phần đường di chuyển của xe theo từng tuyến đường
 Quy trình:
   1. Quét ảnh camera và gom nhóm theo Tuyến đường (Route ID / Camera STT).
   2. Bộ lọc ảnh ban ngày (Daytime Filter: 06:00 - 18:00 UTC+7 + Lọc quang trắc an toàn).
   3. Tích lũy vết tích chuyển động phương tiện qua chuỗi khung hình ban ngày.
   4. Tái cấu trúc hành lang lòng đường (Drivable Corridor), lấp kín các làn xe,
      loại trừ hoàn toàn vỉa hè người đi bộ, cây cối, nhà cửa, bầu trời.
   5. Xuất mặt nạ nhị phân, ảnh trực quan hóa phủ màu dạ quang và báo cáo thống kê.
=============================================================================
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# Cấu hình UTF-8 cho terminal Windows và chống xung đột OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn project
_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

from common.road_extractor import DaytimeFilter, VehicleOccupancyAccumulator, DrivableCorridorReconstructor
from common.matcher import TrafficPairMatcher


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trích xuất mặt nạ lòng đường xe chạy (Drivable Road Mask) theo từng tuyến đường từ ảnh ban ngày"
    )
    # Đường dẫn dữ liệu
    parser.add_argument("--origin_dir", type=str, default="output", help="Thư mục chứa ảnh gốc snapshot của các camera")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục chứa ảnh background tham chiếu")
    parser.add_argument("--output_dir", type=str, default="road_masks", help="Thư mục xuất mặt nạ và ảnh trực quan hóa")
    
    # Checkpoint Hướng 2 (tùy chọn)
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn file checkpoint .pth mô hình Hướng 2 (TrafficDecompositionNet)")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone DINOv3 (dinov3_vits16, dinov2_vits14...)")
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "decomposition", "fast"],
                        help="Phương pháp trích xuất xe: 'auto' (tự động dùng weights nếu có), 'decomposition' (mô hình Hướng 2), 'fast' (trừ nền nhanh không cần GPU)")
    
    # Cấu hình lọc ảnh ban ngày
    parser.add_argument("--start_hour", type=int, default=6, help="Giờ bắt đầu tính ban ngày (mặc định: 6 = 06:00 UTC+7)")
    parser.add_argument("--end_hour", type=int, default=18, help="Giờ kết thúc ban ngày (mặc định: 18 = 17:59:59 UTC+7)")
    parser.add_argument("--min_mean_v", type=float, default=45.0, help="Độ sáng trung bình tối thiểu kênh V (HSV) để xác nhận ảnh ban ngày")
    parser.add_argument("--min_daytime_frames", type=int, default=2, help="Số khung hình ban ngày tối thiểu để xử lý một tuyến đường")
    parser.add_argument("--max_frames_per_route", type=int, default=30, help="Số khung hình ban ngày lấy mẫu tối đa trên mỗi tuyến để tối ưu thời gian")

    # Cấu hình tái cấu trúc lòng đường
    parser.add_argument("--seed_thresh", type=float, default=0.12, help="Ngưỡng kích hoạt hạt giống xe chạy từ bản đồ nhiệt tích lũy")
    parser.add_argument("--horizon_margin", type=float, default=0.15, help="Tỷ lệ chiều cao từ đỉnh ảnh bị loại bỏ (bầu trời, ngọn cây, mái nhà)")
    parser.add_argument("--min_road_ratio", type=float, default=0.05, help="Tỷ lệ diện tích tối thiểu của lòng đường so với toàn ảnh (mặc định 5%%)")

    # Tuyến cụ thể & Thiết bị
    parser.add_argument("--routes", nargs="*", default=None, help="Chỉ định danh sách tuyến cần xử lý (ví dụ: --routes 1 2 256), bỏ trống để xử lý tất cả")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị chạy ('cuda' hoặc 'cpu')")
    parser.add_argument("--save_overlay", action="store_true", default=True, help="Lưu ảnh phủ màu dạ quang trực quan")
    parser.add_argument("--no_overlay", action="store_false", dest="save_overlay", help="Không lưu ảnh overlay")
    parser.add_argument("--save_heatmap", action="store_true", default=True, help="Lưu ảnh bản đồ nhiệt mật độ phương tiện")
    parser.add_argument("--no_heatmap", action="store_false", dest="save_heatmap", help="Không lưu ảnh heatmap")

    return parser.parse_args()


def load_decomposition_model(weights_path: str, backbone_name: str, device: str):
    """Nạp mô hình TrafficDecompositionNet từ checkpoint."""
    import torch
    from common.backbone_loader import get_dino_backbone
    from common.gpu_utils import load_checkpoint
    from direction2_scene_decomposition.models import TrafficDecompositionNet

    dev = torch.device(device if torch.cuda.is_available() and "cuda" in device.lower() else "cpu")
    print(f"📦 [Model Loader] Đang nạp backbone {backbone_name} và checkpoint: {weights_path}")
    backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=backbone_name,
        pretrained=True,
        device=dev,
    )
    model = TrafficDecompositionNet(
        backbone=backbone,
        embed_dim=embed_dim,
        patch_size=patch_size,
    ).to(dev)

    load_checkpoint(
        load_path=weights_path,
        model=model,
        device=dev,
        strict=False,
        verbose=True,
    )
    model.eval()
    return model, dev


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(" 🛣️ CÔNG CỤ TRÍCH XUẤT MẶT NẠ LÒNG ĐƯỜNG XE CHẠY (DRIVABLE ROAD MASK EXTRACTOR)")
    print("=" * 80)
    print(f" Thư mục ảnh gốc       : {args.origin_dir}")
    print(f" Thư mục background    : {args.bg_dir if os.path.exists(args.bg_dir) else 'Không tìm thấy (dùng chế độ tự thân)'}")
    print(f" Thư mục lưu kết quả   : {args.output_dir}")
    print(f" Dải giờ ban ngày      : {args.start_hour:02d}:00 -> {args.end_hour:02d}:00 (UTC+7, V_mean >= {args.min_mean_v})")
    print(f" Khung hình/tuyến tối đa: {args.max_frames_per_route}")
    print(f" Phương pháp trích xuất: {args.method}")
    print("=" * 80 + "\n")

    # 1. Quét toàn bộ ảnh trong thư mục origin
    if not os.path.exists(args.origin_dir):
        raise FileNotFoundError(f"Thư mục ảnh gốc không tồn tại: {args.origin_dir}")

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_files = []
    for root, _, files in os.walk(args.origin_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in supported_exts:
                all_files.append(os.path.join(root, f))

    all_files.sort()
    print(f"🔍 [Scanner] Đã tìm thấy {len(all_files)} file ảnh trong {args.origin_dir}")
    if not all_files:
        print("⚠️ Không có ảnh nào để xử lý. Kết thúc.")
        return

    # 2. Gom nhóm ảnh theo Tuyến đường (Route ID)
    day_filter = DaytimeFilter(
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        min_mean_v=args.min_mean_v,
    )

    route_groups = defaultdict(list)
    for p in all_files:
        r_id, _ = day_filter.parse_image_time(p)
        route_groups[r_id].append(p)

    all_routes = sorted(list(route_groups.keys()), key=lambda x: (int(x) if x.isdigit() else 999999, x))
    print(f"📊 [Grouping] Phát hiện {len(all_routes)} tuyến đường khác nhau.")

    # Lọc danh sách tuyến nếu có chỉ định --routes
    if args.routes:
        target_routes = set(str(int(r)) if r.isdigit() else r for r in args.routes)
        active_routes = [r for r in all_routes if r in target_routes]
        print(f"🎯 [Filter Routes] Chỉ xử lý {len(active_routes)} tuyến chỉ định: {active_routes}")
    else:
        active_routes = all_routes

    if not active_routes:
        print("⚠️ Không có tuyến nào thỏa mãn điều kiện. Kết thúc.")
        return

    # 3. Khởi tạo Matcher để tra cứu Background tham chiếu (nếu có)
    matcher = None
    if os.path.exists(args.bg_dir):
        matcher = TrafficPairMatcher(bg_dir=args.bg_dir, origin_dir=args.origin_dir, match_strategy="route_hourly")
        matcher.build_background_index()
        print(f"🔗 [Matcher] Đã nạp chỉ mục background từ: {args.bg_dir}")

    # 4. Xác định phương pháp và nạp mô hình Hướng 2 nếu cần
    use_model = False
    model = None
    dev = "cpu"

    should_use_model = (args.method == "decomposition") or (args.method == "auto" and args.weights is not None)
    if should_use_model:
        if args.weights and os.path.isfile(args.weights):
            try:
                model, dev = load_decomposition_model(args.weights, args.backbone, args.device)
                use_model = True
                print("✅ [Model] Đã kích hoạt chế độ phân rã học sâu TrafficDecompositionNet.")
            except Exception as e_m:
                print(f"⚠️ [Model Error] Không thể nạp weights ({e_m}). Tự động fallback sang chế độ Fast Subtraction.")
                use_model = False
        else:
            if args.method == "decomposition":
                raise FileNotFoundError(f"Chỉ định --method decomposition nhưng không tìm thấy file --weights: {args.weights}")
            print("💡 [Notice] Không cung cấp --weights, tự động sử dụng phương pháp Fast Subtraction.")
    else:
        print("⚡ [Notice] Chạy chế độ Fast Subtraction (Trừ nền bất biến ánh sáng tốc độ cao).")

    accumulator = VehicleOccupancyAccumulator(
        model=model,
        device=dev,
        img_size=256,
    )

    reconstructor = DrivableCorridorReconstructor(
        seed_threshold=args.seed_thresh,
        min_road_area_ratio=args.min_road_ratio,
        horizon_margin_ratio=args.horizon_margin,
    )

    # 5. Vòng lặp xử lý từng tuyến đường
    summary_records = []
    pbar = tqdm(active_routes, desc="Processing Routes")

    for r_id in pbar:
        paths = route_groups[r_id]
        pbar.set_postfix({"route": r_id, "total_imgs": len(paths)})

        # 5.1. Lọc các khung hình BAN NGÀY
        daytime_paths = []
        for p in paths:
            is_day, reason, _ = day_filter.is_daytime_image(p, verify_content=True)
            if is_day:
                daytime_paths.append(p)

        if len(daytime_paths) < args.min_daytime_frames:
            print(f"\n⚠️ [Bỏ qua Tuyến {r_id}] Chỉ có {len(daytime_paths)} ảnh ban ngày hợp lệ (< tối thiểu {args.min_daytime_frames}).")
            continue

        # 5.2. Tìm ảnh Background ban ngày tham chiếu cho tuyến này
        bg_candidate_path = None
        if matcher:
            bg_res = matcher.find_best_background(r_id, hour=12)
            if bg_res:
                bg_candidate_path = bg_res[0]

        # 5.3. Tích lũy vết tích chuyển động phương tiện
        try:
            accum_heat, sample_bgr = accumulator.accumulate_route(
                frame_paths=daytime_paths,
                bg_path=bg_candidate_path,
                max_frames=args.max_frames_per_route,
                use_model=use_model,
            )
        except Exception as e_acc:
            print(f"\n⚠️ [Lỗi tích lũy Tuyến {r_id}]: {e_acc}")
            continue

        # 5.4. Tái cấu trúc hành lang lòng đường
        road_mask_uint8, stats = reconstructor.reconstruct_road_mask(
            accum_heatmap=accum_heat,
            reference_img_bgr=sample_bgr,
        )

        # 5.5. Lưu các tệp kết quả
        mask_filename = f"route_{r_id}_road_mask.png"
        mask_path = os.path.join(args.output_dir, mask_filename)
        cv2.imwrite(mask_path, road_mask_uint8)

        overlay_path = None
        if args.save_overlay and sample_bgr is not None:
            overlay_bgr = reconstructor.create_road_overlay(
                image_bgr=sample_bgr,
                road_mask=road_mask_uint8,
                color=(0, 230, 118),  # Xanh ngọc dạ quang
                alpha=0.45,
                draw_contour=True,
                contour_color=(255, 255, 255),
                contour_thickness=2,
            )
            overlay_filename = f"route_{r_id}_road_overlay.jpg"
            overlay_path = os.path.join(args.output_dir, overlay_filename)
            cv2.imwrite(overlay_path, overlay_bgr)

        heatmap_path = None
        if args.save_heatmap:
            color_heat = reconstructor.create_heatmap_visualization(accum_heat)
            heat_filename = f"route_{r_id}_heatmap.jpg"
            heatmap_path = os.path.join(args.output_dir, heat_filename)
            cv2.imwrite(heatmap_path, color_heat)

        # Lưu bản ghi thống kê
        rec = {
            "route_id": r_id,
            "total_frames": len(paths),
            "daytime_frames_found": len(daytime_paths),
            "frames_used": min(len(daytime_paths), args.max_frames_per_route),
            "resolution": f"{stats['width']}x{stats['height']}",
            "road_area_pct": stats["road_area_pct"],
            "road_pixels": stats["road_pixels"],
            "total_pixels": stats["total_pixels"],
            "mask_path": mask_path,
            "overlay_path": overlay_path if overlay_path else "",
            "heatmap_path": heatmap_path if heatmap_path else "",
        }
        summary_records.append(rec)

    # 6. Xuất báo cáo tổng kết JSON & CSV
    json_path = os.path.join(args.output_dir, "road_masks_summary.json")
    csv_path = os.path.join(args.output_dir, "road_masks_summary.csv")

    with open(json_path, "w", encoding="utf-8") as f_j:
        json.dump(summary_records, f_j, indent=2, ensure_ascii=False)

    if summary_records:
        keys = list(summary_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f_c:
            writer = csv.DictWriter(f_c, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_records)

    print("\n" + "=" * 80)
    print(" 🎉 [HOÀN TẤT] QUÁ TRÌNH TRÍCH XUẤT MẶT NẠ LÒNG ĐƯỜNG XE CHẠY THÀNH CÔNG!")
    print("=" * 80)
    print(f" Tổng số tuyến hoàn thành  : {len(summary_records)} / {len(active_routes)}")
    print(f" Thư mục lưu mặt nạ        : {args.output_dir}")
    print(f" Báo cáo chi tiết JSON     : {json_path}")
    print(f" Báo cáo chi tiết CSV      : {csv_path}")
    if summary_records:
        avg_pct = np.mean([r["road_area_pct"] for r in summary_records])
        print(f" Tỷ lệ lòng đường trung bình: {avg_pct:.2f}% diện tích khung hình")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
