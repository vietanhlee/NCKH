"""
=============================================================================
Module: validate_temporal_coverage.py
Nghiệp vụ: Technical Validation V2 - Đánh giá tính liên tục chuỗi thời gian
           và phân phối khung hình theo các mốc thời gian trong ngày.
=============================================================================
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
VN_TZ = timezone(timedelta(hours=7))


def audit_temporal_distribution(image_directory: str) -> Dict[str, Any]:
    """
    Phân tích chuỗi thời gian của tất cả các camera trong thư mục ảnh.
    """
    print(f"[Temporal Coverage Audit] Analyzing timestamps in: {image_directory}")
    camera_timestamps: Dict[int, List[int]] = defaultdict(list)
    hourly_counts: Dict[int, int] = {h: 0 for h in range(24)}
    day_counts: Dict[str, int] = defaultdict(int)

    total_frames = 0

    for root_dir, _, filenames in os.walk(image_directory):
        for fname in sorted(filenames):
            if not fname.lower().endswith((".jpg", ".jpeg")):
                continue

            base_name, _ = os.path.splitext(fname)
            parts = base_name.split("_")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                station_id = int(parts[0])
                ts = int(parts[1])
                camera_timestamps[station_id].append(ts)
                total_frames += 1

                try:
                    dt = datetime.fromtimestamp(ts, tz=VN_TZ)
                    hourly_counts[dt.hour] += 1
                    day_key = dt.strftime("%Y-%m-%d")
                    day_counts[day_key] += 1
                except Exception:
                    pass

    # Phân tích khoảng cách giữa các khung hình liên tiếp (Inter-frame intervals)
    all_deltas_sec: List[float] = []
    station_availabilities: List[float] = []

    for station_id, t_list in camera_timestamps.items():
        if len(t_list) >= 2:
            sorted_t = sorted(t_list)
            deltas = np.diff(sorted_t)
            all_deltas_sec.extend(deltas.tolist())

    mean_delta = float(np.mean(all_deltas_sec)) if all_deltas_sec else 0.0
    median_delta = float(np.median(all_deltas_sec)) if all_deltas_sec else 0.0
    std_delta = float(np.std(all_deltas_sec)) if all_deltas_sec else 0.0

    # Tỷ lệ Ngày (06h - 18h) vs Đêm (18h - 06h)
    day_frames = sum(hourly_counts[h] for h in range(6, 18))
    night_frames = total_frames - day_frames
    day_pct = (day_frames / total_frames * 100.0) if total_frames > 0 else 0.0
    night_pct = (night_frames / total_frames * 100.0) if total_frames > 0 else 0.0

    report = {
        "total_analyzed_frames": total_frames,
        "active_stations": len(camera_timestamps),
        "mean_sampling_interval_sec": mean_delta,
        "median_sampling_interval_sec": median_delta,
        "std_sampling_interval_sec": std_delta,
        "daytime_frames": day_frames,
        "daytime_pct": day_pct,
        "nighttime_frames": night_frames,
        "nighttime_pct": night_pct,
        "daily_distribution": dict(day_counts),
        "hourly_distribution": hourly_counts,
    }

    print("\n--- Temporal Coverage Audit Summary ---")
    print(f"Total Analyzed Frames:        {total_frames}")
    print(f"Active Camera Stations:       {len(camera_timestamps)}")
    print(f"Sampling Interval (Mean/Med): {mean_delta:.1f}s / {median_delta:.1f}s (+/- {std_delta:.1f}s)")
    print(f"Daytime (06:00-18:00):        {day_frames} ({day_pct:.1f}%)")
    print(f"Nighttime (18:00-06:00):      {night_frames} ({night_pct:.1f}%)")
    print(f"Calendar Day Coverage:        {list(day_counts.keys())}")
    return report


if __name__ == "__main__":
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "g:/nckh/DINO/direction_data_article/zenodo_bundle/sample_preview/sample_camera_sequences"
    audit_temporal_distribution(test_dir)
