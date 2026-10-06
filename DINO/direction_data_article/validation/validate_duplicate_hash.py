"""
=============================================================================
Module: validate_duplicate_hash.py
Nghiệp vụ: Technical Validation V2 - Phát hiện ảnh lặp / Camera bị đơ (Frozen)
           thông qua đối chiếu mã băm SHA-256 của các khung hình liên tiếp.
=============================================================================
"""

import os
import sys
import hashlib
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def calculate_sha256(file_path: str) -> str:
    """Tính mã băm SHA-256 của tệp hình ảnh."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_frozen_cameras(image_directory: str) -> Dict[str, Any]:
    """
    Quét mã băm của từng trạm camera qua chuỗi thời gian để phát hiện
    các khung hình bị lặp (stalled/frozen frames do kết nối máy chủ camera).
    """
    print(f"[Duplicate & Frozen Camera Audit] Scanning hashes in: {image_directory}")
    camera_frames: Dict[int, List[Tuple[int, str, str]]] = defaultdict(list)
    total_scanned = 0

    for root_dir, _, filenames in os.walk(image_directory):
        for fname in sorted(filenames):
            if fname.lower().endswith((".jpg", ".jpeg")):
                total_scanned += 1
                base_name, _ = os.path.splitext(fname)
                parts = base_name.split("_")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    station_id = int(parts[0])
                    ts = int(parts[1])
                    fp = os.path.join(root_dir, fname)
                    file_hash = calculate_sha256(fp)
                    camera_frames[station_id].append((ts, fp, file_hash))

    duplicate_pairs = 0
    total_consecutive_checks = 0
    frozen_stations: List[int] = []

    for station_id, frame_records in camera_frames.items():
        if len(frame_records) < 2:
            continue
        # Sắp xếp theo timestamp
        sorted_records = sorted(frame_records, key=lambda x: x[0])
        stn_dup = 0
        for i in range(len(sorted_records) - 1):
            total_consecutive_checks += 1
            if sorted_records[i][2] == sorted_records[i + 1][2]:
                duplicate_pairs += 1
                stn_dup += 1

        if stn_dup > 0:
            frozen_stations.append(station_id)

    dup_rate = (duplicate_pairs / total_consecutive_checks * 100.0) if total_consecutive_checks > 0 else 0.0

    report = {
        "total_scanned_images": total_scanned,
        "consecutive_comparisons": total_consecutive_checks,
        "identical_consecutive_frames": duplicate_pairs,
        "consecutive_duplicate_rate_pct": dup_rate,
        "stations_with_duplicates": len(frozen_stations),
    }

    print("\n--- Duplicate & Frozen Frame Audit Summary ---")
    print(f"Total Scanned Frames:          {total_scanned}")
    print(f"Consecutive Frame Pairs:       {total_consecutive_checks}")
    print(f"Identical Consecutive Pairs:   {duplicate_pairs} ({dup_rate:.2f}%)")
    print(f"Stations with Duplicate Drops: {len(frozen_stations)}")
    return report


if __name__ == "__main__":
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "g:/nckh/DINO/direction_data_article/zenodo_bundle/sample_preview/sample_camera_sequences"
    audit_frozen_cameras(test_dir)
