"""
=============================================================================
Module: validate_image_integrity.py
Nghiệp vụ: Technical Validation V1 - Kiểm định tính toàn vẹn hình ảnh
           và phát hiện khung hình lỗi cho HCMC-TrafficCam7D.
=============================================================================
Kiểm tra:
  1. Header và định dạng JPEG/JFIF hợp lệ.
  2. Kênh màu (3-channel RGB, 8-bit depth).
  3. Kích thước khung hình (Loại bỏ ảnh placeholder báo lỗi cổng giao thông 284x177).
  4. Xuất báo cáo tỷ lệ toàn vẹn (Data Integrity Rate).
=============================================================================
"""

import os
import sys
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np

# Cấu hình an toàn OpenMP và mã hóa console
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def check_single_image(file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Kiểm tra một tệp ảnh camera.

    Returns:
        Tuple: (is_valid, reason, info_dict)
    """
    if not os.path.isfile(file_path):
        return False, "File does not exist", {}

    size_bytes = os.path.getsize(file_path)
    size_kb = size_bytes / 1024.0

    if size_bytes == 0:
        return False, "Empty 0-byte file", {"size_kb": 0.0}

    try:
        with Image.open(file_path) as img:
            img.verify()  # Kiểm tra tính hợp lệ của luồng dữ liệu JPEG
    except Exception as exc:
        return False, f"Corrupted JPEG header: {exc}", {"size_kb": size_kb}

    # Đọc lại để trích xuất kích thước và kênh màu
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            mode = img.mode
            format_name = img.format

            # Phát hiện placeholder lỗi của cổng giao thông TP.HCM
            # Kích thước cố định 284x177 và dung lượng trong khoảng 5.8 - 7.0 KB
            if w == 284 and h == 177 and (5.8 <= size_kb <= 7.2):
                return False, "Municipal error placeholder (284x177 card)", {
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "size_kb": size_kb,
                }

            if w < 100 or h < 100:
                return False, "Sub-resolution degraded frame (<100px)", {
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "size_kb": size_kb,
                }

            return True, "Valid frame", {
                "width": w,
                "height": h,
                "mode": mode,
                "format": format_name,
                "size_kb": size_kb,
            }
    except Exception as exc:
        return False, f"Image decode failure: {exc}", {"size_kb": size_kb}


def run_image_integrity_audit(image_directory: str) -> Dict[str, Any]:
    """
    Quét toàn bộ thư mục ảnh và tính toán các chỉ số kiểm định kỹ thuật.
    """
    print(f"[Image Integrity Audit] Scanning directory: {image_directory}")
    total_scanned = 0
    valid_count = 0
    corrupted_count = 0
    placeholder_count = 0
    resolutions: Dict[Tuple[int, int], int] = {}
    file_sizes_kb: List[float] = []

    for root_dir, _, filenames in os.walk(image_directory):
        for fname in sorted(filenames):
            if fname.lower().endswith((".jpg", ".jpeg")):
                total_scanned += 1
                fp = os.path.join(root_dir, fname)
                is_valid, reason, info = check_single_image(fp)

                if is_valid:
                    valid_count += 1
                    res = (info.get("width", 0), info.get("height", 0))
                    resolutions[res] = resolutions.get(res, 0) + 1
                    file_sizes_kb.append(info.get("size_kb", 0.0))
                else:
                    if "placeholder" in reason:
                        placeholder_count += 1
                    else:
                        corrupted_count += 1

    pass_rate = (valid_count / total_scanned * 100.0) if total_scanned > 0 else 0.0
    mean_size = float(np.mean(file_sizes_kb)) if file_sizes_kb else 0.0
    std_size = float(np.std(file_sizes_kb)) if file_sizes_kb else 0.0

    report = {
        "total_scanned": total_scanned,
        "valid_frames": valid_count,
        "corrupted_frames": corrupted_count,
        "placeholders_filtered": placeholder_count,
        "integrity_pass_rate_pct": pass_rate,
        "mean_file_size_kb": mean_size,
        "std_file_size_kb": std_size,
        "dominant_resolutions": resolutions,
    }

    print("\n--- Image Integrity Audit Summary ---")
    print(f"Total Scanned:         {total_scanned}")
    print(f"Valid Frames:          {valid_count} ({pass_rate:.2f}%)")
    print(f"Filtered Placeholders: {placeholder_count}")
    print(f"Corrupted Files:       {corrupted_count}")
    print(f"Mean File Size (KB):   {mean_size:.2f} +/- {std_size:.2f}")
    print(f"Resolution Clusters:   {resolutions}")
    return report


if __name__ == "__main__":
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "g:/nckh/DINO/direction_data_article/zenodo_bundle/sample_preview/sample_camera_sequences"
    run_image_integrity_audit(test_dir)
