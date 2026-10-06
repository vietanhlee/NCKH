"""
=============================================================================
Module: validate_blur_laplacian.py
Nghiệp vụ: Technical Validation V3 - Đo độ sắc nét và phát hiện ảnh mờ/tối
           bằng phương sai toán tử vi phân Laplacian (Laplacian Variance).
=============================================================================
"""

import os
import sys
from typing import Dict, List, Any, Tuple
from PIL import Image
import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def compute_laplacian_sharpness(img_path: str) -> Tuple[float, float, bool]:
    """
    Tính phương sai Laplacian và độ sáng trung bình.

    Returns:
        Tuple: (laplacian_variance, mean_luminance, is_acceptable)
    """
    img = cv2.imread(img_path)
    if img is None:
        return 0.0, 0.0, False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))

    # Tính tích chập Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(laplacian.var())

    # Tiêu chí: không phải ảnh tối đen (lum > 10) và không quá nhòe hoàn toàn (var > 20)
    is_acceptable = (mean_lum >= 10.0) and (lap_var >= 20.0)
    return lap_var, mean_lum, is_acceptable


def audit_image_sharpness(image_directory: str) -> Dict[str, Any]:
    """
    Đánh giá độ sắc nét của toàn bộ ảnh trong thư mục.
    """
    print(f"[Image Sharpness & Blur Audit] Processing directory: {image_directory}")
    lap_vars: List[float] = []
    acceptable_count = 0
    blurry_count = 0
    dark_count = 0
    total = 0

    for root_dir, _, filenames in os.walk(image_directory):
        for fname in sorted(filenames):
            if fname.lower().endswith((".jpg", ".jpeg")):
                total += 1
                fp = os.path.join(root_dir, fname)
                l_var, lum, is_acc = compute_laplacian_sharpness(fp)
                lap_vars.append(l_var)

                if lum < 10.0:
                    dark_count += 1
                elif l_var < 50.0:
                    blurry_count += 1
                else:
                    acceptable_count += 1

    mean_var = float(np.mean(lap_vars)) if lap_vars else 0.0
    median_var = float(np.median(lap_vars)) if lap_vars else 0.0
    std_var = float(np.std(lap_vars)) if lap_vars else 0.0

    report = {
        "total_analyzed": total,
        "sharp_acceptable_frames": acceptable_count,
        "acceptable_ratio_pct": (acceptable_count / total * 100.0) if total > 0 else 0.0,
        "blurry_frames": blurry_count,
        "severely_dark_frames": dark_count,
        "mean_laplacian_variance": mean_var,
        "median_laplacian_variance": median_var,
        "std_laplacian_variance": std_var,
    }

    print("\n--- Image Sharpness Audit Summary ---")
    print(f"Total Analyzed Frames:       {total}")
    print(f"Acceptable Sharp Frames:     {acceptable_count} ({report['acceptable_ratio_pct']:.2f}%)")
    print(f"Blurry Frames (Var < 50):    {blurry_count}")
    print(f"Severe Dark Frames (Lum<10): {dark_count}")
    print(f"Laplacian Variance (Mean):   {mean_var:.2f} +/- {std_var:.2f} (Median: {median_var:.2f})")
    return report


if __name__ == "__main__":
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "g:/nckh/DINO/direction_data_article/zenodo_bundle/sample_preview/sample_camera_sequences"
    audit_image_sharpness(test_dir)
