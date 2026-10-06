"""
=============================================================================
Module: validate_visual_diversity.py
Nghiệp vụ: Technical Validation V3 - Đánh giá tính đa dạng quang học
           (Photometric Diversity, RMS Contrast, Shannon Entropy)
           phục vụ huấn luyện mô hình nền tảng thị giác (Vision Foundation Models).
=============================================================================
"""

import os
import sys
from typing import Dict, List, Any, Tuple
from PIL import Image
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def calculate_frame_metrics(img_pil: Image.Image) -> Tuple[float, float, float, Tuple[float, float, float]]:
    """
    Tính toán các chỉ số quang học của một khung hình:
      1. Perceived Luminance Y (0 - 255)
      2. RMS Contrast
      3. Shannon Entropy (bits)
      4. RGB Mean values

    Returns:
        Tuple: (luminance, rms_contrast, entropy, (mean_r, mean_g, mean_b))
    """
    img_rgb = img_pil.convert("RGB")
    arr = np.asarray(img_rgb, dtype=np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mean_r, mean_g, mean_b = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))

    # Luminance theo chuẩn ITU-R BT.601
    luminance_map = 0.299 * r + 0.587 * g + 0.114 * b
    mean_lum = float(np.mean(luminance_map))

    # RMS Contrast: độ lệch chuẩn của luminance
    rms_contrast = float(np.std(luminance_map))

    # Shannon Entropy trên ảnh grayscale
    gray_uint8 = np.clip(luminance_map, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(gray_uint8, bins=256, range=(0, 256), density=True)
    # Loại bỏ các bin bằng 0
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    return mean_lum, rms_contrast, entropy, (mean_r, mean_g, mean_b)


def run_visual_diversity_audit(image_directory: str) -> Dict[str, Any]:
    """
    Quét thư mục ảnh và tính toán phân phối các chỉ số quang học toàn cục.
    """
    print(f"[Visual Diversity Audit] Measuring photometric properties in: {image_directory}")
    luminances: List[float] = []
    contrasts: List[float] = []
    entropies: List[float] = []
    r_means: List[float] = []
    g_means: List[float] = []
    b_means: List[float] = []

    count = 0
    for root_dir, _, filenames in os.walk(image_directory):
        for fname in sorted(filenames):
            if fname.lower().endswith((".jpg", ".jpeg")):
                fp = os.path.join(root_dir, fname)
                try:
                    with Image.open(fp) as img:
                        lum, cont, ent, (mr, mg, mb) = calculate_frame_metrics(img)
                        luminances.append(lum)
                        contrasts.append(cont)
                        entropies.append(ent)
                        r_means.append(mr)
                        g_means.append(mg)
                        b_means.append(mb)
                        count += 1
                except Exception as exc:
                    print(f"Warning: Failed to process {fp}: {exc}")

    if count == 0:
        print("No valid images found!")
        return {}

    lum_arr = np.array(luminances)
    cont_arr = np.array(contrasts)
    ent_arr = np.array(entropies)

    report = {
        "total_analyzed_frames": count,
        "mean_luminance": float(np.mean(lum_arr)),
        "std_luminance": float(np.std(lum_arr)),
        "min_luminance": float(np.min(lum_arr)),
        "max_luminance": float(np.max(lum_arr)),
        "mean_rms_contrast": float(np.mean(cont_arr)),
        "std_rms_contrast": float(np.std(cont_arr)),
        "mean_shannon_entropy": float(np.mean(ent_arr)),
        "std_shannon_entropy": float(np.std(ent_arr)),
        "channel_means_rgb": (float(np.mean(r_means)), float(np.mean(g_means)), float(np.mean(b_means))),
    }

    print("\n--- Visual Diversity Audit Summary ---")
    print(f"Total Evaluated Frames:  {count}")
    print(f"Mean Luminance (0-255):  {report['mean_luminance']:.2f} +/- {report['std_luminance']:.2f}")
    print(f"Luminance Range:         [{report['min_luminance']:.2f}, {report['max_luminance']:.2f}]")
    print(f"Mean RMS Contrast:       {report['mean_rms_contrast']:.2f} +/- {report['std_rms_contrast']:.2f}")
    print(f"Shannon Entropy (bits):  {report['mean_shannon_entropy']:.2f} +/- {report['std_shannon_entropy']:.2f}")
    print(f"Channel Means (R, G, B): ({report['channel_means_rgb'][0]:.1f}, {report['channel_means_rgb'][1]:.1f}, {report['channel_means_rgb'][2]:.1f})")
    return report


if __name__ == "__main__":
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "g:/nckh/DINO/direction_data_article/zenodo_bundle/sample_preview/sample_camera_sequences"
    run_visual_diversity_audit(test_dir)
