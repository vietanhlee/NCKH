"""
=============================================================================
 Hướng 6: Delta-Guided Unsupervised Vehicle Re-Identification Across Cameras
 Module: RoI Extractor (Trích xuất vùng phương tiện tự động từ bản đồ sai khác)
=============================================================================
"""

import sys
import os
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import cv2
from PIL import Image

# Import common utilities
_dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)

from common.subtraction import BackgroundSubtractor


class DeltaRoIExtractor:
    """
    Bộ trích xuất vùng quan tâm phương tiện (Vehicle Region of Interest - RoI Extractor).
    Khai thác trực tiếp tín hiệu vật lý Delta = ||I_origin - I_bg|| để định vị và cắt các
    phương tiện di chuyển trên đường mà không cần sử dụng mô hình Object Detection nặng.
    """

    def __init__(
        self,
        min_area: int = 400,
        max_area_ratio: float = 0.4,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 3.5,
        target_size: Tuple[int, int] = (256, 128),
        blur_kernel: int = 5,
        threshold_method: str = "otsu",
    ):
        """
        Khởi tạo RoI Extractor.

        Args:
            min_area: Diện tích tối thiểu (pixel) để coi là một phương tiện hợp lệ.
            max_area_ratio: Tỷ lệ diện tích tối đa so với toàn khung hình (lọc xe quá gần/nhiễu nền).
            min_aspect_ratio: Tỷ lệ chiều rộng/chiều cao tối thiểu (w/h).
            max_aspect_ratio: Tỷ lệ chiều rộng/chiều cao tối đa (w/h).
            target_size: Kích thước đầu ra chuẩn của ảnh crop (H, W) cho Re-ID (mặc định 256x128).
            blur_kernel: Kích thước kernel làm mịn.
            threshold_method: Phương pháp phân ngưỡng ("otsu" hoặc "adaptive").
        """
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.target_size = target_size
        self.subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=blur_kernel)
        self.threshold_method = threshold_method

    def extract_rois(
        self,
        origin_img: Union[np.ndarray, Image.Image],
        bg_img: Union[np.ndarray, Image.Image],
    ) -> List[Dict[str, Any]]:
        """
        Trích xuất danh sách các phương tiện từ cặp ảnh origin và background.

        Args:
            origin_img: Ảnh gốc có phương tiện (H, W, 3).
            bg_img: Ảnh nền tĩnh tương ứng (H, W, 3).

        Returns:
            List các dict chứa:
                - 'bbox': [x, y, w, h] (tọa độ pixel).
                - 'crop_pil': PIL Image của phương tiện đã resize về target_size.
                - 'area': Diện tích vùng chuyển động.
                - 'confidence': Điểm tin cậy quang học (cường độ sai khác trung bình).
        """
        if isinstance(origin_img, Image.Image):
            origin_np = np.array(origin_img.convert("RGB"))
        else:
            origin_np = origin_img.copy()

        if isinstance(bg_img, Image.Image):
            bg_np = np.array(bg_img.convert("RGB"))
        else:
            bg_np = bg_img.copy()

        H, W = origin_np.shape[:2]
        if bg_np.shape[:2] != (H, W):
            bg_np = cv2.resize(bg_np, (W, H))

        # 1. Tính bản đồ sai khác Delta
        delta_norm, _ = self.subtractor.compute_delta(origin_np, bg_np)

        # 2. Phân ngưỡng nhị phân
        bin_mask = self.subtractor.extract_binary_mask(delta_norm, method=self.threshold_method)

        # 3. Phép toán hình thái học (Morphological Open & Close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_clean = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel_open)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)

        # 4. Phân tích thành phần liên thông (Connected Components)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

        total_frame_area = H * W
        max_allowed_area = total_frame_area * self.max_area_ratio

        rois = []
        for i in range(1, num_labels):  # Bỏ qua label 0 (nền)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Bộ lọc kích thước và tỷ lệ khung hình
            if area < self.min_area or area > max_allowed_area:
                continue

            aspect_ratio = float(w) / max(1.0, float(h))
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            # Mở rộng nhẹ bbox để không bị cắt xén viền xe (5% padding)
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.05)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(W, x + w + pad_x)
            y2 = min(H, y + h + pad_y)

            crop_np = origin_np[y1:y2, x1:x2]
            if crop_np.shape[0] < 8 or crop_np.shape[1] < 8:
                continue

            crop_pil = Image.fromarray(crop_np).resize(
                (self.target_size[1], self.target_size[0]),  # (W, H)
                Image.BILINEAR,
            )

            # Điểm tin cậy dựa trên cường độ delta trung bình trong bbox
            delta_roi = delta_norm[y1:y2, x1:x2]
            confidence = float(np.mean(delta_roi))

            rois.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "crop_pil": crop_pil,
                "area": area,
                "confidence": confidence,
            })

        # Sắp xếp các phương tiện theo diện tích giảm dần (ưu tiên xe rõ ràng)
        rois.sort(key=lambda r: r["area"], reverse=True)
        return rois
