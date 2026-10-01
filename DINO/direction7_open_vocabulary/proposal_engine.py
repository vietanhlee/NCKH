"""
=============================================================================
 Hướng 7: Open-Vocabulary Traffic Scene Understanding via Delta Proposals
 Module: Proposal Engine (Bộ sinh đề xuất vùng đối tượng không phụ thuộc lớp)
=============================================================================
"""

import sys
import os
from typing import List, Dict, Tuple, Any, Union
import numpy as np
import cv2
from PIL import Image

_dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)

from common.subtraction import BackgroundSubtractor


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 bounding box [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    union_area = (w1 * h1) + (w2 * h2) - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def apply_nms(boxes: List[Dict[str, Any]], iou_thresh: float = 0.5) -> List[Dict[str, Any]]:
    """Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các vùng đề xuất trùng lặp."""
    if not boxes:
        return []

    # Sắp xếp theo score giảm dần
    sorted_boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)
    kept = []

    while sorted_boxes:
        best = sorted_boxes.pop(0)
        kept.append(best)
        sorted_boxes = [b for b in sorted_boxes if compute_iou(best["bbox"], b["bbox"]) < iou_thresh]

    return kept


class DeltaProposalEngine:
    """
    Động cơ sinh đề xuất vùng đối tượng vật lý (Delta Region Proposal Engine).
    Sử dụng trường sai khác quang học và gradient không gian để sinh ra các hộp ứng viên
    chứa phương tiện / chướng ngại vật tiềm năng trong cảnh giao thông.
    """

    def __init__(
        self,
        min_area: int = 300,
        max_area_ratio: float = 0.5,
        nms_threshold: float = 0.45,
        target_crop_size: Tuple[int, int] = (224, 224),
    ):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.nms_threshold = nms_threshold
        self.target_crop_size = target_crop_size
        self.subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)

    def generate_proposals(
        self,
        origin_img: Union[np.ndarray, Image.Image],
        bg_img: Union[np.ndarray, Image.Image],
        max_proposals: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Sinh danh sách các vùng đề xuất (proposals) từ khung cảnh giao thông.

        Args:
            origin_img: Ảnh gốc giao thông (H, W, 3).
            bg_img: Ảnh nền tương ứng (H, W, 3).
            max_proposals: Số lượng đề xuất tối đa cần giữ lại.

        Returns:
            List các dict chứa:
                - 'bbox': [x, y, w, h]
                - 'crop_pil': Ảnh cắt đã resize về target_crop_size
                - 'score': Điểm tin cậy đề xuất (dựa trên năng lượng Delta và biên cạnh)
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

        # 2. Phân ngưỡng thích nghi đa mức (Multi-thresholding) để bắt cả xe lớn và xe nhỏ
        masks = []
        # Ngưỡng Otsu chuẩn
        mask_otsu = self.subtractor.extract_binary_mask(delta_norm, method="otsu")
        masks.append(mask_otsu)

        # Ngưỡng nhạy biên (fixed 0.15)
        mask_sensitive = (delta_norm > 0.15).astype(np.uint8) * 255
        masks.append(mask_sensitive)

        combined_mask = np.maximum(masks[0], masks[1])

        # 3. Làm mịn hình thái học
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # 4. Tìm đường bao (Contours)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_frame_area = H * W
        max_area = total_frame_area * self.max_area_ratio

        raw_proposals = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if area < self.min_area or area > max_area:
                continue

            aspect_ratio = float(w) / max(1.0, float(h))
            if aspect_ratio < 0.25 or aspect_ratio > 4.0:
                continue

            # Mở rộng 5% biên
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
                (self.target_crop_size[1], self.target_crop_size[0]),
                Image.BILINEAR,
            )

            # Điểm chất lượng đề xuất: mật độ delta trung bình * log(area)
            delta_crop = delta_norm[y1:y2, x1:x2]
            score = float(np.mean(delta_crop)) * np.log1p(area)

            raw_proposals.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "crop_pil": crop_pil,
                "score": score,
                "area": area,
            })

        # 5. Khử hộp trùng lặp bằng NMS
        filtered_proposals = apply_nms(raw_proposals, iou_thresh=self.nms_threshold)
        return filtered_proposals[:max_proposals]
