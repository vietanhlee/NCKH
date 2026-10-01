"""
=============================================================================
 Hướng 5: Unsupervised Traffic Anomaly Detection
 Module: Detector
 Thuật toán phát hiện bất thường dựa trên khoảng cách đặc trưng
=============================================================================
"""

import sys
import os
from typing import List, Tuple
import numpy as np
import torch
import logging

# Import local modules
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from memory_bank import AnomalyMemoryBank
from feature_extractor import DeltaConditionedExtractor


class TrafficAnomalyDetector:
    """
    Trình phát hiện bất thường luồng giao thông (Traffic Anomaly Detector).
    Quy trình:
      1. Khởi tạo bằng cách fit() trên tập ảnh bình thường (Normal Images) để xây dựng memory bank.
      2. Suy luận bằng detect() để phát hiện tính bất thường thông qua anomaly score.
    """

    def __init__(self, extractor: DeltaConditionedExtractor, memory_bank: AnomalyMemoryBank, threshold: float = 2.0):
        """
        Khởi tạo Detector.

        Args:
            extractor: Đối tượng DeltaConditionedExtractor trích xuất đặc trưng.
            memory_bank: Đối tượng AnomalyMemoryBank để lưu trữ và so sánh.
            threshold: Ngưỡng điểm số để quyết định có phải bất thường không.
        """
        self.extractor = extractor
        self.memory_bank = memory_bank
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def fit(self, normal_images: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Xây dựng memory bank từ tập ảnh bình thường.

        Args:
            normal_images: Danh sách các tuple (origin_image, bg_image).
        """
        self.logger.info(f"Đang xây dựng Memory Bank với {len(normal_images)} mẫu bình thường...")
        for origin, bg in normal_images:
            try:
                features_dict = self.extractor.extract(origin, bg)
                combined_feat = features_dict['combined']
                self.memory_bank.update(combined_feat.cpu())
            except Exception as e:
                self.logger.warning(f"Lỗi khi trích xuất đặc trưng trong lúc fit: {e}")
        self.logger.info("Hoàn tất xây dựng Memory Bank.")

    def detect(self, origin: np.ndarray, bg: np.ndarray) -> Tuple[bool, float]:
        """
        Phát hiện bất thường cho một cặp ảnh.

        Args:
            origin: Ảnh giao thông hiện tại.
            bg: Ảnh nền tương ứng.

        Returns:
            is_anomaly: True nếu là bất thường, False nếu bình thường.
            score: Giá trị anomaly score.
        """
        try:
            features_dict = self.extractor.extract(origin, bg)
            combined_feat = features_dict['combined'].unsqueeze(0).cpu() # (1, D+3)
            
            score_tensor = self.memory_bank.compute_anomaly_score(combined_feat)
            score = score_tensor.item()
            is_anomaly = score > self.threshold
            return is_anomaly, score
        except Exception as e:
            self.logger.error(f"Lỗi khi phát hiện bất thường: {e}")
            return False, 0.0

    def detect_batch(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[bool, float]]:
        """
        Phát hiện bất thường cho một lô (batch) cặp ảnh.

        Args:
            pairs: Danh sách các tuple (origin, bg).

        Returns:
            Danh sách các tuple (is_anomaly, score).
        """
        results = []
        for origin, bg in pairs:
            results.append(self.detect(origin, bg))
        return results
