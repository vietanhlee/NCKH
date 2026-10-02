"""
=============================================================================
 Hướng B: Weak Supervision — Context Modeling
 Mã hóa biến ngữ cảnh c_t (Thời gian, Chế độ hồng ngoại, Loại đường, Mật độ)
 phục vụ học ma trận nhầm lẫn có điều kiện pi_j^(c) của từng Labeling Function
=============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class ContextEncoder:
    """
    Mã hóa ngữ cảnh ngoại cảnh giao thông c_t thành vector one-hot hoặc categorical index:
      - is_ir: 0 (Day/Color), 1 (Night/Infrared)
      - hour_group: 0 (Đêm 22h-5h), 1 (Thấp điểm 9h-16h & 20h-21h), 2 (Cao điểm 6h-8h & 17h-19h)
      - road_type: 0 (Trục chính), 1 (Nút giao), 2 (Đường nhỏ)
      - density_bin: 0 (Thưa), 1 (Trung bình), 2 (Đông)
    Tổng số tổ hợp: 2 x 3 x 3 x 3 = 54 tổ hợp ngữ cảnh.
    """

    def __init__(self):
        self.num_contexts = 54

    @staticmethod
    def get_hour_group(hour: int) -> int:
        if 22 <= hour or hour <= 5:
            return 0  # Đêm
        elif (6 <= hour <= 8) or (17 <= hour <= 19):
            return 2  # Giờ cao điểm
        else:
            return 1  # Thấp điểm ban ngày

    def encode(
        self,
        hour: int,
        is_ir: bool = False,
        road_type: int = 0,
        density_bin: int = 1,
    ) -> Tuple[int, np.ndarray]:
        """
        Returns:
            context_id: Số nguyên trong khoảng [0, 53].
            one_hot: Mảng numpy shape (54,) chứa 1.0 tại context_id.
        """
        ir_idx = 1 if is_ir else 0
        h_idx = self.get_hour_group(hour)
        r_idx = max(0, min(road_type, 2))
        d_idx = max(0, min(density_bin, 2))

        # Tính flat index
        ctx_id = ir_idx * 27 + h_idx * 9 + r_idx * 3 + d_idx
        one_hot = np.zeros(self.num_contexts, dtype=np.float32)
        one_hot[ctx_id] = 1.0
        return ctx_id, one_hot

    def get_context_id(
        self,
        hour: int = 12,
        is_night: bool = False,
        is_rain: bool = False,
        is_major_artery: bool = True,
        reliability_score: float = 1.0,
    ) -> int:
        """Helper tiện dụng trả về nhanh context_id nguyên [0, 53]."""
        road_type = 0 if is_major_artery else 1
        density_bin = 1
        ctx_id, _ = self.encode(hour=hour, is_ir=is_night, road_type=road_type, density_bin=density_bin)
        return ctx_id


# Alias cho tương thích ngược
TrafficContextClassifier = ContextEncoder

