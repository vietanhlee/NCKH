"""
=============================================================================
 Hướng B: Weak Supervision — LF4: Temporal Similarity Labeling Function
 Tận dụng độ tương đồng cosine đặc trưng DINO giữa các khung hình liên tiếp
 Kết hợp mật độ xe: Tĩnh + nhiều xe -> Kẹt; Tĩnh + ít xe -> Vắng
=============================================================================
"""

import numpy as np
from typing import Tuple, Optional


class TemporalLabelingFunction:
    """
    LF4: Phân tích động thái thay đổi qua thời gian.
    """

    def __init__(
        self,
        still_threshold: float = 0.92,
        time_gap_max_sec: float = 300.0,
    ):
        self.still_threshold = still_threshold
        self.max_gap = time_gap_max_sec

    def __call__(
        self,
        cosine_sim: float,
        dt_seconds: float,
        detector_raw: float = 0.0,
        density_median: float = 3.0,
    ) -> Tuple[int, float]:
        """
        Args:
            cosine_sim: Độ tương đồng cosine giữa đặc trưng 2 frame liên tiếp [0, 1].
            dt_seconds: Khoảng cách thời gian (giây).
            detector_raw: Mật độ xe thô từ LF1.
            density_median: Mật độ xe trung vị của camera.

        Returns:
            label: 0..3 hoặc -1 (Abstain).
            raw_metric: Giá trị cosine_sim.
        """
        if dt_seconds > self.max_gap or dt_seconds <= 0:
            # Khoảng cách quá xa không thể phân tích thời gian -> Abstain
            return -1, float(cosine_sim)

        # Cảnh hầu như bất động
        if cosine_sim >= self.still_threshold:
            if detector_raw >= density_median:
                # Đứng yên nhưng nhiều xe -> Kẹt cứng (Jammed)
                return 3, float(cosine_sim)
            elif detector_raw <= (density_median * 0.3):
                # Đứng yên nhưng mặt đường trống -> Thông thoáng (Free-flow)
                return 0, float(cosine_sim)

        return -1, float(cosine_sim)
