"""
=============================================================================
 Hướng B: Weak Supervision — LF1: Detector-Based Labeling Function
 Trích xuất mức ùn tắc từ mật độ bounding boxes phương tiện trong Road Mask
 Trả về: 0 (Free), 1 (Moderate), 2 (Slow), 3 (Jammed), hoặc -1 (Abstain)
=============================================================================
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple


class DetectorLabelingFunction:
    """
    LF1: Ước lượng mức độ ùn tắc từ kết quả phát hiện phương tiện (YOLOv8/RT-DETR).
    Tỷ lệ mật độ box = số box trong road mask / (diện tích road / 10^4 px).
    Bỏ qua (Abstain = -1) khi độ tin cậy trung bình < conf_threshold.
    """

    def __init__(
        self,
        conf_threshold: float = 0.35,
        thresholds: Tuple[float, float, float] = (1.5, 3.5, 6.0),
    ):
        self.conf_threshold = conf_threshold
        self.th1, self.th2, self.th3 = thresholds

    def __call__(
        self,
        boxes: List[List[float]],
        confidences: List[float],
        road_area_sq_px: float = 10000.0,
        is_ir: bool = False,
    ) -> Tuple[int, float]:
        """
        Args:
            boxes: Danh sách tọa độ [x1, y1, x2, y2] của xe trong road mask.
            confidences: Danh sách điểm tin cậy tương ứng.
            road_area_sq_px: Diện tích lòng đường (pixel vuông).
            is_ir: Cờ chế độ hồng ngoại ban đêm.

        Returns:
            label: 0..3 hoặc -1 (Abstain).
            raw_density: Mật độ thô (số xe / 10^4 px).
        """
        if len(confidences) == 0:
            if is_ir:
                # Ban đêm hồng ngoại nếu 0 box có thể do quá tối không nhìn thấy
                return -1, 0.0
            return 0, 0.0

        mean_conf = float(np.mean(confidences))
        if mean_conf < self.conf_threshold:
            # Độ tin cậy trung bình quá thấp -> Abstain
            return -1, float(len(boxes))

        # Mật độ xe trên đơn vị 10^4 px lòng đường
        raw_density = float(len(boxes) / max(1.0, (road_area_sq_px / 10000.0)))

        if raw_density <= self.th1:
            label = 0
        elif raw_density <= self.th2:
            label = 1
        elif raw_density <= self.th3:
            label = 2
        else:
            label = 3

        return label, raw_density
