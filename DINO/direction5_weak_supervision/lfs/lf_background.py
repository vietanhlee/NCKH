"""
=============================================================================
 Hướng B: Weak Supervision — LF2: Background-Based Labeling Function
 Trích xuất mức ùn tắc từ tỷ lệ chiếm dụng lòng đường rho_proxy
 Tự động Bỏ qua (Abstain = -1) khi độ tin cậy r_i < 0.5 hoặc camera bị lệch
=============================================================================
"""

from typing import Tuple, Optional


class BackgroundLabelingFunction:
    """
    LF2: Tận dụng tín hiệu tiên nghiệm quang học từ background subtraction.
    Bỏ qua khi r_i < 0.5 hoặc align_ok = False.
    """

    def __init__(
        self,
        reliability_threshold: float = 0.50,
        thresholds: Tuple[float, float, float] = (0.15, 0.35, 0.60),
    ):
        self.r_threshold = reliability_threshold
        self.th1, self.th2, self.th3 = thresholds

    def __call__(
        self,
        rho_proxy: float,
        reliability_r_i: float = 1.0,
        align_ok: bool = True,
        has_bg: bool = True,
    ) -> Tuple[int, float]:
        """
        Returns:
            label: 0..3 hoặc -1 (Abstain).
            raw_rho: Giá trị chiếm dụng thô rho_proxy.
        """
        if not has_bg or not align_ok or reliability_r_i < self.r_threshold:
            # Tín hiệu background không đáng tin -> Tự động bỏ qua (Abstain)
            return -1, float(rho_proxy)

        if rho_proxy <= self.th1:
            label = 0
        elif rho_proxy <= self.th2:
            label = 1
        elif rho_proxy <= self.th3:
            label = 2
        else:
            label = 3

        return label, float(rho_proxy)
