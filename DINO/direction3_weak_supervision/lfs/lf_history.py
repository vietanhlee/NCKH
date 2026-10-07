"""
=============================================================================
 Hướng B: Weak Supervision — LF5: Historical Slot Labeling Function
 Mức ùn tắc phổ biến nhất trong lịch sử cùng camera x khung giờ x thứ trong tuần
=============================================================================
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple


class HistoricalLabelingFunction:
    """
    LF5: Nhãn tiên nghiệm từ mẫu lưu thông lịch sử.
    Bỏ qua nếu camera có ít hơn min_historical_days ngày dữ liệu.
    """

    def __init__(self, min_historical_days: int = 5):
        self.min_days = min_historical_days

    def __call__(
        self,
        historical_labels: List[int],
    ) -> Tuple[int, float]:
        valid = [l for l in historical_labels if l >= 0]
        if len(valid) < self.min_days:
            return -1, float(len(valid))

        counter = Counter(valid)
        most_common, count = counter.most_common(1)[0]
        agreement_ratio = count / len(valid)

        # Chỉ tin cậy khi có trên 60% số ngày đồng thuận
        if agreement_ratio >= 0.60:
            return most_common, float(agreement_ratio)

        return -1, float(agreement_ratio)
