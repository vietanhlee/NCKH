"""
=============================================================================
 Hướng C: Anomaly Detection — Persistence Filtering & Event Tracking
 Bộ lọc tính kéo dài thời gian (Persistence Filter) và Quản lý Vòng đời Sự kiện
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class PersistenceEventTracker:
    """
    Bộ theo dõi và lọc tính kéo dài thời gian cho sự cố bất thường:
    - Bỏ qua các đột biến thoáng qua (xe tải siêu trường siêu trọng chạy lướt qua).
    - Chỉ báo động khi bất thường kéo dài >= N cửa sổ quan sát liên tiếp.
    - Ngưỡng tự động theo phân vị 99.5% từ dữ liệu bình thường.
    """

    def __init__(
        self,
        min_consecutive_windows: int = 3,
        percentile_threshold: float = 99.5,
    ):
        """
        Args:
            min_consecutive_windows: Số cửa sổ liên tiếp tối thiểu N để xác nhận sự kiện (mặc định 3).
            percentile_threshold: Phân vị dùng để chọn ngưỡng tự động (mặc định 99.5%).
        """
        self.min_consecutive = min_consecutive_windows
        self.percentile = percentile_threshold

        self.threshold: Optional[float] = None
        self.consecutive_count: int = 0
        self.active_event: Optional[Dict[str, Union[int, float, str]]] = None
        self.confirmed_events: List[Dict[str, Union[int, float, str]]] = []

    def calibrate_threshold(self, normal_scores: np.ndarray) -> float:
        """
        Tính toán ngưỡng phân vị 99.5% trên tập dữ liệu bình thường.
        """
        assert len(normal_scores) > 0, "Tập điểm bình thường không được rỗng."
        self.threshold = float(np.percentile(normal_scores, self.percentile))
        return self.threshold

    def update(
        self,
        step_idx: int,
        score: float,
        event_type: str = "TRAFFIC_INCIDENT",
        metadata: Optional[Dict] = None
    ) -> Optional[Dict[str, Union[int, float, str]]]:
        """
        Cập nhật điểm bất thường tại bước thời gian hiện tại:
        
        Args:
            step_idx: Chỉ số thời gian (hoặc timestamp)
            score: Điểm bất thường (road_score hoặc static_score)
            event_type: Loại sự kiện ("TRAFFIC_INCIDENT" hoặc "CAMERA_FAULT")
            metadata: Thông tin thêm (tọa độ, frame ID, v.v.)
            
        Returns:
            event_alert: Dict thông tin sự kiện nếu sự kiện được kích hoạt mới hoặc đang diễn ra, ngược lại None.
        """
        if self.threshold is None:
            raise RuntimeError("Cần gọi calibrate_threshold() trước khi theo dõi sự kiện.")

        is_exceeded = score >= self.threshold

        if is_exceeded:
            self.consecutive_count += 1

            if self.active_event is None:
                # Bắt đầu chuỗi ứng viên sự kiện
                self.active_event = {
                    "start_step": step_idx,
                    "event_type": event_type,
                    "max_score": score,
                    "consecutive_steps": self.consecutive_count,
                    "status": "CANDIDATE",
                    "metadata": metadata or {},
                }
            else:
                self.active_event["consecutive_steps"] = self.consecutive_count
                self.active_event["max_score"] = max(self.active_event["max_score"], score)

            # Đạt đủ điều kiện số cửa sổ kéo dài liên tiếp >= N
            if self.consecutive_count >= self.min_consecutive:
                if self.active_event["status"] == "CANDIDATE":
                    self.active_event["status"] = "CONFIRMED"
                    self.active_event["confirmed_step"] = step_idx
                    self.confirmed_events.append(self.active_event)

                return self.active_event

        else:
            # Ngắt quãng chuỗi bất thường -> Kết thúc sự kiện nếu có
            if self.active_event is not None and self.active_event["status"] == "CONFIRMED":
                self.active_event["end_step"] = step_idx - 1
                self.active_event["status"] = "RESOLVED"

            self.consecutive_count = 0
            self.active_event = None

        return None
