"""
=============================================================================
 Hướng C: Anomaly Detection — Camera Fault Disentanglement
 Phân tách Sự cố Giao thông (Traffic Incident) và Lỗi Camera (Camera Fault)
 Dựa trên tỷ lệ bất thường giữa Lòng đường và Ngoại cảnh tĩnh
=============================================================================
"""

from typing import Dict, Tuple, Optional, Union
import torch


class CameraFaultClassifier:
    """
    Phân loại nguồn gốc bất thường:
    - Nếu vùng bất thường tập trung ở ngoại cảnh ngoài đường (tòa nhà, cột đèn, bầu trời):
      -> Camera bị xoay lệch góc, rung lắc mạnh, hoặc ống kính bị che phủ/nhiễu.
    - Nếu vùng bất thường chỉ tập trung trên mặt đường:
      -> Sự cố giao thông thực sự (ngập lụt, vật cản, tai nạn, xe dừng chết máy).
    """

    def __init__(
        self,
        fault_ratio_threshold: float = 1.3,
        min_static_score_threshold: float = 0.35,
    ):
        """
        Args:
            fault_ratio_threshold: Tỷ lệ (static_score / road_score) để khẳng định lỗi camera.
            min_static_score_threshold: Ngưỡng tối thiểu của static_score để xem xét lỗi camera.
        """
        self.fault_ratio_threshold = fault_ratio_threshold
        self.min_static_threshold = min_static_score_threshold

    def classify(
        self,
        road_score: float,
        static_score: float,
        road_threshold: float,
        static_threshold: float,
    ) -> Dict[str, Union[bool, str, float]]:
        """
        Đưa ra chẩn đoán sự kiện cho khung hình:
        
        Returns:
            dict chứa:
              - is_anomaly: Có bất thường hay không
              - event_type: "NORMAL", "TRAFFIC_INCIDENT", "CAMERA_FAULT"
              - confidence: Điểm tự tin
        """
        is_road_anomaly = road_score > road_threshold
        is_static_anomaly = static_score > static_threshold

        if not is_road_anomaly and not is_static_anomaly:
            return {
                "is_anomaly": False,
                "event_type": "NORMAL",
                "road_score": road_score,
                "static_score": static_score,
            }

        # Nếu vùng tĩnh ngoài đường bị biến đổi mạnh vượt trội so với mặt đường
        if is_static_anomaly and (
            (static_score / (road_score + 1e-5) >= self.fault_ratio_threshold)
            or (static_score >= self.min_static_threshold and not is_road_anomaly)
        ):
            return {
                "is_anomaly": True,
                "event_type": "CAMERA_FAULT",
                "road_score": road_score,
                "static_score": static_score,
            }

        # Nếu chỉ có mặt đường bị bất thường
        if is_road_anomaly:
            return {
                "is_anomaly": True,
                "event_type": "TRAFFIC_INCIDENT",
                "road_score": road_score,
                "static_score": static_score,
            }

        return {
            "is_anomaly": False,
            "event_type": "NORMAL",
            "road_score": road_score,
            "static_score": static_score,
        }
