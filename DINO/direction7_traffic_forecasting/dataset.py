"""
=============================================================================
 Hướng D: Traffic Forecasting — Spatio-Temporal Dataset & Chronological Split
 Quản lý Dữ liệu Chuỗi Thời gian Đồ thị Camera trên lưới đều Delta t = 5 phút
 Hỗ trợ Mặt nạ mất tín hiệu (Missing Mask) và Che nút ngẫu nhiên (Node Dropout)
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SpatioTemporalTrafficDataset(Dataset):
    """
    Dataset chuỗi thời gian không gian mạng lưới camera:
      - X: Tensor (T_total, num_nodes, num_features)
           features bao gồm: [occupancy_scalar, dino_cls_features (e.g. 32-dim), time_of_day_sin/cos]
      - M: Tensor (T_total, num_nodes) - Mặt nạ nhị phân (1: quan sát được, 0: mất tín hiệu/rớt mạng)
      - T_in: 12 bước (60 phút lịch sử)
      - T_out: 12 bước (dự báo trước 5, 10, ..., 60 phút tương lai)
    """

    def __init__(
        self,
        features: np.ndarray,
        missing_mask: np.ndarray,
        history_steps: int = 12,
        horizon_steps: int = 12,
        node_dropout_prob: float = 0.0,
    ):
        """
        Args:
            features: Mảng (T_total, N, F)
            missing_mask: Mảng (T_total, N)
            history_steps: T_in (mặc định 12 = 60 phút)
            horizon_steps: T_out (mặc định 12 = 60 phút)
            node_dropout_prob: Xác suất che ngẫu nhiên các nút camera lúc train (giả lập rớt mạng)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.missing_mask = torch.tensor(missing_mask, dtype=torch.float32)
        self.T_in = history_steps
        self.T_out = horizon_steps
        self.node_dropout = node_dropout_prob

        self.num_samples = len(self.features) - self.T_in - self.T_out + 1
        assert self.num_samples > 0, "Dữ liệu quá ngắn so với T_in + T_out."

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_x = idx
        end_x = idx + self.T_in
        start_y = end_x
        end_y = end_y = start_y + self.T_out

        x_hist = self.features[start_x:end_x].clone()        # (T_in, N, F)
        m_hist = self.missing_mask[start_x:end_x].clone()    # (T_in, N)

        # Target cần dự đoán: Lấy kênh đầu tiên (mật độ/chiếm dụng) cho T_out
        y_future = self.features[start_y:end_y, :, 0].clone()     # (T_out, N)
        m_future = self.missing_mask[start_y:end_y].clone()       # (T_out, N)

        # Áp dụng Node Dropout lúc huấn luyện nếu prob > 0
        if self.node_dropout > 0:
            N = x_hist.shape[1]
            drop_mask = (torch.rand(N) >= self.node_dropout).float()  # (N,)
            x_hist = x_hist * drop_mask.view(1, N, 1)
            m_hist = m_hist * drop_mask.view(1, N)

        # Chuyển shape sang chuẩn Graph WaveNet: (F + 1, N, T_in)
        # Nối thêm mặt nạ quan sát m_hist vào kênh đặc trưng
        # x_hist: (T_in, N, F) -> permute -> (F, N, T_in)
        x_trans = x_hist.permute(2, 1, 0)
        m_trans = m_hist.unsqueeze(0)  # (1, N, T_in)
        x_in = torch.cat([x_trans, m_trans], dim=0)  # (F + 1, N, T_in)

        # y_future: (T_out, N) -> permute -> (N, T_out)
        y_target = y_future.permute(1, 0)
        m_target = m_future.permute(1, 0)

        return {
            "x_input": x_in,         # (F + 1, N, T_in)
            "y_target": y_target,    # (N, T_out)
            "m_target": m_target,    # (N, T_out)
        }


def chronological_split_data(
    features: np.ndarray,
    missing_mask: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Phân chia dữ liệu TUYỆT ĐỐI THEO TRẬT TỰ THỜI GIAN (Chronological Split):
    - 70% tuần đầu: Train
    - 10% tuần giữa: Val
    - 20% tuần cuối: Test
    Tuyệt đối không shuffle ngẫu nhiên để tránh rò rỉ thông tin tương lai.
    """
    T = len(features)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_data = (features[:train_end], missing_mask[:train_end])
    val_data = (features[train_end:val_end], missing_mask[train_end:val_end])
    test_data = (features[val_end:], missing_mask[val_end:])

    return train_data, val_data, test_data
