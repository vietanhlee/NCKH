"""
=============================================================================
 Hướng C: Anomaly Detection — Coreset Normal Memory Bank
 Ngân hàng đặc trưng bình thường (Normal Memory Bank) phân vùng theo Camera x Khung giờ
 Thuật toán K-Center Greedy Selection (PatchCore-style) nén 90% bộ nhớ
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np


class NormalMemoryBank:
    """
    Ngân hàng vector đặc trưng chuẩn (Normal Memory Bank) cho một bối cảnh (Camera, TimeGroup).
    Lưu trữ các biểu diễn patch bình thường không có sự cố, được nén bằng thuật toán Coreset K-Center Greedy.
    """

    def __init__(
        self,
        camera_id: str,
        time_group: str,
        feature_dim: int = 128,
        device: torch.device = torch.device("cpu"),
    ):
        self.camera_id = camera_id
        self.time_group = time_group
        self.feature_dim = feature_dim
        self.device = device
        self.bank: Optional[torch.Tensor] = None  # Tensor (M, D) sau coreset

    def fit_coreset(
        self,
        features: torch.Tensor,
        subsampling_ratio: float = 0.10,
        min_samples: int = 100,
        max_samples: int = 5000,
    ) -> None:
        """
        Nén ma trận đặc trưng lớn về tập Coreset đại diện bằng K-Center Greedy.

        Args:
            features: Tensor (N, D) chứa toàn bộ các patch bình thường thu thập từ nhiều ngày.
            subsampling_ratio: Tỷ lệ chọn mẫu coreset (mặc định 10%).
            min_samples: Số lượng mẫu tối thiểu cần giữ lại.
            max_samples: Số lượng mẫu tối đa để tránh vượt RAM.
        """
        features = features.to(self.device)
        features = F.normalize(features, p=2, dim=-1)
        N, D = features.shape

        n_coreset = int(N * subsampling_ratio)
        n_coreset = max(min_samples, min(n_coreset, max_samples, N))

        if n_coreset >= N:
            self.bank = features.clone()
            return

        # Thuật toán K-Center Greedy Subsampling
        selected_indices = []
        # 1. Chọn điểm đầu tiên ngẫu nhiên
        first_idx = np.random.choice(N)
        selected_indices.append(first_idx)

        # Vector khoảng cách nhỏ nhất từ mỗi điểm đến tập đã chọn: (N,)
        first_center = features[first_idx:first_idx + 1]
        min_distances = torch.norm(features - first_center, p=2, dim=-1)

        for _ in range(1, n_coreset):
            # Chọn điểm có khoảng cách min lớn nhất (điểm xa nhất so với các center hiện tại)
            farthest_idx = int(torch.argmax(min_distances).item())
            selected_indices.append(farthest_idx)

            # Cập nhật min_distances với center mới
            new_center = features[farthest_idx:farthest_idx + 1]
            dist_to_new = torch.norm(features - new_center, p=2, dim=-1)
            min_distances = torch.minimum(min_distances, dist_to_new)

        self.bank = features[selected_indices].clone()

    def add_background_prior_patches(self, bg_patches: torch.Tensor) -> None:
        """Bổ sung thêm các patch từ ảnh nền sạch làm mẫu đối sánh phụ."""
        bg_patches = bg_patches.to(self.device)
        bg_patches = F.normalize(bg_patches, p=2, dim=-1)
        if self.bank is None:
            self.bank = bg_patches
        else:
            self.bank = torch.cat([self.bank, bg_patches], dim=0)

    def save(self, filepath: Union[str, Path]) -> None:
        """Lưu bank ra đĩa."""
        data = {
            "camera_id": self.camera_id,
            "time_group": self.time_group,
            "feature_dim": self.feature_dim,
            "bank": self.bank.cpu() if self.bank is not None else None,
        }
        torch.save(data, filepath)

    def load(self, filepath: Union[str, Path]) -> None:
        """Tải bank từ đĩa."""
        data = torch.load(filepath, map_location=self.device)
        self.camera_id = data["camera_id"]
        self.time_group = data["time_group"]
        self.feature_dim = data["feature_dim"]
        if data["bank"] is not None:
            self.bank = data["bank"].to(self.device)
