"""
=============================================================================
 Hướng D: Traffic Forecasting — Spatial Graph Construction & Normalization
 Xây dựng Ma trận kề Đồ thị Mạng lưới Camera:
   1. Ma trận kề vật lý (Physical Distance Matrix) từ khoảng cách đường bộ / Haversine
   2. Ma trận kề Thích ứng Tự học (Adaptive Adjacency Matrix)
   3. Chuẩn hóa Random Walk Laplacian hai chiều (Forward / Backward Transition)
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_haversine_distance(
    coords: np.ndarray,
) -> np.ndarray:
    """
    Tính ma trận khoảng cách đại vòng tròn (Haversine Distance in km) giữa N camera.
    
    Args:
        coords: Mảng numpy (N, 2) chứa [kinh độ, vĩ độ] (hoặc ngược lại).
    Returns:
        dist_matrix: Mảng numpy (N, N) chứa khoảng cách tính bằng km.
    """
    N = coords.shape[0]
    R_earth = 6371.0  # Bán kính Trái Đất (km)

    lat = np.radians(coords[:, 1])
    lon = np.radians(coords[:, 0])

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    dist_km = R_earth * c
    return dist_km


def build_gaussian_adjacency_matrix(
    dist_matrix: np.ndarray,
    sigma: float = 2.0,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Xây dựng ma trận kề trọng số Gauss:
        W_ij = exp(- dist_ij^2 / sigma^2) nếu dist_ij <= kappa, ngược lại 0.
    """
    W = np.exp(- (dist_matrix ** 2) / (sigma ** 2))
    # Cắt tỉa các liên kết quá xa (nhiễu)
    W[W < threshold] = 0.0
    # Xóa đường chéo chính (tự nối sẽ được xử lý riêng)
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def calculate_random_walk_matrix(adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Chuẩn hóa Random Walk Transition Matrix: D^{-1} A
    """
    d = torch.sum(adj_matrix, dim=-1)
    d_inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    return d_inv.unsqueeze(-1) * adj_matrix


class AdaptiveAdjacencyLayer(nn.Module):
    """
    Ma trận kề Thích ứng Tự học (Adaptive Adjacency Matrix - Graph WaveNet style).
    Học mối tương quan giao thông ẩn giữa các nút camera không nhất thiết gần nhau về địa lý
    (ví dụ: hai nút camera trên cùng một trục xuyên tâm cửa ngõ thành phố).
    
    Công thức:
        \\tilde{A}_{adp} = \\operatorname{Softmax}\\big(\\operatorname{ReLU}(E_1 E_2^T)\\big)
    """

    def __init__(self, num_nodes: int, embed_dim: int = 16):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        # Hai ma trận embedding có thể học được
        self.source_embed = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.target_embed = nn.Parameter(torch.randn(num_nodes, embed_dim))
        
        # Khởi tạo chuẩn hóa Xavier
        nn.init.xavier_uniform_(self.source_embed)
        nn.init.xavier_uniform_(self.target_embed)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            A_adp: Tensor (N, N) ma trận kề thích ứng sau Softmax.
        """
        # (N, d) x (d, N) -> (N, N)
        similarity = torch.mm(self.source_embed, self.target_embed.t())
        adj = F.softmax(F.relu(similarity), dim=-1)
        return adj
