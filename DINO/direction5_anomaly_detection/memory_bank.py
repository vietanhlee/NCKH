"""
=============================================================================
 Hướng 5: Unsupervised Traffic Anomaly Detection
 Module: Memory Bank
 Quản lý và cập nhật đặc trưng bình thường (Normal Features) theo cơ chế EMA
=============================================================================
"""

import torch
import torch.nn.functional as F


class AnomalyMemoryBank:
    """
    Bộ nhớ lưu trữ các đặc trưng bình thường (Normal Features) của luồng giao thông.
    Hỗ trợ cập nhật trung bình di động (EMA - Exponential Moving Average) và
    tính toán khoảng cách k-NN để tìm điểm bất thường.
    """

    def __init__(self, feature_dim: int, bank_size: int = 1000, momentum: float = 0.99):
        """
        Khởi tạo Memory Bank.

        Args:
            feature_dim: Số chiều của vector đặc trưng.
            bank_size: Số lượng phần tử tối đa trong memory bank.
            momentum: Hệ số cập nhật EMA (0.0 đến 1.0).
        """
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.momentum = momentum
        
        # Đệm vòng (Ring buffer) lưu trữ features
        self.bank = torch.zeros((bank_size, feature_dim), dtype=torch.float32)
        self.is_full = False
        self.pointer = 0

    def update(self, features: torch.Tensor) -> None:
        """
        Cập nhật memory bank bằng các đặc trưng mới. Sử dụng EMA nếu bank đã đầy,
        hoặc thêm vào nếu chưa đầy.

        Args:
            features: Tensor chứa đặc trưng mới, kích thước (B, feature_dim).
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
            
        B = features.shape[0]
        for i in range(B):
            if not self.is_full:
                self.bank[self.pointer] = features[i]
                self.pointer += 1
                if self.pointer >= self.bank_size:
                    self.is_full = True
                    self.pointer = 0
            else:
                # Cập nhật theo EMA
                self.bank[self.pointer] = self.momentum * self.bank[self.pointer] + (1 - self.momentum) * features[i]
                self.pointer = (self.pointer + 1) % self.bank_size

    def compute_anomaly_score(self, query: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Tính toán điểm số bất thường (anomaly score) dựa trên khoảng cách k-NN.

        Args:
            query: Tensor chứa đặc trưng truy vấn, kích thước (B, feature_dim).
            k: Số lượng láng giềng gần nhất để tính trung bình khoảng cách.

        Returns:
            anomaly_scores: Tensor chứa điểm bất thường cho mỗi query, kích thước (B,).
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
            
        current_size = self.bank_size if self.is_full else self.pointer
        if current_size == 0:
            return torch.zeros(query.shape[0], dtype=torch.float32, device=query.device)

        valid_bank = self.bank[:current_size].to(query.device)
        
        # Tính khoảng cách Euclidean giữa query và bank bằng torch.cdist
        # query: (B, D), valid_bank: (N, D) -> dists: (B, N)
        dists = torch.cdist(query, valid_bank, p=2.0)
        
        # Lấy top k khoảng cách nhỏ nhất
        k_actual = min(k, current_size)
        topk_dists, _ = torch.topk(dists, k_actual, dim=1, largest=False)
        
        # Anomaly score là trung bình của top k khoảng cách
        anomaly_scores = torch.mean(topk_dists, dim=1)
        return anomaly_scores
