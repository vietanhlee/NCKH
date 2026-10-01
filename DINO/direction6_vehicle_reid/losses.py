"""
=============================================================================
 Hướng 6: Delta-Guided Unsupervised Vehicle Re-Identification Across Cameras
 Module: Losses (Hàm mất mát học tương phản trên chuỗi Tracklet phương tiện)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackletContrastiveLoss(nn.Module):
    """
    Hàm mất mát học biểu diễn Re-ID tự giám sát từ chuỗi Tracklet.
    Nguyên lý:
      - Các góc nhìn khác nhau của cùng một xe trong chuỗi thời gian của camera
        được kéo lại gần nhau trong không gian embedding.
      - Các phương tiện khác nhau trong batch bị đẩy xa nhau với khoảng cách tối thiểu (margin).
    """

    def __init__(self, temperature: float = 0.07, margin: float = 0.3):
        """
        Khởi tạo TrackletContrastiveLoss.

        Args:
            temperature: Hệ số nhiệt độ cho hàm softmax tương phản.
            margin: Khoảng cách phân cách an toàn giữa các phương tiện khác nhau.
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        tracklet_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính toán hàm mất mát tương phản đa mẫu dựa trên Tracklet IDs.

        Args:
            embeddings: Tensor đặc trưng đã chuẩn hóa L2, shape (B, D).
            tracklet_ids: Tensor 1D chứa định danh tracklet của từng mẫu, shape (B,).

        Returns:
            loss: Giá trị mất mát scala.
        """
        device = embeddings.device
        B = embeddings.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 1. Ma trận tương đồng cosine giữa mọi cặp mẫu trong batch
        sim_matrix = torch.matmul(embeddings, embeddings.T)  # (B, B)

        # 2. Mặt nạ cặp dương và cặp âm
        # pos_mask[i, j] = True nếu cùng tracklet và i != j
        id_matrix = tracklet_ids.unsqueeze(1) == tracklet_ids.unsqueeze(0)  # (B, B)
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = id_matrix & (~diag_mask)
        neg_mask = (~id_matrix)

        if not pos_mask.any():
            # Nếu trong batch ngẫu nhiên không có 2 mẫu nào cùng ID, dùng self-supervised diagonal
            logits = sim_matrix / self.temperature
            labels = torch.arange(B, device=device)
            return F.cross_entropy(logits, labels)

        # 3. Softmax Contrastive Loss cho các mẫu có cặp dương
        losses = []
        for i in range(B):
            pos_indices = torch.where(pos_mask[i])[0]
            if len(pos_indices) == 0:
                continue

            # Mẫu dương tốt nhất (hoặc trung bình)
            pos_sim = sim_matrix[i, pos_indices]

            # Mẫu âm
            neg_indices = torch.where(neg_mask[i])[0]
            if len(neg_indices) == 0:
                continue
            neg_sim = sim_matrix[i, neg_indices]

            # InfoNCE log-sum-exp
            numerator = torch.exp(pos_sim / self.temperature).sum()
            denominator = numerator + torch.exp(neg_sim / self.temperature).sum()
            losses.append(-torch.log(numerator / (denominator + 1e-8)))

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = torch.stack(losses).mean()
        return loss
