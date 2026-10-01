"""
=============================================================================
 Hướng 5: Temporal Contrastive Learning for Traffic Density Estimation
 Module: Losses (Hàm mất mát tương phản thời gian và ước lượng mật độ)
=============================================================================
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalContrastiveLoss(nn.Module):
    """
    Hàm mất mát học tương phản thời gian kết hợp giám sát mật độ (nếu có).
    Bao gồm:
      1. InfoNCE Contrastive Loss: Kéo gần các cửa sổ thời gian liền kề cùng một camera,
         đẩy xa các cửa sổ thời gian từ camera khác hoặc khung giờ khác trong mini-batch.
      2. Supervised Density Loss (Smooth L1): Tùy chọn giám sát downstream bài toán đếm/mật độ.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_density: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_density = lambda_density
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0)

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        pred_density: Optional[torch.Tensor] = None,
        target_density: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tính toán hàm mất mát.

        Args:
            z_a: Embedding tương phản của cửa sổ gốc, shape (B, D), đã chuẩn hóa L2.
            z_b: Embedding tương phản của cửa sổ liền kề (cặp dương), shape (B, D), đã chuẩn hóa L2.
            pred_density: Dự đoán mật độ từ mô hình, shape (B,).
            target_density: Nhãn mật độ thực tế, shape (B,).

        Returns:
            Dict chứa 'loss_total', 'loss_contrastive', 'loss_density'.
        """
        B = z_a.shape[0]

        # 1. Ma trận tương quan cosine giữa tất cả các mẫu trong batch
        # logits_ab[i, j] = sim(z_a[i], z_b[j]) / tau
        logits_ab = torch.matmul(z_a, z_b.T) / self.temperature
        logits_ba = torch.matmul(z_b, z_a.T) / self.temperature

        # Nhãn mục tiêu là đường chéo chính (i == j là cặp dương)
        labels = torch.arange(B, device=z_a.device)

        loss_a = F.cross_entropy(logits_ab, labels)
        loss_b = F.cross_entropy(logits_ba, labels)
        loss_contrastive = 0.5 * (loss_a + loss_b)

        # 2. Giám sát mật độ (nếu có nhãn hợp lệ >= 0)
        loss_density = torch.tensor(0.0, device=z_a.device)
        if pred_density is not None and target_density is not None:
            mask = target_density >= 0
            if mask.sum() > 0:
                loss_density = self.smooth_l1(pred_density[mask], target_density[mask])

        loss_total = loss_contrastive + self.lambda_density * loss_density

        return {
            "loss_total": loss_total,
            "loss_contrastive": loss_contrastive,
            "loss_density": loss_density,
        }
