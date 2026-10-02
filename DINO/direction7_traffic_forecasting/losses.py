"""
=============================================================================
 Hướng D: Traffic Forecasting — Multi-Task Loss Functions
 Các hàm mất mát chuyên biệt:
   1. Masked MAE Loss (bỏ qua camera mất tín hiệu)
   2. Ordinal Classification Loss cho 4 mức độ ùn tắc
   3. Focal Loss cho Onset Head (cực đoan mất cân bằng sự kiện kẹt xe)
=============================================================================
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMAELoss(nn.Module):
    """Hàm mất mát L1 MAE có mặt nạ che (Masked MAE)."""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, N, T_out)
            target: (B, N, T_out)
            mask: (B, N, T_out) nhị phân (1: có dữ liệu, 0: camera mất tín hiệu)
        """
        diff = torch.abs(pred - target) * mask
        loss = torch.sum(diff) / (torch.sum(mask) + self.eps)
        return loss


class FocalLoss(nn.Module):
    """Focal Loss cho các sự kiện thiểu số cực đoan (kẹt xe bùng phát)."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_prob: torch.Tensor, target_bin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_prob: (B, N, 3) xác suất sau sigmoid trong [0, 1]
            target_bin: (B, N, 3) nhãn nhị phân {0, 1}
        """
        p = torch.clamp(pred_prob, self.eps, 1.0 - self.eps)
        pt = torch.where(target_bin == 1, p, 1.0 - p)
        alpha_t = torch.where(target_bin == 1, self.alpha, 1.0 - self.alpha)
        loss = -alpha_t * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        return torch.mean(loss)


class MultiTaskForecastingLoss(nn.Module):
    """
    Hàm mất mát tổng hợp đa nhiệm:
        L = L_reg + lambda_cls * L_cls + lambda_onset * L_onset
    """
    def __init__(
        self,
        lambda_cls: float = 0.5,
        lambda_onset: float = 1.0,
    ):
        super().__init__()
        self.masked_mae = MaskedMAELoss()
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
        self.lambda_cls = lambda_cls
        self.lambda_onset = lambda_onset

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds:
              - "continuous_pred": (B, N, T_out)
              - "ordinal_logits": (B, N, T_out, 4)
              - "onset_prob": (B, N, 3)
            targets:
              - "y_target": (B, N, T_out)
              - "m_target": (B, N, T_out)
              - "cls_target": (B, N, T_out) nhãn lớp 0..3 (tùy chọn)
              - "onset_target": (B, N, 3) nhãn onset (tùy chọn)
        """
        y_true = targets["y_target"]
        m_true = targets["m_target"]

        # 1. Regression Loss
        loss_reg = self.masked_mae(preds["continuous_pred"], y_true, m_true)

        # 2. Classification Loss (nếu có nhãn lớp)
        loss_cls = torch.tensor(0.0, device=y_true.device)
        if "cls_target" in targets:
            logits = preds["ordinal_logits"].flatten(0, 2)  # (B*N*T, 4)
            cls_t = targets["cls_target"].flatten(0, 2)     # (B*N*T)
            mask_t = m_true.flatten(0, 2) > 0.5
            if mask_t.sum() > 0:
                loss_cls = F.cross_entropy(logits[mask_t], cls_t[mask_t])

        # 3. Onset Warning Loss (nếu có nhãn onset)
        loss_onset = torch.tensor(0.0, device=y_true.device)
        if "onset_target" in targets:
            loss_onset = self.focal_loss(preds["onset_prob"], targets["onset_target"])

        total_loss = loss_reg + self.lambda_cls * loss_cls + self.lambda_onset * loss_onset

        return {
            "loss": total_loss,
            "loss_reg": loss_reg,
            "loss_cls": loss_cls,
            "loss_onset": loss_onset,
        }
