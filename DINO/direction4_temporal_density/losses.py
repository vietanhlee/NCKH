"""
=============================================================================
 Hướng 4: Spatio-Temporal DINO for Continuous Road Space Occupancy 
          and Congestion Level Estimation
 Module: Losses (Hàm mất mát Đa nhiệm với Huber Smoothness và Cảnh báo Onset)
 Chuẩn Q1: Dùng phạt Huber/L1 cho tính trơn thời gian chống xóa mất sự kiện đột ngột
=============================================================================
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalDensityLoss(nn.Module):
    """
    Hàm mất mát đa mục tiêu cho bài toán ước lượng mật độ và ùn tắc giao thông:
      1. Loss Chiếm dụng Mặt đường (Occupancy Smooth L1): Giữa rho_hat và rho_proxy (tính trên road mask).
      2. Loss Phân loại Mức độ ùn tắc (LoS Cross-Entropy).
      3. Loss Xu hướng Biến thiên d(rho)/dt (Huber/Smooth L1).
      4. Loss Trơn Thời gian (Temporal Dynamic Smoothness): Dùng Huber Loss (beta=0.05) thay vì L2,
         tránh xóa nhòa các biến động đột ngột (tai nạn, tắc đường nhanh).
      5. Loss Khởi phát Kẹt xe (Onset Warning BCE Loss) cho chế độ Causal.
    """

    def __init__(
        self,
        weight_occupancy: float = 5.0,
        weight_los: float = 1.0,
        weight_trend: float = 2.0,
        weight_smooth: float = 1.0,
        weight_onset: float = 2.0,
    ):
        super().__init__()
        self.weight_occupancy = weight_occupancy
        self.weight_los = weight_los
        self.weight_trend = weight_trend
        self.weight_smooth = weight_smooth
        self.weight_onset = weight_onset

        self.smooth_l1 = nn.SmoothL1Loss(beta=0.05)
        self.huber_loss = nn.HuberLoss(delta=0.05)
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = preds["pred_occupancy"].device

        target_occ = targets["current_occupancy"].to(device)
        target_los = targets["current_los"].to(device)
        target_trend = targets["trend"].to(device)
        target_occ_seq = targets["occupancy_seq"].to(device)

        # 1. Hồi quy độ chiếm dụng mặt đường rho_proxy
        loss_occ = self.smooth_l1(preds["pred_occupancy"], target_occ)

        # 2. Phân loại mức độ dịch vụ ùn tắc (LoS)
        loss_los = self.ce_loss(preds["logits_los"], target_los)

        # 3. Dự đoán xu hướng biến thiên d(rho)/dt
        loss_trend = self.smooth_l1(preds["pred_trend"], target_trend)

        # 4. Ràng buộc bảo toàn động học liên tục giữa các khung hình kề nhau bằng Huber Loss
        pred_seq = preds["pred_occupancy_seq"]
        if pred_seq.shape[1] > 1:
            pred_diff = pred_seq[:, 1:] - pred_seq[:, :-1]
            target_diff = target_occ_seq[:, 1:] - target_occ_seq[:, :-1]
            loss_smooth = self.huber_loss(pred_diff, target_diff)
        else:
            loss_smooth = torch.tensor(0.0, device=device)

        # 5. Onset warning loss nếu có nhãn khởi phát
        loss_onset = torch.tensor(0.0, device=device)
        if "pred_onset" in preds and "onset_label" in targets:
            target_onset = targets["onset_label"].to(device).float()
            pred_onset_clamped = torch.clamp(preds["pred_onset"], min=1e-6, max=1.0 - 1e-6)
            loss_onset = self.bce_loss(pred_onset_clamped, target_onset)

        # Tổng hợp mất mát có trọng số
        loss_total = (
            self.weight_occupancy * loss_occ
            + self.weight_los * loss_los
            + self.weight_trend * loss_trend
            + self.weight_smooth * loss_smooth
            + self.weight_onset * loss_onset
        )

        return {
            "loss_total": loss_total,
            "loss_occupancy": loss_occ,
            "loss_los": loss_los,
            "loss_trend": loss_trend,
            "loss_smooth": loss_smooth,
            "loss_onset": loss_onset,
        }


# Aliases tương thích ngược
TemporalDensityLoss = SpatioTemporalDensityLoss
TemporalDensityMultiTaskLoss = SpatioTemporalDensityLoss

