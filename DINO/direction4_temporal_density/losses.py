"""
=============================================================================
 Hướng 5: Spatio-Temporal DINO for Continuous Road Space Occupancy 
          and Congestion Level of Service (LoS) Estimation
 Module: Losses (Hàm mất mát mỏ neo chiếm dụng vật lý và cấp độ dịch vụ LoS)
=============================================================================
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalDensityLoss(nn.Module):
    """
    Hàm mất mát đa mục tiêu cho bài toán ước lượng mật độ và ùn tắc giao thông:
      1. Loss Chiếm dụng Mặt đường (Occupancy Loss): Ràng buộc Smooth L1 giữa rho_hat và rho_phys
         được tính trực tiếp từ mỏ neo sai khác quang học Delta.
      2. Loss Phân loại Cấp độ Dịch vụ (LoS Cross-Entropy): Ràng buộc phân loại 4 mức độ dịch vụ HCM.
      3. Loss Xu hướng Biến thiên (Trend MSE Loss): Ràng buộc tốc độ thay đổi mật độ d(rho)/dt.
      4. Loss Trơn Thời gian (Temporal Dynamic Smoothness): Đảm bảo tính liên tục tự nhiên của dòng xe.
    """

    def __init__(
        self,
        weight_occupancy: float = 5.0,
        weight_los: float = 1.0,
        weight_trend: float = 2.0,
        weight_smooth: float = 1.0,
    ):
        super().__init__()
        self.weight_occupancy = weight_occupancy
        self.weight_los = weight_los
        self.weight_trend = weight_trend
        self.weight_smooth = weight_smooth

        self.smooth_l1 = nn.SmoothL1Loss(beta=0.05)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Tính toán hàm mất mát tổng hợp.

        Args:
            preds:
                - 'pred_occupancy_seq': (B, K)
                - 'pred_occupancy': (B,)
                - 'logits_los': (B, 4)
                - 'pred_trend': (B,)
            targets:
                - 'occupancy_seq': (B, K)
                - 'current_occupancy': (B,)
                - 'current_los': (B,)
                - 'trend': (B,)

        Returns:
            Dict chứa 'loss_total' và các thành phần loss chi tiết.
        """
        device = preds["pred_occupancy"].device

        target_occ = targets["current_occupancy"].to(device)
        target_los = targets["current_los"].to(device)
        target_trend = targets["trend"].to(device)
        target_occ_seq = targets["occupancy_seq"].to(device)

        # 1. Mất mát hồi quy độ chiếm dụng mặt đường hiện tại
        loss_occ = self.smooth_l1(preds["pred_occupancy"], target_occ)

        # 2. Mất mát phân loại cấp độ dịch vụ LoS (HCM)
        loss_los = self.ce_loss(preds["logits_los"], target_los)

        # 3. Mất mát dự đoán xu hướng biến thiên d(rho)/dt
        loss_trend = self.mse_loss(preds["pred_trend"], target_trend)

        # 4. Ràng buộc bảo toàn động học liên tục giữa các khung hình kề nhau
        pred_seq = preds["pred_occupancy_seq"]
        if pred_seq.shape[1] > 1:
            pred_diff = pred_seq[:, 1:] - pred_seq[:, :-1]
            target_diff = target_occ_seq[:, 1:] - target_occ_seq[:, :-1]
            loss_smooth = self.mse_loss(pred_diff, target_diff)
        else:
            loss_smooth = torch.tensor(0.0, device=device)

        # Tổng hợp mất mát có trọng số
        loss_total = (
            self.weight_occupancy * loss_occ
            + self.weight_los * loss_los
            + self.weight_trend * loss_trend
            + self.weight_smooth * loss_smooth
        )

        return {
            "loss_total": loss_total,
            "loss_occupancy": loss_occ,
            "loss_los": loss_los,
            "loss_trend": loss_trend,
            "loss_smooth": loss_smooth,
        }


# Tương thích ngược với tên gọi cũ
TemporalContrastiveLoss = SpatioTemporalDensityLoss
