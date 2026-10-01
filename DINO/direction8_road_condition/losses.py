"""
=============================================================================
 Hướng 8: Self-Supervised Road Surface Condition Estimation
 Module: Losses (Hàm mất mát đa mục tiêu và ràng buộc tính trơn bề mặt)
=============================================================================
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceConsistencyLoss(nn.Module):
    """
    Hàm mất mát tự giám sát tình trạng mặt đường kết hợp mỏ neo vật lý:
      1. Illumination Loss: Giám sát phân loại điều kiện chiếu sáng ngày/đêm.
      2. Physical Specular Proxy Loss: Ràng buộc chỉ số ướt wetness theo tỷ lệ phản xạ gương quang học.
      3. Roughness Texture Loss: Ràng buộc chỉ số hư hại kết cấu mặt đường theo độ biến thiên gradient.
    """

    def __init__(
        self,
        weight_illum: float = 1.0,
        weight_wetness: float = 2.0,
        weight_degradation: float = 1.0,
    ):
        super().__init__()
        self.weight_illum = weight_illum
        self.weight_wetness = weight_wetness
        self.weight_degradation = weight_degradation

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
            preds: Dict đầu ra từ RoadConditionClassifier:
                - 'logits_illum': (B, 3)
                - 'pred_wetness': (B,)
                - 'pred_degradation': (B,)
            targets: Dict chứa các chỉ số vật lý mỏ neo:
                - 'illum_class': (B,)
                - 'specular_ratio': (B,)
                - 'roughness': (B,)

        Returns:
            Dict chứa loss_total và các thành phần loss chi tiết.
        """
        device = preds["logits_illum"].device

        # 1. Loss chiếu sáng
        loss_illum = self.ce_loss(preds["logits_illum"], targets["illum_class"].to(device))

        # 2. Loss độ ẩm ướt (mỏ neo specular ratio)
        target_wetness = targets["specular_ratio"].to(device)
        # Chuẩn hóa giá trị specular_ratio về dải phù hợp [0, 1]
        target_wetness = torch.clamp(target_wetness * 20.0, 0.0, 1.0)
        loss_wetness = self.mse_loss(preds["pred_wetness"], target_wetness)

        # 3. Loss kết cấu suy giảm (mỏ neo roughness)
        target_roughness = targets["roughness"].to(device)
        target_roughness = torch.clamp(target_roughness * 5.0, 0.0, 1.0)
        loss_deg = self.mse_loss(preds["pred_degradation"], target_roughness)

        loss_total = (
            self.weight_illum * loss_illum
            + self.weight_wetness * loss_wetness
            + self.weight_degradation * loss_deg
        )

        return {
            "loss_total": loss_total,
            "loss_illum": loss_illum,
            "loss_wetness": loss_wetness,
            "loss_degradation": loss_deg,
        }
