"""
=============================================================================
 Hướng 3: Scene Decomposition — Loss Functions
 Hàm mất mát Phân rã cảnh kết hợp Giám sát nền thực nghiệm (Background Supervision)
 và Ràng buộc hình học (Mask Sparsity & Total Variation Regularization)
=============================================================================
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def total_variation_loss(mask: torch.Tensor) -> torch.Tensor:
    """
    Tính toán Total Variation (TV) Loss ép mặt nạ mượt mà,
    hạn chế tối đa hiện tượng nhiễu hạt lấm tấm.
    """
    diff_i = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])
    diff_j = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
    return torch.mean(diff_i) + torch.mean(diff_j)


class DecompositionLoss(nn.Module):
    """
    Hàm mất mát tổng thể cho mô hình Traffic-Decompose:
      \\mathcal{L} = \\mathcal{L}_{recon} + \\lambda_{bg} \\mathcal{L}_{bg} + \\lambda_{sparse} \\mathcal{L}_{sparse} + \\lambda_{tv} \\mathcal{L}_{tv}
    """

    def __init__(
        self,
        lambda_bg: float = 1.5,
        lambda_sparse: float = 0.05,
        lambda_tv: float = 0.1,
    ):
        super().__init__()
        self.lambda_bg = lambda_bg
        self.lambda_sparse = lambda_sparse
        self.lambda_tv = lambda_tv

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            preds: Dict kết quả dự đoán từ TrafficDecompositionNet
            targets: Dict chứa ảnh gốc 'origin' và ảnh nền thật 'bg'
        """
        target_origin = targets["origin"]
        target_bg = targets["bg"]

        pred_origin = preds["recon_origin"]
        pred_bg = preds["pred_bg"]
        pred_mask = preds["pred_mask"]

        # 1. Sai số tái tạo ảnh gốc (Reconstruction Loss)
        l_recon = F.l1_loss(pred_origin, target_origin)

        # 2. Sai số giám sát nền tĩnh từ ảnh background thật (Background Supervision)
        l_bg = F.l1_loss(pred_bg, target_bg)

        # 3. Ràng buộc độ thưa của Mask (Sparsity Loss)
        # Ép diện tích phương tiện không được chiếm toàn bộ ảnh
        l_sparse = torch.mean(pred_mask)

        # 4. Ràng buộc làm mịn viền Mask (Total Variation Loss)
        l_tv = total_variation_loss(pred_mask)

        # Tổng hợp mất mát có trọng số
        total_loss = (
            l_recon
            + self.lambda_bg * l_bg
            + self.lambda_sparse * l_sparse
            + self.lambda_tv * l_tv
        )

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_recon": float(l_recon.item()),
            "loss_bg": float(l_bg.item()),
            "loss_sparse": float(l_sparse.item()),
            "loss_tv": float(l_tv.item()),
        }

        return total_loss, loss_dict
