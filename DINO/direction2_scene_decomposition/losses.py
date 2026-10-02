"""
=============================================================================
 Hướng 2 (II.A): Scene Decomposition — Loss Functions
 Hàm mất mát Tự giám sát phân rã cảnh với Prior mềm có độ bất định (Uncertainty)
 Ràng buộc nền dùng chung nhiều ngày (L_shared) và Hàm loại trừ (L_excl) chuẩn Q1
=============================================================================
"""

import math
from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    channel: int = 3
) -> torch.Tensor:
    """
    Tính Structural Similarity (SSIM) Loss vi phân 2D (Wang et al., 2004).
    Trả về giá trị SSIM trung bình (dải [-1, 1], càng gần 1 càng giống).
    """
    # Tạo 1D Gaussian kernel
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()

    # Tạo 2D separable kernel
    kernel_2d = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    kernel = kernel_2d.repeat(channel, 1, 1, 1)

    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def total_variation_loss(m: torch.Tensor) -> torch.Tensor:
    """Làm mịn viền mặt nạ Alpha, khử đốm nhiễu."""
    diff_i = torch.abs(m[:, :, 1:, :] - m[:, :, :-1, :])
    diff_j = torch.abs(m[:, :, :, 1:] - m[:, :, :, :-1])
    return torch.mean(diff_i) + torch.mean(diff_j)


class NoiseAwareDecompositionLoss(nn.Module):
    """
    Hàm mất mát Đa mục tiêu theo công thức A.9:
      L = L_rec + lambda_p * L_prior + lambda_sh * L_shared + lambda_ex * L_excl
          + lambda_tv * L_tv + lambda_sp * L_sparse + lambda_bin * L_bin
    """

    def __init__(
        self,
        lambda_prior: float = 1.0,
        lambda_shared: float = 1.0,
        lambda_excl: float = 0.5,
        lambda_tv: float = 0.01,
        lambda_sparse: float = 0.01,
        lambda_bin: float = 0.05,
        tau_excl: float = 0.05,
        charbonnier_eps: float = 1e-3,
    ):
        super().__init__()
        self.lambda_prior = lambda_prior
        self.lambda_shared = lambda_shared
        self.lambda_excl = lambda_excl
        self.lambda_tv = lambda_tv
        self.lambda_sparse = lambda_sparse
        self.lambda_bin = lambda_bin
        self.tau_excl = tau_excl
        self.charbonnier_eps = charbonnier_eps

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        origin: Union[torch.Tensor, Dict[str, torch.Tensor]],
        prior_bg: Optional[torch.Tensor] = None,
        epoch: int = 10,
        warmup_bin_epoch: int = 10,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            preds: Dict gồm 'recon_origin', 'pred_bg', 'pred_fg', 'alpha_mask', 'sigma', 'log_sigma'.
            origin: Ảnh gốc thật (B, 3, H, W) HOẶC dict targets {'origin': ..., 'bg': ...}.
            prior_bg: Ảnh background median làm prior (B, 3, H, W).
            epoch: Epoch huấn luyện hiện tại.
        """
        if isinstance(origin, dict):
            targets = origin
            prior_bg = targets.get("bg", targets.get("prior_bg", targets.get("prior", prior_bg)))
            origin = targets.get("origin", targets.get("frame", None))

        assert origin is not None, "origin frame không được None"
        if prior_bg is None:
            # Fallback nếu không có prior_bg: dùng chính pred_bg đã detached
            prior_bg = preds["pred_bg"].detach()
        I_recon = preds["recon_origin"]
        B_hat = preds["pred_bg"]
        M_alpha = preds["alpha_mask"]
        sigma = preds["sigma"]
        log_sigma = preds["log_sigma"]

        # 1. Tái tạo (Reconstruction): 0.85 * (1 - SSIM) / 2 + 0.15 * L1
        ssim_val = ssim_loss(I_recon, origin)
        l_ssim = (1.0 - ssim_val) / 2.0
        l_l1 = F.l1_loss(I_recon, origin)
        l_rec = 0.85 * l_ssim + 0.15 * l_l1

        # 2. Prior mềm có độ bất định (Laplace NLL)
        # Điểm mấu chốt: Stop-grad từ các nhánh khác, chỉ L_prior tối ưu sigma
        l1_diff_bg = torch.abs(B_hat - prior_bg).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        l_prior = torch.mean(l1_diff_bg / (sigma + 1e-6) + log_sigma)

        # 3. Nền dùng chung nhiều ngày (L_shared với hàm Charbonnier)
        # Với batch K frames cùng camera x slot khác ngày, tính median theo trục batch
        if B_hat.shape[0] > 1:
            bg_median = torch.median(B_hat, dim=0, keepdim=True)[0].detach()  # Stop-gradient
            diff_shared = B_hat - bg_median
            l_shared = torch.mean(torch.sqrt(diff_shared ** 2 + self.charbonnier_eps ** 2))
        else:
            l_shared = torch.tensor(0.0, device=B_hat.device)

        # 4. Loại trừ (L_excl): Nền giải thích được thì không được gọi là xe
        diff_origin_bg = torch.abs(origin - B_hat.detach()).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        excl_weight = torch.exp(-diff_origin_bg / self.tau_excl)
        l_excl = torch.mean(M_alpha * excl_weight)

        # 5. Các ràng buộc hình học bổ trợ
        l_tv = total_variation_loss(M_alpha)
        l_sparse = torch.mean(M_alpha)

        # 6. Nhị phân hóa (L_bin): Đẩy M về 0 hoặc 1, kích hoạt sau epoch warmup
        if epoch >= warmup_bin_epoch:
            l_bin = torch.mean(M_alpha * (1.0 - M_alpha))
        else:
            l_bin = torch.tensor(0.0, device=B_hat.device)

        total_loss = (
            l_rec
            + self.lambda_prior * l_prior
            + self.lambda_shared * l_shared
            + self.lambda_excl * l_excl
            + self.lambda_tv * l_tv
            + self.lambda_sparse * l_sparse
            + (self.lambda_bin * l_bin if epoch >= warmup_bin_epoch else 0.0)
        )

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_rec": float(l_rec.item()),
            "loss_recon": float(l_rec.item()),  # Alias tương thích ngược
            "loss_prior": float(l_prior.item()),
            "loss_shared": float(l_shared.item()),
            "loss_excl": float(l_excl.item()),
            "loss_tv": float(l_tv.item()),
            "loss_sparse": float(l_sparse.item()),
            "mean_sigma": float(sigma.mean().item()),
            "mean_alpha": float(M_alpha.mean().item()),
        }

        return total_loss, loss_dict


# Alias tương thích ngược
DecompositionLoss = NoiseAwareDecompositionLoss
