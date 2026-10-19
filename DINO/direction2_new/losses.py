"""
=============================================================================
 Hướng 2 (II.A): Scene Decomposition — Loss Functions
 Hàm mất mát Tự giám sát phân rã cảnh với Prior mềm có độ bất định (Uncertainty)
 Ràng buộc nền dùng chung nhiều ngày (L_shared), Khử cạnh tĩnh vạch kẻ đường,
 Căn chỉnh ánh sáng cục bộ và Hàm loại trừ (L_excl) chuẩn Q1
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models import compute_illumination_invariant_guide
except ImportError:
    from direction2_new.models import compute_illumination_invariant_guide


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
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()

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
    Hàm mất mát Đa mục tiêu chống nghiệm suy biến (Anti-Shortcut Decomposition Loss):
      1. Căn chỉnh ánh sáng cục bộ cho Prior (prior_aligned) trước khi tính sai số.
      2. Ngắt gradient kéo B_hat về phía xe cộ trong L_rec (tránh in bóng xe lên mặt đường).
      3. Hướng dẫn mặt nạ M_alpha bằng tín hiệu bất biến ánh sáng + khử cạnh tĩnh vạch kẻ đường,
         có cân bằng trọng số lớp (Positive Class Balancing) để M_alpha đạt 0.90-0.98 trên thân xe.
      4. Tách biệt hoàn toàn lớp Tiền cảnh F (triệt tiêu nền trên F) và Nền B_hat (xóa sạch xe trên B_hat).
    """

    def __init__(
        self,
        lambda_prior: float = 1.0,
        lambda_shared: float = 1.0,
        lambda_excl: float = 0.35,
        lambda_tv: float = 0.01,
        lambda_sparse: float = 0.005,
        lambda_bin: float = 0.05,
        tau_excl: float = 0.05,
        charbonnier_eps: float = 1e-3,
        lambda_bg: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        if lambda_bg is not None:
            lambda_prior = lambda_bg
        self.lambda_prior = lambda_prior
        self.lambda_bg = lambda_prior
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
        pi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(origin, dict):
            targets = origin
            prior_bg = targets.get("bg", targets.get("prior_bg", targets.get("prior", prior_bg)))
            if pi is None:
                pi = targets.get("pi", targets.get("pi_map", None))
            origin = targets.get("origin", targets.get("frame", None))

        assert origin is not None, "origin frame không được None"
        if prior_bg is None:
            prior_bg = preds["pred_bg"].detach()

        B_hat = preds["pred_bg"]
        M_alpha = torch.clamp(preds["alpha_mask"], 1e-5, 1.0 - 1e-5)
        sigma = preds["sigma"]
        log_sigma = preds["log_sigma"]

        m_guide, prior_aligned, static_edge_map = compute_illumination_invariant_guide(origin, prior_bg, pi=pi)

        # Hợp nhất bổ sung Semantic TAM Guidance pi nếu có
        if pi is not None:
            pi_4d = pi.unsqueeze(1).float() if pi.dim() == 3 else pi.float()
            if pi_4d.shape[-2:] != origin.shape[-2:]:
                pi_full = F.interpolate(pi_4d, size=origin.shape[-2:], mode="bilinear", align_corners=False)
            else:
                pi_full = pi_4d
            m_guide = torch.clamp(torch.max(m_guide, pi_full.detach()), 0.0, 1.0)

        if "pred_fg" in preds:
            bg_safe_for_rec = (1.0 - m_guide) * B_hat + m_guide * B_hat.detach()
            I_recon = M_alpha * preds["pred_fg"] + (1.0 - M_alpha) * bg_safe_for_rec
        else:
            I_recon = preds["recon_origin"]

        ssim_val = ssim_loss(I_recon, origin)
        l_ssim = (1.0 - ssim_val) / 2.0
        l_l1 = F.l1_loss(I_recon, origin)
        l_rec = 0.85 * l_ssim + 0.15 * l_l1

        l1_diff_bg = torch.abs(B_hat - prior_aligned).mean(dim=1, keepdim=True)
        sigma_min = 0.01
        log_sigma_ratio = log_sigma - math.log(sigma_min)
        laplace_nll = torch.mean(l1_diff_bg / (sigma + 1e-6) + log_sigma_ratio)
        direct_inpaint_loss = torch.mean((1.0 + 2.5 * m_guide) * l1_diff_bg) + 0.25 * (1.0 - ssim_loss(B_hat, prior_aligned))
        l_prior = 0.5 * laplace_nll + 0.5 * direct_inpaint_loss

        if B_hat.shape[0] > 1:
            bg_median = torch.median(B_hat, dim=0, keepdim=True)[0].detach()
            diff_shared = B_hat - bg_median
            l_shared = torch.mean(torch.sqrt(diff_shared ** 2 + self.charbonnier_eps ** 2))
        else:
            l_shared = torch.tensor(0.0, device=B_hat.device)

        diff_origin_prior = torch.abs(origin - prior_aligned).mean(dim=1, keepdim=True)
        true_static_zone = torch.clamp(1.0 - 1.5 * m_guide, 0.0, 1.0)
        excl_weight = true_static_zone * (torch.exp(-diff_origin_prior / self.tau_excl) + 1.5 * static_edge_map)
        l_excl = torch.mean(M_alpha * excl_weight.detach())

        bce_per_pixel = F.binary_cross_entropy(M_alpha, m_guide, reduction="none")
        pixel_balance_w = 1.0 + 3.5 * m_guide + 2.0 * static_edge_map * true_static_zone
        l_guide = torch.mean(bce_per_pixel * pixel_balance_w.detach())

        if "pred_fg" in preds:
            pred_fg = preds["pred_fg"]
            fg_active = torch.max(M_alpha.detach(), m_guide)
            diff_fg_pos = torch.abs(pred_fg - origin).mean(dim=1, keepdim=True)
            l_fg_pos = torch.mean(fg_active * diff_fg_pos) / (torch.mean(fg_active) + 1e-4)
            l_fg_bg_suppress = torch.mean((1.0 - fg_active) * pred_fg)
            l_fg = l_fg_pos + 0.15 * l_fg_bg_suppress
        else:
            l_fg = torch.tensor(0.0, device=origin.device)

        l_tv = total_variation_loss(M_alpha)
        l_sparse = torch.mean(M_alpha * true_static_zone)

        # 8. Nhị phân hóa (L_bin): Đẩy M về 0 hoặc 1 với lịch trình làm mịn (Smooth Annealed Ramp)
        ramp_epochs = 4
        if epoch >= warmup_bin_epoch:
            ramp_weight = min(1.0, float(epoch - warmup_bin_epoch + 1) / float(ramp_epochs))
            l_bin = ramp_weight * torch.mean(M_alpha * (1.0 - M_alpha))
        else:
            ramp_weight = 0.0
            l_bin = torch.tensor(0.0, device=B_hat.device)

        total_loss = (
            l_rec
            + self.lambda_prior * l_prior
            + self.lambda_shared * l_shared
            + self.lambda_excl * l_excl
            + 0.85 * l_guide
            + 0.35 * l_fg
            + self.lambda_tv * l_tv
            + self.lambda_sparse * l_sparse
            + self.lambda_bin * l_bin
        )

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_rec": float(l_rec.item()),
            "loss_recon": float(l_rec.item()),
            "loss_prior": float(l_prior.item()),
            "loss_shared": float(l_shared.item()),
            "loss_excl": float(l_excl.item()),
            "loss_guide": float(l_guide.item()),
            "loss_fg": float(l_fg.item()),
            "loss_tv": float(l_tv.item()),
            "loss_sparse": float(l_sparse.item()),
            "loss_bin": float(l_bin.item()),
            "bin_ramp_w": float(ramp_weight),
            "mean_sigma": float(sigma.mean().item()),
            "mean_alpha": float(M_alpha.mean().item()),
        }

        return total_loss, loss_dict


class SceneDecompositionLossV2(nn.Module):
    """
    Hàm mất mát Giai đoạn 2 của Hướng 2 Mới (Chưng cất sang mạng một frame):
    Học hoàn toàn từ Nhãn giả Giai đoạn 1 (B_pseudo, M_pseudo, conf, ell_gt).
    """
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_bg_pl: float = 1.0,
        lambda_mask_pl: float = 0.5,
        lambda_ell: float = 0.1,
        lambda_excl: float = 0.35,
        lambda_tv: float = 0.01,
        lambda_bin: float = 0.05,
        tau_excl: float = 0.05,
        warmup_bin_epoch: int = 10,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_bg_pl = lambda_bg_pl
        self.lambda_mask_pl = lambda_mask_pl
        self.lambda_ell = lambda_ell
        self.lambda_excl = lambda_excl
        self.lambda_tv = lambda_tv
        self.lambda_bin = lambda_bin
        self.tau_excl = tau_excl
        self.warmup_bin_epoch = warmup_bin_epoch

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        origin: torch.Tensor,
        bg_pseudo: Optional[torch.Tensor] = None,
        mask_pseudo: Optional[torch.Tensor] = None,
        conf: Optional[torch.Tensor] = None,
        ell_gt: Optional[torch.Tensor] = None,
        pi: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        I_recon = preds["recon_origin"]
        B_hat = preds["pred_bg"]
        M_alpha = torch.clamp(preds["alpha_mask"], 1e-5, 1.0 - 1e-5)
        sigma = preds["sigma"]
        log_sigma = preds["log_sigma"]
        pred_ell = preds.get("pred_ell", None)

        ssim_val = ssim_loss(I_recon, origin)
        l_ssim = (1.0 - ssim_val) / 2.0
        l_l1 = F.l1_loss(I_recon, origin)
        l_rec = 0.85 * l_ssim + 0.15 * l_l1

        if bg_pseudo is not None:
            l1_diff = torch.abs(B_hat - bg_pseudo).mean(dim=1, keepdim=True)
            sigma_min = 0.01
            log_sigma_ratio = log_sigma - math.log(sigma_min)
            l_bg_pl = 0.5 * torch.mean(l1_diff / (sigma + 1e-6) + log_sigma_ratio) + 0.5 * torch.mean(l1_diff)
        else:
            l_bg_pl = torch.tensor(0.0, device=origin.device)

        if mask_pseudo is not None:
            bce = F.binary_cross_entropy(M_alpha, mask_pseudo, reduction="none")
            balance_w = 1.0 + 3.0 * mask_pseudo.detach()
            bce = bce * balance_w
            if conf is not None:
                bce = bce * conf

            if pi is not None:
                pi_4d = pi.unsqueeze(1).float() if pi.dim() == 3 else pi.float()
                if pi_4d.shape[-2:] != M_alpha.shape[-2:]:
                    pi_up = F.interpolate(pi_4d, size=M_alpha.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    pi_up = pi_4d
                drop_mask = (M_alpha > 0.5) & (mask_pseudo < 0.2) & (pi_up > 0.5)
                bce = bce.masked_fill(drop_mask, 0.0)

            l_mask_pl = bce.mean()
        else:
            l_mask_pl = torch.tensor(0.0, device=origin.device)

        if pred_ell is not None and ell_gt is not None:
            l_ell = F.mse_loss(pred_ell, ell_gt)
        else:
            l_ell = torch.tensor(0.0, device=origin.device)

        ref_bg = bg_pseudo.detach() if bg_pseudo is not None else B_hat.detach()
        diff_bg = torch.abs(origin - ref_bg).mean(dim=1, keepdim=True)
        excl_weight = torch.exp(-diff_bg / self.tau_excl)
        if mask_pseudo is not None:
            excl_weight = excl_weight * torch.clamp(1.0 - mask_pseudo.detach(), 0.0, 1.0)
        l_excl = torch.mean(M_alpha * excl_weight)

        if "pred_fg" in preds:
            pred_fg = preds["pred_fg"]
            diff_fg = torch.abs(pred_fg - origin).mean(dim=1, keepdim=True)
            l_fg_pos = torch.mean(M_alpha * diff_fg) / (torch.mean(M_alpha).detach() + 1e-4)
            l_fg_bg = torch.mean((1.0 - M_alpha.detach()) * pred_fg)
            l_fg = l_fg_pos + 0.15 * l_fg_bg
        else:
            l_fg = torch.tensor(0.0, device=origin.device)

        l_tv = total_variation_loss(M_alpha)
        ramp_epochs = 4
        if epoch >= self.warmup_bin_epoch:
            ramp_weight = min(1.0, float(epoch - self.warmup_bin_epoch + 1) / float(ramp_epochs))
            l_bin = ramp_weight * torch.mean(M_alpha * (1.0 - M_alpha))
        else:
            ramp_weight = 0.0
            l_bin = torch.tensor(0.0, device=origin.device)

        total_loss = (
            self.lambda_rec * l_rec
            + self.lambda_bg_pl * l_bg_pl
            + self.lambda_mask_pl * l_mask_pl
            + self.lambda_ell * l_ell
            + self.lambda_excl * l_excl
            + 0.30 * l_fg
            + self.lambda_tv * l_tv
            + self.lambda_bin * l_bin
        )

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_rec": float(l_rec.item()),
            "loss_bg_pl": float(l_bg_pl.item()),
            "loss_mask_pl": float(l_mask_pl.item()),
            "loss_ell": float(l_ell.item()),
            "loss_excl": float(l_excl.item()),
            "loss_tv": float(l_tv.item()),
            "loss_bin": float(l_bin.item()),
            "bin_ramp_w": float(ramp_weight),
            "mean_sigma": float(sigma.mean().item()),
            "mean_alpha": float(M_alpha.mean().item()),
        }
        return total_loss, loss_dict


DecompositionLoss = NoiseAwareDecompositionLoss
