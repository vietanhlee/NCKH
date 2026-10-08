"""
=============================================================================
 ST-BMB: Loss Functions Module
 Hệ Thống Hàm Mất Mát Ràng Buộc Vật Lý Bất Biến & Chống Suy Biến Toàn Diện
=============================================================================
Cải tiến đột phá chuẩn Production:
  1. Cross-Frame Background Consistency Loss (L_cross):
     Triệt tiêu hoàn toàn 2 nghiệm suy biến kinh điển (B=I, alpha=0) và (alpha=1, F=I).
     Nền B_hat_t ở frame t phải giải thích được frame s tại vùng frame s không có xe.
  2. Temporal Uncertainty Consistency (L_temp_bg):
     - Detach triệt để đầu ra frame trước: bg_prev.detach(), sigma_prev.detach().
     - Khóa chặn w_illum: Khóa regularizer ||g - 1||^2 + ||b||^2 + (1 - w)^2 kéo w -> 1.
     - Chroma Loss: Nâng sàn mẫu số lên >= 0.05 chống nhiễu đêm/bóng râm.
     - Đồng bộ công thức Laplace NLL với mẫu số 0.5*(sigma_t + sigma_prev).
  3. One-sided Familiarity Loss (L_fam):
     ReLU(M_fam - tau) * alpha (tau ~ 0.85): Chỉ phạt khi vùng cực kỳ quen bị đoán là xe.
  4. Edge-Aware Total Variation (Edge-TV):
     Làm mịn mặt nạ xe có định hướng theo cạnh biên ảnh gốc.
  5. Warm-up Schedule: Tăng dần trọng số Cross-frame và Binarization theo tiến trình.
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_2d_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, channel: int = 3) -> torch.Tensor:
    """Tính Structural Similarity (SSIM) Loss vi phân 2D (Wang et al., 2004)."""
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


def edge_aware_tv_loss(alpha: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """
    Edge-aware Total Variation Loss:
    Làm mịn viền mặt nạ Alpha có định hướng, bảo toàn cạnh biên tại ranh giới xe.
    """
    # Gradient ảnh gốc (B, 1, H, W-1) và (B, 1, H-1, W)
    grad_img_x = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), dim=1, keepdim=True)

    # Gradient mặt nạ alpha
    grad_alpha_x = torch.abs(alpha[:, :, :, 1:] - alpha[:, :, :, :-1])
    grad_alpha_y = torch.abs(alpha[:, :, 1:, :] - alpha[:, :, :-1, :])

    # Trọng số viền: Nơi ảnh gốc có biên độ cạnh lớn thì cho phép alpha biến thiên
    w_x = torch.exp(-10.0 * grad_img_x).detach()
    w_y = torch.exp(-10.0 * grad_img_y).detach()

    loss_x = torch.mean(grad_alpha_x * w_x)
    loss_y = torch.mean(grad_alpha_y * w_y)
    return loss_x + loss_y


def gradient_exclusion_loss(fg: torch.Tensor, bg: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Ràng buộc loại trừ gradient (Zhang et al., CVPR 2018):
    Triệt tiêu sự đồng xuất hiện của biên cạnh giữa xe cộ (hoặc mặt nạ xe) và nền mặt đường.
    """
    def grad_xy(img):
        gx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        gy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        return gx, gy

    effective_fg = (alpha * fg) if alpha is not None else fg
    fg_gx, fg_gy = grad_xy(effective_fg)
    bg_gx, bg_gy = grad_xy(bg)

    min_h = min(fg_gy.shape[2], bg_gy.shape[2])
    min_w = min(fg_gx.shape[3], bg_gx.shape[3])

    loss_x = torch.mean(torch.tanh(10.0 * fg_gx[:, :, :min_h, :min_w]) * torch.tanh(10.0 * bg_gx[:, :, :min_h, :min_w]))
    loss_y = torch.mean(torch.tanh(10.0 * fg_gy[:, :, :min_h, :min_w]) * torch.tanh(10.0 * bg_gy[:, :, :min_h, :min_w]))
    return loss_x + loss_y


class StreamingDecompositionLoss(nn.Module):
    """
    Hệ thống hàm mất mát tối ưu hóa luồng ST-BMB toàn diện:
      L_total = L_rec + lambda_temp * L_temp_bg + warmup * lambda_cross * L_cross
                + lambda_fam * L_fam + lambda_sparse * L_sparse + lambda_tv * L_edge_tv
                + lambda_excl * L_excl + warmup * lambda_bin * L_bin + lambda_illum_reg * L_illum_reg
    """

    def __init__(
        self,
        lambda_temp: float = 2.0,
        lambda_cross: float = 1.5,
        lambda_fam: float = 0.5,
        lambda_chroma: float = 0.5,
        lambda_illum_reg: float = 0.1,
        lambda_sparse: float = 0.01,
        lambda_tv: float = 0.02,
        lambda_excl: float = 0.2,
        lambda_bin: float = 0.05,
        lambda_adaptor_sup: float = 0.5,
        ssim_weight: float = 0.2,
        charbonnier_eps: float = 1e-3,
        fam_tau: float = 0.85,
    ):
        super().__init__()
        self.lambda_temp = lambda_temp
        self.lambda_cross = lambda_cross
        self.lambda_fam = lambda_fam
        self.lambda_chroma = lambda_chroma
        self.lambda_illum_reg = lambda_illum_reg
        self.lambda_sparse = lambda_sparse
        self.lambda_tv = lambda_tv
        self.lambda_excl = lambda_excl
        self.lambda_bin = lambda_bin
        self.lambda_adaptor_sup = lambda_adaptor_sup
        self.ssim_weight = ssim_weight
        self.charbonnier_eps = charbonnier_eps
        self.fam_tau = fam_tau

        # Hệ số warm-up có thể điều chỉnh qua epoch
        self.warmup_factor: float = 1.0

    def set_warmup_factor(self, factor: float):
        """Cập nhật hệ số warm-up [0.0, 1.0] cho các ràng buộc nâng cao."""
        self.warmup_factor = max(0.0, min(1.0, float(factor)))

    def charbonnier_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Hàm mất mát L1 mượt (Charbonnier) chống bão hòa gradient."""
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.charbonnier_eps ** 2))

    def cross_frame_bg_loss(
        self,
        step_outputs: List[Dict[str, torch.Tensor]],
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-Frame Background Consistency Loss:
          Nền dự đoán B_hat_t ở frame t phải giải thích được frame s tại những vùng
          mà frame s không có xe (vis_s = (1 - alpha_s).detach()):
            L_cross = 1 / (W * (W - 1)) * sum_{t != s} ( sum(vis_s * |Align(B_hat_t -> s) - I_s|) / (3 * sum(vis_s) + eps) )
        """
        W_len = len(step_outputs)
        if W_len <= 1:
            return torch.tensor(0.0, device=frames.device, requires_grad=True)

        total_cross = torch.tensor(0.0, device=frames.device)
        pair_count = 0

        for t in range(W_len):
            bg_t = step_outputs[t]["pred_bg"]  # (B, 3, H, W)

            for s in range(W_len):
                if t == s:
                    continue

                frame_s = frames[:, s]  # (B, 3, H, W)
                alpha_s = step_outputs[s]["alpha_mask"]  # (B, 1, H, W)
                vis_s = (1.0 - alpha_s).detach()         # (B, 1, H, W)

                # Căn chỉnh quang sai đơn giản giữa t và s theo vùng đường nhìn thấy được:
                # Tính gain kênh màu RGB (BẮT BUỘC DETACH để gradient hướng trực tiếp về B_hat_t tái tạo I_s)
                denom_illum = (bg_t * vis_s).sum(dim=[-2, -1], keepdim=True) + 1e-4
                numer_illum = (frame_s * vis_s).sum(dim=[-2, -1], keepdim=True) + 1e-4
                gain_t_to_s = (numer_illum / denom_illum).detach().clamp(0.05, 4.0)
                # KHÔNG clamp cứng trước khi tính diff để tránh gradient death khi bị cháy sáng (vượt 1.0)
                bg_t_aligned = bg_t * gain_t_to_s

                # L1 có trọng số theo vis_s
                diff = torch.abs(bg_t_aligned - frame_s)  # (B, 3, H, W)
                weighted_diff = vis_s * diff              # (B, 3, H, W)

                # Mẫu số: 3 * sum(vis_s) + eps
                num = weighted_diff.sum(dim=[1, 2, 3])    # (B,)
                den = 3.0 * vis_s.sum(dim=[1, 2, 3]) + 1e-4  # (B,)

                pair_loss = torch.mean(num / den)
                total_cross = total_cross + pair_loss
                pair_count += 1

        return total_cross / max(1, pair_count)

    def temporal_uncertainty_consistency(
        self,
        bg_t: torch.Tensor,
        bg_prev: torch.Tensor,
        sigma_t: torch.Tensor,
        sigma_prev: torch.Tensor,
        bg_prev_aligned: Optional[torch.Tensor] = None,
        soft_weight: Optional[torch.Tensor] = None,
        gain: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ràng buộc nền bất biến có trọng số độ bất định quang học:
          - bg_prev, sigma_prev và bg_prev_aligned BẮT BUỘC DETACH().
          - Nâng sàn mẫu số Chroma Loss lên >= 0.05 chống nhiễu đêm/bóng râm.
          - Regularizer ||g - 1||^2 + ||b||^2 + (1 - w)^2 kéo w -> 1, chống gian lận.
        """
        # Detach triệt để frame trước (bg_prev và sigma_prev)
        bg_prev_det = bg_prev.detach()
        sigma_prev_det = sigma_prev.detach()

        # ref_bg KHÔNG detach nếu là bg_prev_aligned, để gradient của sai khác quang học
        # truyền trực tiếp về gain và bias của PhotometricIlluminationAdaptor
        ref_bg = bg_prev_aligned if bg_prev_aligned is not None else bg_prev_det

        diff = torch.abs(bg_t - ref_bg)  # (B, 3, H, W)
        if soft_weight is not None:
            diff = diff * soft_weight

        # Đồng bộ mẫu số: combined_sigma = 0.5 * (sigma_t + sigma_prev_det)
        combined_sigma = 0.5 * (sigma_t + sigma_prev_det)  # (B, 1, H, W)
        sigma_min = 0.01

        # Laplace Negative Log-Likelihood
        nll = diff / (combined_sigma + 1e-4) + torch.log((combined_sigma + 1e-4) / sigma_min)

        # Ràng buộc Sắc độ Bất biến (Normalized Chromaticity Invariance):
        # Nâng sàn mẫu số lên >= 0.05 để chống phân kỳ chia cho 0 trong bóng râm / ban đêm
        denom_t = bg_t.sum(dim=1, keepdim=True).clamp(min=0.05)
        denom_prev = ref_bg.sum(dim=1, keepdim=True).clamp(min=0.05)
        chroma_t = bg_t / denom_t
        chroma_prev = ref_bg / denom_prev
        loss_chroma = torch.mean(torch.abs(chroma_t - chroma_prev))

        # Regularizer khóa chặt Photometric Adaptor (kéo g -> 1, b -> 0, w -> 1)
        loss_illum_reg = torch.tensor(0.0, device=bg_t.device)
        if gain is not None and bias is not None and soft_weight is not None:
            loss_illum_reg = (
                torch.mean((gain - 1.0) ** 2)
                + torch.mean(bias ** 2)
                + torch.mean((1.0 - soft_weight) ** 2)
            )

        temp_loss = torch.mean(nll) + self.lambda_chroma * loss_chroma
        return temp_loss, loss_chroma, loss_illum_reg

    def forward(
        self,
        step_outputs: List[Dict[str, torch.Tensor]],
        frames: torch.Tensor,
        target_jitter_gain: Optional[torch.Tensor] = None,
        target_jitter_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Tính toán toàn bộ hàm mất mát trên chuỗi W khung hình.
        Args:
            step_outputs: Danh sách kết quả dự đoán của từng frame (length W).
            frames: Tensor ảnh gốc chuỗi (B, W, 3, H, W).
            target_jitter_gain: Nhãn gain g (B,) từ jitter augmentation để giám sát trực tiếp Adaptor.
            target_jitter_bias: Nhãn bias b (B,) từ jitter augmentation để giám sát trực tiếp Adaptor.
        Returns:
            total_loss: Scalar loss phục vụ backward.
            loss_dict: Dictionary các giá trị thành phần để log.
        """
        B, W_len, C, H, W = frames.shape
        loss_rec_total = torch.tensor(0.0, device=frames.device)
        loss_temp_total = torch.tensor(0.0, device=frames.device)
        loss_fam_total = torch.tensor(0.0, device=frames.device)
        loss_sparse_total = torch.tensor(0.0, device=frames.device)
        loss_tv_total = torch.tensor(0.0, device=frames.device)
        loss_excl_total = torch.tensor(0.0, device=frames.device)
        loss_bin_total = torch.tensor(0.0, device=frames.device)
        loss_illum_reg_total = torch.tensor(0.0, device=frames.device)

        for t in range(W_len):
            out_t = step_outputs[t]
            cur_frame = frames[:, t]

            pred_bg = out_t["pred_bg"]
            pred_fg = out_t["pred_fg"]
            alpha = out_t["alpha_mask"]
            sigma = out_t["sigma"]
            recon = out_t["recon_origin"]
            bg_fam = out_t["bg_familiarity"]

            # 1. Tái tạo ảnh vật lý: Charbonnier L1 + SSIM
            l1_rec = self.charbonnier_loss(recon, cur_frame)
            ssim_val = ssim_2d_loss(recon, cur_frame)
            rec_t = (1.0 - self.ssim_weight) * l1_rec + self.ssim_weight * (1.0 - ssim_val)
            loss_rec_total = loss_rec_total + rec_t

            # 2. Ràng buộc bất biến thời gian liên frame (Temporal Uncertainty Consistency)
            if t > 0:
                prev_out = step_outputs[t - 1]
                bg_aligned = out_t.get("bg_prev_aligned", None)
                soft_w = out_t.get("illum_soft_weight", None)
                gain = out_t.get("illum_gain", None)
                bias = out_t.get("illum_bias", None)

                temp_step, chroma_step, illum_reg_step = self.temporal_uncertainty_consistency(
                    bg_t=pred_bg,
                    bg_prev=prev_out["pred_bg"],
                    sigma_t=sigma,
                    sigma_prev=prev_out["sigma"],
                    bg_prev_aligned=bg_aligned,
                    soft_weight=soft_w,
                    gain=gain,
                    bias=bias,
                )
                loss_temp_total = loss_temp_total + temp_step
                loss_illum_reg_total = loss_illum_reg_total + illum_reg_step

            # 3. Familiarity Loss Một Chiều: ReLU(bg_fam.detach() - tau) * alpha
            # BẮT BUỘC DETACH bg_fam để chỉ phạt khi vùng cực quen bị đoán là xe,
            # tuyệt đối không cho gradient làm suy biến các tham số familiarity map
            fam_conflict = torch.mean(F.relu(bg_fam.detach() - self.fam_tau) * alpha)
            loss_fam_total = loss_fam_total + fam_conflict

            # 4. Regularization: Sparsity, Edge-Aware TV, Exclusion, Binarization
            loss_sparse_total = loss_sparse_total + torch.mean(alpha)
            loss_tv_total = loss_tv_total + edge_aware_tv_loss(alpha, cur_frame)
            loss_excl_total = loss_excl_total + gradient_exclusion_loss(pred_fg, pred_bg, alpha)
            loss_bin_total = loss_bin_total + torch.mean(4.0 * alpha * (1.0 - alpha))

        # 5. Cross-Frame Background Loss: Triệt tiêu 2 nghiệm suy biến
        loss_cross = self.cross_frame_bg_loss(step_outputs, frames)

        # 6. Giám sát trực tiếp Photometric Illumination Adaptor với nhãn Jitter (nếu có)
        loss_adaptor_sup = torch.tensor(0.0, device=frames.device)
        if target_jitter_gain is not None and target_jitter_bias is not None and W_len > 1:
            pred_g = step_outputs[-1].get("illum_gain", None)
            pred_b = step_outputs[-1].get("illum_bias", None)
            if pred_g is not None and pred_b is not None:
                if target_jitter_gain.dim() == 1:
                    tg = target_jitter_gain.view(-1, 1, 1, 1).expand(-1, 3, 1, 1).to(device=frames.device, dtype=frames.dtype)
                else:
                    tg = target_jitter_gain.to(device=frames.device, dtype=frames.dtype)

                if target_jitter_bias.dim() == 1:
                    tb = target_jitter_bias.view(-1, 1, 1, 1).expand(-1, 3, 1, 1).to(device=frames.device, dtype=frames.dtype)
                else:
                    tb = target_jitter_bias.to(device=frames.device, dtype=frames.dtype)

                loss_adaptor_sup = F.mse_loss(pred_g, tg) + F.mse_loss(pred_b, tb)

        # Chuẩn hóa trung bình qua chuỗi W
        norm_rec = loss_rec_total / W_len
        norm_temp = loss_temp_total / max(1, W_len - 1)
        norm_fam = loss_fam_total / W_len
        norm_sparse = loss_sparse_total / W_len
        norm_tv = loss_tv_total / W_len
        norm_excl = loss_excl_total / W_len
        norm_bin = loss_bin_total / W_len
        norm_illum_reg = loss_illum_reg_total / max(1, W_len - 1)

        # Tổng hợp với Warm-up Schedule
        total_loss = (
            norm_rec
            + self.lambda_temp * norm_temp
            + self.warmup_factor * self.lambda_cross * loss_cross
            + self.lambda_fam * norm_fam
            + self.lambda_sparse * norm_sparse
            + self.lambda_tv * norm_tv
            + self.lambda_excl * norm_excl
            + self.warmup_factor * self.lambda_bin * norm_bin
            + self.lambda_illum_reg * norm_illum_reg
            + self.lambda_adaptor_sup * loss_adaptor_sup
        )

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_rec": float(norm_rec.item()),
            "loss_temp_bg": float(norm_temp.item()),
            "loss_cross": float(loss_cross.item()),
            "loss_fam": float(norm_fam.item()),
            "loss_sparse": float(norm_sparse.item()),
            "loss_tv": float(norm_tv.item()),
            "loss_excl": float(norm_excl.item()),
            "loss_bin": float(norm_bin.item()),
            "loss_illum_reg": float(norm_illum_reg.item()),
            "loss_adaptor_sup": float(loss_adaptor_sup.item()),
            "mean_sigma": float(torch.mean(step_outputs[-1]["sigma"]).item()),
            "mean_soft_weight": float(torch.mean(step_outputs[-1].get("illum_soft_weight", torch.tensor(1.0))).item()),
            "warmup_factor": float(self.warmup_factor),
        }

        return total_loss, loss_dict
