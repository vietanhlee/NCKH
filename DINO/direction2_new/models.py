"""
=============================================================================
 Hướng 2 (II.A): Phân rã cảnh giao thông tự giám sát từ Background Prior có nhiễu
 Noise-Aware Traffic Scene Decomposition with Imperfect Background Priors
 Kiến trúc 4 đầu ra: Alpha Matte (M), Tiền cảnh (F), Nền tái tạo (B_hat), và Độ bất định (sigma)
 Hỗ trợ 2 chế độ suy luận: Chế độ (a) chỉ 1 frame; Chế độ (b) frame + prior (Prior Dropout)
=============================================================================
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from common.backbone_loader import imagenet_normalize
except ImportError:
    def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


def align_prior_illumination(
    origin: torch.Tensor,
    prior: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    kernel_size: int = 31,
) -> torch.Tensor:
    """
    Căn chỉnh ánh sáng cục bộ (Local Photometric Gain & Bias Alignment) cho ảnh Prior:
    Khắc phục hiện tượng lệch phơi sáng / bóng nắng giữa thời điểm chụp ảnh gốc (origin)
    và thời điểm trích xuất ảnh nền (prior), chỉ ước lượng trên vùng nền tĩnh.
    """
    if mask is None:
        raw_diff = torch.abs(origin - prior).mean(dim=1, keepdim=True)
        w_stat = torch.exp(-4.0 * raw_diff).detach()
    else:
        w_stat = torch.clamp(1.0 - mask.detach(), min=0.05, max=1.0)

    pad = kernel_size // 2
    w_sum = F.avg_pool2d(w_stat, kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False) + 1e-5

    mu_orig = F.avg_pool2d(w_stat * origin, kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False) / w_sum
    mu_prior = F.avg_pool2d(w_stat * prior, kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False) / w_sum

    var_orig = F.avg_pool2d(w_stat * (origin - mu_orig).pow(2), kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False) / w_sum
    var_prior = F.avg_pool2d(w_stat * (prior - mu_prior).pow(2), kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False) / w_sum

    std_orig = torch.sqrt(torch.clamp(var_orig, min=1e-5))
    std_prior = torch.sqrt(torch.clamp(var_prior, min=1e-5))

    gain = torch.clamp((std_orig + 0.03) / (std_prior + 0.03), min=0.55, max=1.85)
    prior_aligned = mu_orig + gain * (prior - mu_prior)
    return torch.clamp(prior_aligned, 0.0, 1.0)


def _sobel_gradients(img_gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tính gradient không gian Sobel (gx, gy, mag) trên ảnh đa mức xám (B, 1, H, W)."""
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=img_gray.device, dtype=img_gray.dtype).view(1, 1, 3, 3) / 4.0
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=img_gray.device, dtype=img_gray.dtype).view(1, 1, 3, 3) / 4.0
    gx = F.conv2d(img_gray, kx, padding=1)
    gy = F.conv2d(img_gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return gx, gy, mag


def compute_illumination_invariant_guide(
    origin: torch.Tensor,
    prior: torch.Tensor,
    pi: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Trích xuất mặt nạ hướng dẫn chuyển động bất biến ánh sáng (Illumination-Invariant Motion Guide):
      1. Căn chỉnh ánh sáng cục bộ giữa prior và origin để khử lệch sáng mặt đường.
      2. Kết hợp sai khác trắc quang đa kênh và sai khác cấu trúc cục bộ (Local Structural Dissimilarity)
         để bắt trọn thân xe có màu xám/bạc trùng màu nhựa đường.
      3. Triệt tiêu cạnh tĩnh (Static Edge Suppression): Loại bỏ vạch sơn kẻ đường, mũi tên chỉ hướng,
         bờ vỉa hè và dòng timestamp vốn xuất hiện ở cả origin lẫn prior.
      4. Tích hợp Semantic TAM Guidance (pi từ ViT deep tokens nếu có): Hợp nhất đa tầng ngữ nghĩa
         giúp bắt trọn 100% xe cộ kể cả khi sơn xe trùng khít màu nhựa đường và khử bóng râm ảo.
      5. Đóng mở hình thái học khả vi (Morphological Opening & Closing) giúp lấp đầy lỗ thủng thân xe.
    """
    with torch.no_grad():
        prior_aligned = align_prior_illumination(origin, prior)
        coarse_diff = torch.abs(origin - prior_aligned).mean(dim=1, keepdim=True)
        coarse_mask = torch.clamp((coarse_diff - 0.06) / 0.12, 0.0, 1.0)
        prior_aligned = align_prior_illumination(origin, prior, mask=coarse_mask)

        abs_diff = torch.abs(origin - prior_aligned)
        diff_max = abs_diff.max(dim=1, keepdim=True)[0]
        diff_mean = abs_diff.mean(dim=1, keepdim=True)
        diff_photo = 0.55 * diff_max + 0.45 * diff_mean

        lum_orig = 0.299 * origin[:, 0:1] + 0.587 * origin[:, 1:2] + 0.114 * origin[:, 2:3]
        lum_prior = 0.299 * prior_aligned[:, 0:1] + 0.587 * prior_aligned[:, 1:2] + 0.114 * prior_aligned[:, 2:3]

        mu_o = F.avg_pool2d(lum_orig, kernel_size=7, stride=1, padding=3)
        mu_p = F.avg_pool2d(lum_prior, kernel_size=7, stride=1, padding=3)
        sigma_o = F.avg_pool2d(lum_orig * lum_orig, kernel_size=7, stride=1, padding=3) - mu_o * mu_o
        sigma_p = F.avg_pool2d(lum_prior * lum_prior, kernel_size=7, stride=1, padding=3) - mu_p * mu_p
        sigma_op = F.avg_pool2d(lum_orig * lum_prior, kernel_size=7, stride=1, padding=3) - mu_o * mu_p

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_local = ((2.0 * mu_o * mu_p + c1) * (2.0 * sigma_op + c2)) / (
            (mu_o * mu_o + mu_p * mu_p + c1) * (torch.clamp(sigma_o + sigma_p, min=0.0) + c2)
        )
        diff_struct = torch.clamp((1.0 - ssim_local) * 0.5, 0.0, 1.0)

        gx_o, gy_o, mag_o = _sobel_gradients(lum_orig)
        gx_p, gy_p, mag_p = _sobel_gradients(lum_prior)

        mag_p_dil = F.max_pool2d(mag_p, kernel_size=3, stride=1, padding=1)
        cos_sim_edge = (gx_o * gx_p + gy_o * gy_p) / (mag_o * mag_p + 1e-4)
        static_edge_map = torch.clamp(mag_p_dil * torch.clamp(cos_sim_edge, min=0.2), 0.0, 1.0)

        raw_motion = (diff_photo + 0.45 * diff_struct) / (1.0 + 4.5 * mag_p_dil)

        eroded_3 = -F.max_pool2d(-raw_motion, kernel_size=3, stride=1, padding=1)
        opened_3 = F.max_pool2d(eroded_3, kernel_size=3, stride=1, padding=1)
        dilated_7 = F.max_pool2d(opened_3, kernel_size=7, stride=1, padding=3)
        closed_7 = -F.max_pool2d(-dilated_7, kernel_size=7, stride=1, padding=3)

        m_guide = torch.clamp((closed_7 - 0.05) / 0.10, 0.0, 1.0)
        m_guide = F.avg_pool2d(m_guide, kernel_size=3, stride=1, padding=1)

        # 5. Tích hợp Semantic TAM Guidance nếu có: Hợp nhất đa mức với xác suất tiền cảnh từ ViT
        if pi is not None:
            B_g, _, H_g, W_g = m_guide.shape
            if pi.dim() == 3:
                pi_in = pi.unsqueeze(1).float()
            else:
                pi_in = pi.float()
            if pi_in.shape[-2:] != (H_g, W_g):
                pi_up = F.interpolate(pi_in, size=(H_g, W_g), mode="bilinear", align_corners=False)
            else:
                pi_up = pi_in
            m_guide = torch.clamp(torch.max(m_guide, pi_up), 0.0, 1.0)

    return m_guide.detach(), prior_aligned.detach(), static_edge_map.detach()


def inpaint_spatial_features(feat: torch.Tensor, valid_gate: torch.Tensor, num_iters: int = 3) -> torch.Tensor:
    """
    Nội suy đặc trưng vùng nền (Contextual Feature Inpainting) cho chế độ Single-Frame:
    Thay thế các đặc trưng nằm trong vùng có xe (valid_gate -> 0) bằng trung bình có trọng số
    từ các vùng mặt đường sạch lân cận qua nhiều bước lan truyền.
    """
    curr_feat = feat * valid_gate
    curr_weight = valid_gate.clamp(min=1e-4)
    for _ in range(num_iters):
        avg_f = F.avg_pool2d(curr_feat, kernel_size=7, stride=1, padding=3, count_include_pad=False)
        avg_w = F.avg_pool2d(curr_weight, kernel_size=7, stride=1, padding=3, count_include_pad=False) + 1e-5
        inpainted = avg_f / avg_w
        curr_feat = valid_gate * feat + (1.0 - valid_gate) * inpainted
        curr_weight = torch.clamp(valid_gate + (1.0 - valid_gate) * (avg_w.clamp(max=1.0)), min=1e-4, max=1.0)
    return curr_feat


class ConvBlock(nn.Module):
    """Khối tích chập kép cơ bản với GroupNorm/BatchNorm và GELU."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleDecompDecoder(nn.Module):
    """
    Decoder đa tỉ lệ kiểu DPT / Feature Pyramid phục hồi không gian sắc nét:
    Từ lưới patch token (ví dụ 14x14 hoặc 16x28) upsample dần lên 256x256 / 256x448.
    """
    def __init__(self, in_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)

        self.up1 = nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1)
        self.conv1 = ConvBlock(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = ConvBlock(64, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = ConvBlock(32, 32)

        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = ConvBlock(32, 32)

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        feat = self.proj(x)
        feat = self.conv1(self.up1(feat))
        feat = self.conv2(self.up2(feat))
        feat = self.conv3(self.up3(feat))
        feat = self.conv4(self.up4(feat))
        if feat.shape[-2:] != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        return feat


class TrafficDecompositionNet(nn.Module):
    """
    Mạng phân rã cảnh giao thông học từ Prior không hoàn hảo (II.A):
      - Encoder: ViT Backbone (DINOv3/v2) trích xuất đặc trưng toàn cục.
      - Decoder: MultiScaleDecompDecoder tái cấu trúc biểu diễn không gian.
      - 4 Heads chuyên biệt:
          1. Alpha Mask M_alpha in [0, 1] (Sigmoid)
          2. Foreground F in [0, 1] (Sigmoid)
          3. Inpainted Background B_hat in [0, 1] (Sigmoid - Chống rò rỉ xe cộ từ rgb_skip)
          4. Uncertainty Map sigma in [0.01, 0.50] (kẹp log-sigma)
      - Chế độ (a): Suy luận chỉ từ 1 frame ảnh gốc.
      - Chế độ (b): Nhận thêm Prior qua nhánh CNN phụ với Prior-Dropout & Căn chỉnh ánh sáng cục bộ.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        patch_size: int = 16,
        freeze_backbone: bool = False,
        unfreeze_last_blocks: int = 0,
        prior_dropout: float = 0.50,
        num_light_codes: int = 4,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.prior_dropout = prior_dropout
        self.num_light_codes = num_light_codes

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif unfreeze_last_blocks > 0:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if hasattr(self.backbone, "blocks"):
                blocks = self.backbone.blocks
                for b in blocks[-unfreeze_last_blocks:]:
                    for p in b.parameters():
                        p.requires_grad = True

        self.prior_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Conv2d(64, embed_dim, kernel_size=1),
        )

        self.decoder = MultiScaleDecompDecoder(in_dim=embed_dim, hidden_dim=256)

        self.rgb_skip = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
        )

        self.head_alpha = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.head_fg = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.head_bg = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.head_log_sigma = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self.head_ell = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_light_codes),
        )

    def extract_patch_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất tensor 2D đặc trưng (B, embed_dim, H_p, W_p) từ ViT có chuẩn hóa ImageNet."""
        B, C, H, W = x.shape
        H_p = H // self.patch_size
        W_p = W // self.patch_size

        x_norm = imagenet_normalize(x)
        feats = self.backbone(x_norm)
        if isinstance(feats, dict):
            patch_tokens = feats.get("x_norm_patchtokens", None)
            if patch_tokens is None:
                patch_tokens = list(feats.values())[0]
        elif isinstance(feats, torch.Tensor):
            if feats.dim() == 3:
                patch_tokens = feats[:, 1:] if feats.shape[1] > H_p * W_p else feats
            elif feats.dim() == 2:
                return feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H_p, W_p)
            else:
                patch_tokens = feats
        else:
            patch_tokens = feats[0]

        if patch_tokens.dim() == 3:
            if patch_tokens.shape[1] == H_p * W_p:
                feat_map = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, H_p, W_p)
            else:
                side = int(math.sqrt(patch_tokens.shape[1]))
                feat_map = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, side, side)
        else:
            feat_map = patch_tokens

        return feat_map

    def forward(
        self,
        x: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
        use_prior: bool = True,
        pi: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        feat_map = self.extract_patch_feature_map(x)  # (B, embed_dim, Hp, Wp)

        has_active_prior = False
        m_guide = None
        prior_aligned = None
        static_edge_map = None

        if prior is not None and use_prior:
            m_guide, prior_aligned, static_edge_map = compute_illumination_invariant_guide(x, prior, pi=pi)
            p_keep = (1.0 - self.prior_dropout) if self.training else 1.0
            if torch.rand(1).item() < p_keep:
                has_active_prior = True
                prior_feat = self.prior_encoder(x - prior_aligned)
                if prior_feat.shape[-2:] != feat_map.shape[-2:]:
                    prior_feat = F.interpolate(prior_feat, size=feat_map.shape[-2:], mode="bilinear", align_corners=False)
                feat_map = feat_map + 0.35 * prior_feat

        B_cur, D_cur, Hp, Wp = feat_map.shape
        patch_tokens = feat_map.flatten(2).transpose(1, 2)  # (B, P, D)
        pi_full = None
        if pi is not None:
            if pi.dim() == 3:
                pi_in = pi.unsqueeze(1).float()
            else:
                pi_in = pi.float()
            if pi_in.shape[-2:] != (Hp, Wp):
                pi_down = F.interpolate(pi_in, size=(Hp, Wp), mode="bilinear", align_corners=False)
            else:
                pi_down = pi_in
            w_static = (1.0 - pi_down).view(B_cur, -1, 1).clamp(min=1e-3)
            static_token_avg = (patch_tokens * w_static).sum(dim=1) / w_static.sum(dim=1).clamp(min=1e-4)

            # Bản đồ pi kích thước đầy đủ (H, W) để bảo vệ lá chắn nền vehicle_shield
            if pi_in.shape[-2:] != (H, W):
                pi_full = F.interpolate(pi_in, size=(H, W), mode="bilinear", align_corners=False)
            else:
                pi_full = pi_in
        else:
            static_token_avg = patch_tokens.mean(dim=1)  # (B, D)

        pred_ell = self.head_ell(static_token_avg)  # (B, num_light_codes)

        dec_feat = self.decoder(feat_map, target_hw=(H, W))
        rgb_feat_x = self.rgb_skip(x)

        # 1. Nhánh Tiền cảnh & Mặt nạ Alpha
        dec_feat_fg = dec_feat + rgb_feat_x
        alpha_raw = self.head_alpha(dec_feat_fg)

        if prior is not None and use_prior and not self.training and m_guide is not None:
            lane_suppress = torch.clamp(1.0 - 2.5 * static_edge_map * (1.0 - m_guide), 0.0, 1.0)
            alpha_clean = alpha_raw * lane_suppress
            alpha_mask = torch.max(alpha_clean, 0.90 * m_guide)
        else:
            alpha_mask = alpha_raw

        pred_fg = self.head_fg(dec_feat_fg)
        isolated_fg = alpha_mask * pred_fg

        # 2. Nhánh Nền sạch (Clean Road Background): NGẮT RÒ RỈ XE CỘ TỪ RGB_SKIP(X)
        shield_components = [alpha_mask.detach()]
        if m_guide is not None:
            shield_components.append(m_guide.detach())
        if pi_full is not None:
            shield_components.append(pi_full.detach())

        vehicle_shield = shield_components[0]
        for sc in shield_components[1:]:
            vehicle_shield = torch.max(vehicle_shield, sc)

        shield_dilated = F.max_pool2d(vehicle_shield, kernel_size=7, stride=1, padding=3)
        bg_vis_gate = torch.clamp(1.0 - shield_dilated * 1.4, 0.0, 1.0)

        if has_active_prior and prior_aligned is not None:
            rgb_feat_prior = self.rgb_skip(prior_aligned)
            dec_feat_bg = dec_feat + bg_vis_gate * rgb_feat_x + (1.0 - bg_vis_gate) * rgb_feat_prior
            pred_bg_raw = self.head_bg(dec_feat_bg)
            inpaint_w = torch.clamp(shield_dilated * 1.25, 0.0, 1.0)
            pred_bg = (1.0 - inpaint_w) * pred_bg_raw + inpaint_w * (0.80 * prior_aligned + 0.20 * pred_bg_raw)
        else:
            dec_inpainted = inpaint_spatial_features(dec_feat, bg_vis_gate, num_iters=2)
            skip_inpainted = inpaint_spatial_features(rgb_feat_x, bg_vis_gate, num_iters=3)
            dec_feat_bg = dec_inpainted + skip_inpainted
            pred_bg = self.head_bg(dec_feat_bg)

        # 3. Tính độ bất định Laplace sigma trơn tru C^\infty chống triệt tiêu gradient
        # sigma = sigma_min + (sigma_max - sigma_min) * sigmoid(raw_s) in (0.01, 0.50)
        raw_sigma = self.head_log_sigma(dec_feat_fg)
        sigma = 0.01 + 0.49 * torch.sigmoid(raw_sigma)
        log_sigma = torch.log(sigma)

        # 4. Tái tạo ảnh gốc vật lý: I_recon = M * F + (1 - M) * B_hat
        recon_origin = alpha_mask * pred_fg + (1.0 - alpha_mask) * pred_bg

        return {
            "alpha_mask": alpha_mask,
            "pred_mask": alpha_mask,
            "pred_fg": pred_fg,
            "isolated_fg": isolated_fg,
            "pred_bg": pred_bg,
            "sigma": sigma,
            "log_sigma": log_sigma,
            "pred_ell": pred_ell,
            "recon_origin": recon_origin,
        }
