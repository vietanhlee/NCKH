"""
=============================================================================
 Hướng 2 (II.A): Phân rã cảnh giao thông tự giám sát từ Background Prior có nhiễu
 Noise-Aware Traffic Scene Decomposition with Imperfect Background Priors
 Kiến trúc 4 đầu ra: Alpha Matte (M), Tiền cảnh (F), Nền tái tạo (B_hat), và Độ bất định (sigma)
 Hỗ trợ 2 chế độ suy luận: Chế độ (a) chỉ 1 frame; Chế độ (b) frame + prior (Prior Dropout)
=============================================================================
"""

import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # 4 tầng upsample liên tục
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
          3. Inpainted Background B_hat in [0, 1] (Sigmoid)
          4. Uncertainty Map sigma in [0.01, 0.50] (kẹp log-sigma)
      - Chế độ (a): Suy luận chỉ từ 1 frame ảnh gốc.
      - Chế độ (b): Nhận thêm Prior qua nhánh CNN phụ với Prior-Dropout.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        patch_size: int = 16,
        freeze_backbone: bool = False,
        prior_dropout: float = 0.50,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.prior_dropout = prior_dropout

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Nhánh mã hóa prior (phục vụ Chế độ b)
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Conv2d(64, embed_dim, kernel_size=1),
        )

        # Decoder chính dùng chung
        self.decoder = MultiScaleDecompDecoder(in_dim=embed_dim, hidden_dim=256)

        # 4 Heads
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
        # Đầu sigma: dự đoán s = log(sigma), kẹp trong [log 0.01, log 0.5]
        self.head_log_sigma = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def extract_patch_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất tensor 2D đặc trưng (B, embed_dim, H_p, W_p) từ ViT."""
        B, C, H, W = x.shape
        H_p = H // self.patch_size
        W_p = W // self.patch_size

        feats = self.backbone(x)
        if isinstance(feats, dict):
            patch_tokens = feats.get("x_norm_patchtokens", None)
            if patch_tokens is None:
                patch_tokens = list(feats.values())[0]
        elif isinstance(feats, torch.Tensor):
            if feats.dim() == 3:
                # Bỏ CLS token nếu có
                patch_tokens = feats[:, 1:] if feats.shape[1] > H_p * W_p else feats
            elif feats.dim() == 2:
                # Fallback: mở rộng thành tensor không gian (B, embed_dim, H_p, W_p)
                return feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H_p, W_p)
            else:
                patch_tokens = feats
        else:
            patch_tokens = feats[0]

        if patch_tokens.dim() == 3:
            # Reshape về dạng không gian (B, D, H_p, W_p)
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
        use_prior: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward phân rã cảnh.
        Args:
            x: Ảnh frame gốc (B, 3, H, W).
            prior: Ảnh background median (B, 3, H, W). Tùy chọn.
            use_prior: Nếu True và có prior, kích hoạt Chế độ (b).
        """
        B, C, H, W = x.shape
        feat_map = self.extract_patch_feature_map(x)

        # Chế độ (b) Frame + Prior: hợp nhất đặc trưng prior nếu có và không bị dropout
        if prior is not None and use_prior:
            # Prior-dropout trong quá trình train để mô hình không phụ thuộc cứng vào prior
            p_keep = (1.0 - self.prior_dropout) if self.training else 1.0
            if torch.rand(1).item() < p_keep:
                prior_feat = self.prior_encoder(prior)
                if prior_feat.shape[-2:] != feat_map.shape[-2:]:
                    prior_feat = F.interpolate(prior_feat, size=feat_map.shape[-2:], mode="bilinear", align_corners=False)
                feat_map = feat_map + 0.3 * prior_feat

        # Giải mã đa tỉ lệ
        dec_feat = self.decoder(feat_map, target_hw=(H, W))

        # 4 đầu ra
        alpha_mask = self.head_alpha(dec_feat)
        pred_fg = self.head_fg(dec_feat)
        pred_bg = self.head_bg(dec_feat)

        # Tính độ bất định sigma = exp(clamp(s, log(0.01), log(0.5)))
        log_sigma = self.head_log_sigma(dec_feat)
        log_sigma_clamped = torch.clamp(log_sigma, min=math.log(0.01), max=math.log(0.50))
        sigma = torch.exp(log_sigma_clamped)

        # Tái tạo ảnh gốc vật lý: I_recon = M * F + (1 - M) * B_hat
        recon_origin = alpha_mask * pred_fg + (1.0 - alpha_mask) * pred_bg

        return {
            "alpha_mask": alpha_mask,       # M_alpha: [0, 1] (B, 1, H, W)
            "pred_mask": alpha_mask,        # Alias cho tương thích ngược
            "pred_fg": pred_fg,             # F: [0, 1] (B, 3, H, W)
            "pred_bg": pred_bg,             # B_hat: [0, 1] (B, 3, H, W)
            "sigma": sigma,                 # Độ bất định sigma: (B, 1, H, W)
            "log_sigma": log_sigma_clamped, # Phục vụ Laplace NLL
            "recon_origin": recon_origin,   # I_recon: (B, 3, H, W)
        }
