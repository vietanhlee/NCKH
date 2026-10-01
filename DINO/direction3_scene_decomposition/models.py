"""
=============================================================================
 Hướng 3: Scene Decomposition — Model Architecture (Traffic-Decompose)
 Mạng nơ-ron phân rã cảnh giao thông thành các lớp: Nền đường (Background),
 Phương tiện (Foreground) và Mặt nạ xác suất phân bố (Alpha Mask)
=============================================================================
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Khối tích chập kép cơ bản với BatchNorm và LeakyReLU."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBranch(nn.Module):
    """
    Nhánh giải mã (Decoder Branch) phóng đại không gian đặc trưng từ bottleneck
    trở về kích thước ảnh RGB ban đầu.
    """
    def __init__(self, in_channels: int = 384, out_channels: int = 3):
        super().__init__()
        # 16x16 -> 32x32
        self.up1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)
        self.conv1 = ConvBlock(256, 256)
        # 32x32 -> 64x64
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = ConvBlock(128, 128)
        # 64x64 -> 128x128
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = ConvBlock(64, 64)
        # 128x128 -> 256x256
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = ConvBlock(32, 32)

        # Đầu ra dự đoán điểm ảnh
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Đưa về đoạn [0.0, 1.0] tương đương dải pixel chuẩn hóa
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.up1(x))
        x = self.conv2(self.up2(x))
        x = self.conv3(self.up3(x))
        x = self.conv4(self.up4(x))
        return self.out_conv(x)


class TrafficDecompositionNet(nn.Module):
    """
    Kiến trúc mạng phân rã cảnh giao thông không giám sát:
      - Shared Feature Extractor: ViT Backbone (DINOv3/v2).
      - Background Decoder Branch: Phục hồi mặt đường sạch xe (Unsupervised Inpainting).
      - Foreground Decoder Branch: Tái tạo chi tiết xe.
      - Foreground Mask Head: Dự đoán trọng số pha trộn Alpha $M_{\\text{fg}} \\in [0, 1]$.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        patch_size: int = 16,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 3 nhánh giải mã độc lập
        self.bg_decoder = DecoderBranch(in_channels=embed_dim, out_channels=3)
        self.fg_decoder = DecoderBranch(in_channels=embed_dim, out_channels=3)
        self.mask_head = DecoderBranch(in_channels=embed_dim, out_channels=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Ảnh giao thông gốc (B, 3, H, W)
        Returns:
            Dict chứa:
              - 'pred_bg': Ảnh nền tĩnh dự đoán (B, 3, H, W)
              - 'pred_fg': Ảnh phương tiện dự đoán (B, 3, H, W)
              - 'pred_mask': Mặt nạ alpha phương tiện (B, 1, H, W)
              - 'recon_origin': Ảnh tái tạo từ công thức compositing
        """
        B, C, H, W = x.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size

        # 1. Trích xuất đặc trưng patch từ backbone
        with torch.no_grad() if next(self.backbone.parameters()).requires_grad is False else torch.enable_grad():
            if hasattr(self.backbone, "get_intermediate_layers"):
                out = self.backbone.get_intermediate_layers(x, n=1)[0]
                if isinstance(out, tuple):
                    out = out[0]
                tokens = out[:, 1:] if out.shape[1] == (h_p * w_p + 1) else out
            elif hasattr(self.backbone, "forward_features"):
                feat = self.backbone.forward_features(x)
                tokens = feat.get("x_norm_patchtokens", list(feat.values())[0]) if isinstance(feat, dict) else feat[:, 1:]
            else:
                cls_t = self.backbone(x)
                tokens = cls_t.unsqueeze(1).expand(-1, h_p * w_p, -1)

            # Định hình lại không gian 2D: (B, embed_dim, h_p, w_p)
            tokens_spatial = tokens.view(B, h_p, w_p, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        # 2. Giải mã các lớp độc lập
        pred_bg = self.bg_decoder(tokens_spatial)
        pred_fg = self.fg_decoder(tokens_spatial)
        pred_mask = self.mask_head(tokens_spatial)

        # 3. Phép tổng hợp Alpha Compositing
        # \\hat{I}_{origin} = M_{fg} * \\hat{I}_{fg} + (1 - M_{fg}) * \\hat{I}_{bg}
        recon_origin = pred_mask * pred_fg + (1.0 - pred_mask) * pred_bg

        return {
            "pred_bg": pred_bg,
            "pred_fg": pred_fg,
            "pred_mask": pred_mask,
            "recon_origin": recon_origin,
        }
