"""
=============================================================================
 Hướng 3: Foreground-Enhanced Counting — Model Architecture
 DINOv3 Vision Transformer tích hợp 4 cơ chế tiêm Prior:
   1. 'early' : Kênh thứ 4 ở Patch Embedding (Zero-init như ControlNet)
   2. 'late'  : CNN nhỏ trích xuất đặc trưng Delta nối vào [CLS] token
   3. 'global': FiLM Modulation từ thống kê toàn cục của Delta
   4. 'none'  : Baseline thuần RGB không dùng background
 Kết hợp Delta-Dropout (30%) chống lệ thuộc vào Background Prior theo chuẩn Q1
=============================================================================
"""

import math
from typing import Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def adapt_patch_embed_to_4ch(
    patch_embed_module: nn.Module,
    init_mode: str = "zero"
) -> nn.Module:
    """
    Biến đổi tầng Patch Embedding từ 3 kênh sang 4 kênh (RGB + Delta).
    Kế thừa 100% trọng số của 3 kênh RGB gốc.
    Kênh thứ 4 được khởi tạo theo init_mode:
      - 'zero'  : Khởi tạo bằng 0 (Zero-convolution giống ControlNet) để giữ nguyên hành vi pretrained
      - 'mean'  : Khởi tạo bằng trung bình 3 kênh RGB
      - 'random': Khởi tạo ngẫu nhiên Gaussian
    """
    conv_orig = None
    if hasattr(patch_embed_module, "proj"):
        conv_orig = patch_embed_module.proj
    elif isinstance(patch_embed_module, nn.Conv2d):
        conv_orig = patch_embed_module

    if conv_orig is None or not isinstance(conv_orig, nn.Conv2d):
        return patch_embed_module

    out_c, in_c, k_h, k_w = conv_orig.weight.shape
    if in_c == 4:
        return patch_embed_module

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=out_c,
        kernel_size=(k_h, k_w),
        stride=conv_orig.stride,
        padding=conv_orig.padding,
        bias=(conv_orig.bias is not None),
    )

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = conv_orig.weight
        if init_mode == "zero":
            new_conv.weight[:, 3:4, :, :].zero_()
        elif init_mode == "mean":
            new_conv.weight[:, 3:4, :, :] = torch.mean(conv_orig.weight, dim=1, keepdim=True)
        else:  # random
            nn.init.trunc_normal_(new_conv.weight[:, 3:4, :, :], std=0.02)

        if conv_orig.bias is not None:
            new_conv.bias = conv_orig.bias

    if hasattr(patch_embed_module, "proj"):
        patch_embed_module.proj = new_conv
    else:
        patch_embed_module = new_conv

    return patch_embed_module


class DeltaCNNEncoder(nn.Module):
    """Mạng CNN gọn nhẹ phục vụ Late Fusion trích xuất vector biểu diễn từ Delta Map."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.net(delta)


class RegressionHead(nn.Module):
    """Đầu hồi quy 2 tầng MLP dự đoán số lượng xe [xe_may, o_to, tong]."""
    def __init__(self, in_features: int, hidden_dim: int = 512, out_features: int = 3, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_features),
            nn.ReLU(),  # Số lượng phương tiện luôn không âm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DINOv3FGCountingModel(nn.Module):
    """
    Mô hình đếm xe tăng cường Foreground hỗ trợ 4 chế độ hợp nhất (Fusion Strategies):
      - 'early' : RGB + Delta (4-kênh với Zero-init)
      - 'late'  : Trích xuất Delta bằng CNN nhỏ và nối vào CLS token
      - 'global': FiLM Modulation từ thống kê toàn cục
      - 'none'  : Chỉ dùng RGB (No-bg baseline chuẩn)
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        mode: str = "early",
        init_mode: str = "zero",
        delta_dropout: float = 0.30,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.delta_dropout = delta_dropout

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if self.mode in ["early", "4channel"]:
            # Biến đổi patch embedding sang 4 kênh
            if hasattr(self.backbone, "patch_embed"):
                self.backbone.patch_embed = adapt_patch_embed_to_4ch(self.backbone.patch_embed, init_mode=init_mode)
            elif hasattr(self.backbone, "conv1"):
                self.backbone.conv1 = adapt_patch_embed_to_4ch(self.backbone.conv1, init_mode=init_mode)
            self.reg_head = RegressionHead(in_features=embed_dim)

        elif self.mode == "late":
            self.delta_encoder = DeltaCNNEncoder(out_dim=128)
            self.reg_head = RegressionHead(in_features=embed_dim + 128)

        elif self.mode == "global":
            # FiLM Generator từ 4 giá trị thống kê: [mean, std, max, 90th percentile]
            self.film_gen = nn.Sequential(
                nn.Linear(4, 64),
                nn.GELU(),
                nn.Linear(64, embed_dim * 2),  # gamma và beta
            )
            # Khởi tạo gamma=1, beta=0
            with torch.no_grad():
                self.film_gen[-1].weight.zero_()
                self.film_gen[-1].bias[:embed_dim].fill_(1.0)
                self.film_gen[-1].bias[embed_dim:].zero_()
            self.reg_head = RegressionHead(in_features=embed_dim)

        elif self.mode in ["none", "no-bg"]:
            self.reg_head = RegressionHead(in_features=embed_dim)

        else:
            raise ValueError(f"Chế độ fusion không hợp lệ: {self.mode}. Chọn: early, late, global, none.")

    def forward(
        self,
        rgb: torch.Tensor,
        delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward dự đoán số lượng xe.
        Args:
            rgb: Ảnh màu chuẩn hóa (B, 3, H, W).
            delta: Bản đồ sai khác (B, 1, H, W). Tùy chọn nếu mode='none'.
        """
        B = rgb.shape[0]

        # Áp dụng Delta-Dropout lúc train (xác suất 30% đặt delta = 0)
        if delta is not None and self.training and self.delta_dropout > 0:
            if torch.rand(1).item() < self.delta_dropout:
                delta = torch.zeros_like(delta)

        if self.mode in ["early", "4channel"]:
            if rgb.shape[1] == 4:
                x_4ch = rgb
            else:
                if delta is None:
                    delta = torch.zeros(B, 1, rgb.shape[2], rgb.shape[3], device=rgb.device, dtype=rgb.dtype)
                x_4ch = torch.cat([rgb, delta], dim=1)
            feats = self.backbone(x_4ch)
            cls_token = feats.get("x_norm_clstoken", list(feats.values())[0]) if isinstance(feats, dict) else (feats[:, 0] if feats.dim() > 2 else feats)
            return self.reg_head(cls_token)

        elif self.mode == "late":
            feats = self.backbone(rgb)
            cls_token = feats.get("x_norm_clstoken", list(feats.values())[0]) if isinstance(feats, dict) else (feats[:, 0] if feats.dim() > 2 else feats)
            if delta is None:
                delta = torch.zeros(B, 1, rgb.shape[2], rgb.shape[3], device=rgb.device, dtype=rgb.dtype)
            d_feat = self.delta_encoder(delta)
            fused = torch.cat([cls_token, d_feat], dim=-1)
            return self.reg_head(fused)

        elif self.mode == "global":
            feats = self.backbone(rgb)
            cls_token = feats.get("x_norm_clstoken", list(feats.values())[0]) if isinstance(feats, dict) else (feats[:, 0] if feats.dim() > 2 else feats)
            if delta is not None:
                d_flat = delta.view(B, -1)
                mean_d = d_flat.mean(dim=1, keepdim=True)
                std_d = d_flat.std(dim=1, keepdim=True)
                max_d = d_flat.max(dim=1, keepdim=True)[0]
                p90_d = torch.quantile(d_flat, q=0.9, dim=1, keepdim=True)
                stats = torch.cat([mean_d, std_d, max_d, p90_d], dim=-1)
            else:
                stats = torch.zeros(B, 4, device=rgb.device, dtype=rgb.dtype)

            film_params = self.film_gen(stats)
            gamma = film_params[:, :self.embed_dim]
            beta = film_params[:, self.embed_dim:]
            mod_cls = gamma * cls_token + beta
            return self.reg_head(mod_cls)

        else:  # none
            feats = self.backbone(rgb)
            cls_token = feats.get("x_norm_clstoken", list(feats.values())[0]) if isinstance(feats, dict) else (feats[:, 0] if feats.dim() > 2 else feats)
            return self.reg_head(cls_token)
