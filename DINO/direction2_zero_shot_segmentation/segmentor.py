"""
=============================================================================
 Hướng 2: Zero-Shot Vehicle Segmentation — Lightweight Student Segmentor
 Đầu giải mã phân đoạn nhẹ (Lightweight SegHead) được huấn luyện trên Pseudo-labels
 Đạt tốc độ suy luận thời gian thực (>60 FPS) và không cần ảnh nền khi triển khai
=============================================================================
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss cho bài toán phân đoạn nhị phân đối tượng mất cân bằng."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class LightweightSegDecoder(nn.Module):
    """
    Decoder tích hợp các khối Deconvolution / Conv-Upsample nhận Patch Tokens
    từ Vision Backbone để khôi phục Mask phân đoạn sắc nét kích thước gốc.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 128,
        num_classes: int = 1,
    ):
        super().__init__()
        # Giảm chiều đặc trưng từ embedding ViT
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 3 tầng ConvTranspose2d liên tiếp để phóng đại 16x (từ 14x14 lên 224x224)
        self.upsample = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            # 56x56 -> 112x112
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            # 112x112 -> 224x224
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            # Tầng phân loại điểm ảnh cuối cùng
            nn.Conv2d(hidden_dim // 8, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: Tensor đặc trưng không gian (B, H_p, W_p, embed_dim)
        Returns:
            logits: Mặt nạ phân đoạn (B, 1, H, W)
        """
        # Chuyển đổi định dạng: (B, embed_dim, H_p, W_p)
        x = patch_tokens.permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        logits = self.upsample(x)
        return logits


class VehicleSegmentor(nn.Module):
    """
    Mô hình phân đoạn trọn gói kết hợp Frozen Backbone DINOv3 và LightweightSegDecoder.
    """

    def __init__(self, backbone: nn.Module, embed_dim: int = 384, patch_size: int = 16):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.decoder = LightweightSegDecoder(embed_dim=embed_dim)

        # Đóng băng backbone để huấn luyện cực nhanh
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size

        with torch.no_grad():
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

            tokens = tokens.view(B, h_p, w_p, -1)

        mask_logits = self.decoder(tokens)
        return mask_logits
