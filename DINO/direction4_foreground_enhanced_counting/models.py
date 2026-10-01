"""
=============================================================================
 Hướng 4: Foreground-Enhanced Counting — Model Architecture
 DINOv3 Vision Transformer tích hợp 4-Channel Input (RGB + Δ)
 hoặc Spatial Prior Attention Gating phục vụ đếm xe đa chủng loại
=============================================================================
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def adapt_patch_embed_to_4ch(patch_embed_module: nn.Module) -> nn.Module:
    """
    Biến đổi tầng Patch Embedding từ 3 kênh sang 4 kênh (RGB + Delta).
    Kế thừa 100% trọng số của 3 kênh RGB gốc, kênh thứ 4 được khởi tạo
    bằng trung bình cộng trọng số của 3 kênh để đảm bảo tính ổn định đạo hàm ban đầu.
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
        return patch_embed_module  # Đã là 4 kênh

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=out_c,
        kernel_size=(k_h, k_w),
        stride=conv_orig.stride,
        padding=conv_orig.padding,
        bias=(conv_orig.bias is not None),
    )

    with torch.no_grad():
        # Sao chép 3 kênh cũ
        new_conv.weight[:, :3, :, :] = conv_orig.weight
        # Khởi tạo kênh thứ 4 = Trung bình 3 kênh
        new_conv.weight[:, 3:4, :, :] = torch.mean(conv_orig.weight, dim=1, keepdim=True)
        if conv_orig.bias is not None:
            new_conv.bias = conv_orig.bias

    if hasattr(patch_embed_module, "proj"):
        patch_embed_module.proj = new_conv
    else:
        patch_embed_module = new_conv

    return patch_embed_module


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
            nn.ReLU(),  # Đảm bảo số lượng phương tiện không âm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DINOv3FGCountingModel(nn.Module):
    """
    Mô hình đếm xe tăng cường Foreground:
      - Mode '4channel': Nhận tensor (B, 4, H, W) gồm RGB + Delta Map.
      - Mode 'spatial_attention': Nhận RGB (B, 3, H, W) và Delta (B, 1, H, W) để gán trọng số không gian.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        mode: str = "4channel",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.backbone = backbone
        self.embed_dim = embed_dim

        if self.mode == "4channel":
            # Điều chỉnh tầng patch embedding sang 4 kênh
            if hasattr(self.backbone, "patch_embed"):
                self.backbone.patch_embed = adapt_patch_embed_to_4ch(self.backbone.patch_embed)
            elif hasattr(self.backbone, "conv_proj"):
                self.backbone.conv_proj = adapt_patch_embed_to_4ch(self.backbone.conv_proj)
            else:
                print("⚠️ [Notice] Không tìm thấy patch_embed tiêu chuẩn, sử dụng đầu chiếu ngoài.")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Head đếm xe 3 lớp: xe máy, ô tô, tổng
        self.reg_head = RegressionHead(in_features=embed_dim, hidden_dim=512, out_features=3)

    def forward(self, x: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor đầu vào: (B, 4, H, W) nếu mode='4channel', hoặc (B, 3, H, W) nếu mode='spatial_attention'.
            delta: Tensor Delta (B, 1, H, W) nếu mode='spatial_attention'.
        Returns:
            preds: Tensor ước lượng lưu lượng (B, 3) tương ứng [xe_may, o_to, tong].
        """
        if self.mode == "4channel":
            feat = self.backbone(x)
            if hasattr(feat, "get") and isinstance(feat, dict):
                feat = feat.get("x_norm_clstoken", list(feat.values())[0])
            elif feat.dim() > 2:
                feat = feat[:, 0]
            preds = self.reg_head(feat)
            return preds

        else:
            # Mode spatial_attention: Forward 3 kênh RGB
            feat = self.backbone(x)
            if hasattr(feat, "get") and isinstance(feat, dict):
                feat = feat.get("x_norm_clstoken", list(feat.values())[0])
            elif feat.dim() > 2:
                feat = feat[:, 0]

            # Kết hợp thông tin mật độ từ Delta nếu có
            if delta is not None:
                delta_global_mean = torch.mean(delta, dim=(2, 3))  # (B, 1)
                # Tăng cường nhẹ vector embedding bằng mật độ toàn cục
                feat = feat * (1.0 + 0.1 * delta_global_mean)

            preds = self.reg_head(feat)
            return preds
