"""
=============================================================================
 Hướng C: Anomaly Detection — Feature Extraction & Road-Aware Masking
 Trích xuất đặc trưng Patch DINOv3 và phân loại Patch Lòng đường vs Ngoại cảnh
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from common.backbone_loader import imagenet_normalize
except ImportError:
    def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
        if x.min() >= -0.05 and x.max() <= 1.05 and x.shape[1] >= 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            return (x[:, :3] - mean) / std
        return x


class DINOv3PatchFeatureExtractor(nn.Module):
    """
    Trích xuất biểu diễn patch-level từ Meta DINOv3 (đóng băng).
    Ánh xạ từ không gian d-dim (384/768) sang không gian compact 128-dim
    giúp giảm chi phí bộ nhớ và tăng tốc độ tìm kiếm láng giềng gần nhất (k-NN).
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 384,
        proj_dim: int = 128,
        patch_size: int = 16,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.patch_size = patch_size

        # Đóng băng toàn bộ trọng số backbone DINOv3
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Tầng chiếu ngẫu nhiên trực giao cố định (Fixed Random Orthogonal Projection)
        # hoặc Linear Projection để nén chiều không gian đặc trưng mà bảo toàn khoảng cách L2
        proj_matrix = torch.empty(feature_dim, proj_dim)
        nn.init.orthogonal_(proj_matrix)
        self.register_buffer("proj_matrix", proj_matrix)

    def extract_patch_tokens(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Trích xuất tokens từ ảnh:
        Args:
            rgb: Tensor (B, 3, H, W)
        Returns:
            patch_tokens: (B, N_patches, proj_dim)
            h_patches: Số patch theo chiều cao (H / patch_size)
            w_patches: Số patch theo chiều rộng (W / patch_size)
        """
        B, C, H, W = rgb.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size

        rgb_norm = imagenet_normalize(rgb)
        out = self.backbone(rgb_norm)
        if isinstance(out, dict):
            # Lấy patch tokens nếu có trong dict
            tokens = out.get("x_norm_patchtokens", None)
            if tokens is None:
                # Fallback nếu DINO trả về block tổng hợp
                tokens = list(out.values())[0]
        elif isinstance(out, torch.Tensor):
            tokens = out
        else:
            tokens = out[0]

        # Xử lý nếu tokens có token CLS ở đầu: (B, 1 + N, D) -> (B, N, D)
        if tokens.shape[1] == (h_p * w_p + 1):
            patch_feats = tokens[:, 1:, :]
        elif tokens.shape[1] == (h_p * w_p):
            patch_feats = tokens
        elif tokens.dim() == 4:
            # Dạng (B, D, h_p, w_p)
            patch_feats = tokens.flatten(2).transpose(1, 2)
        else:
            # Fallback nếu khác kích thước
            patch_feats = tokens[:, :h_p * w_p, :]

        # Chiếu xuống proj_dim và chuẩn hóa L2
        projected = torch.matmul(patch_feats, self.proj_matrix)  # (B, N, proj_dim)
        projected = F.normalize(projected, p=2, dim=-1)

        return projected, h_p, w_p

    @staticmethod
    def downsample_mask_to_patches(
        mask: torch.Tensor,
        h_patches: int,
        w_patches: int,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Hạ mẫu Binary Mask (B, 1, H, W) hoặc (H, W) về cấp độ Patch (B, N_patches).
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)

        patch_mask = F.adaptive_avg_pool2d(mask.float(), (h_patches, w_patches))
        binary_patch_mask = (patch_mask >= threshold).float()
        return binary_patch_mask.flatten(1)  # (B, N_patches)
