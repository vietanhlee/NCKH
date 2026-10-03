"""
=============================================================================
 Hướng G: FrozenExtractor
 Module trích xuất đặc trưng DINOv3 đông cứng và giảm chiều PCA cho TAM
 Hỗ trợ chuẩn hóa L2, ma trận chiếu PCA 64 chiều, chuẩn khung hình 448x256
=============================================================================
"""

import os
import sys
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo xung đột OpenMP không xảy ra trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from common.backbone_loader import get_dino_backbone, extract_tokens


class FrozenExtractor(nn.Module):
    """
    Trích xuất đặc trưng không gian từ DINOv3 đông cứng và chiếu về không gian PCA d chiều (mặc định d=64).
    Đầu vào: Tensor ảnh (B, 3, 256, 448) [H=256, W=448].
    Đầu ra: Tensor đặc trưng patch (B, 448, d) đã chuẩn hóa L2 đơn vị.
    """
    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        pca_dim: int = 64,
        weights_path: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.pca_dim = pca_dim
        self.device = torch.device(device)
        self.model_name = model_name

        # 1. Nạp backbone DINOv3 và đóng băng hoàn toàn tham số
        self.backbone, self.embed_dim, self.patch_size = get_dino_backbone(
            model_name=model_name,
            pretrained=True,
            weights_path=weights_path,
            device=self.device,
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 2. Khởi tạo ma trận chiếu PCA (buffer không gradient)
        # Khởi tạo mặc định bằng phân rã trực giao để dùng được ngay trước khi fit PCA
        q, _ = torch.linalg.qr(torch.randn(self.embed_dim, pca_dim))
        self.register_buffer("pca_components", q)  # (embed_dim, pca_dim)
        self.register_buffer("pca_mean", torch.zeros(self.embed_dim))
        self.register_buffer("is_pca_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_pca(self, sample_tokens: torch.Tensor):
        """
        Fit ma trận PCA từ tập mẫu tokens thu thập từ các frame.
        sample_tokens: Tensor (N, embed_dim)
        """
        sample_tokens = sample_tokens.to(self.device).float()
        # Chuẩn hóa L2 trước khi PCA
        sample_tokens = F.normalize(sample_tokens, p=2, dim=-1)
        mean = sample_tokens.mean(dim=0)
        centered = sample_tokens - mean
        # SVD
        _, _, V = torch.pca_lowrank(centered, q=self.pca_dim, center=False)
        self.pca_mean.copy_(mean)
        self.pca_components.copy_(V[:, :self.pca_dim])
        self.is_pca_fitted.copy_(torch.tensor(True))
        print(f"✅ [FrozenExtractor] Đã fit thành công ma trận chiếu PCA: {self.embed_dim} -> {self.pca_dim}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất đặc trưng patch và chiếu PCA.
        x: (B, 3, H, W) - khuyến nghị H=256, W=448.
        Returns:
            u: (B, P, pca_dim) đã chuẩn hóa L2, P = (H/patch_size) * (W/patch_size).
        """
        B, C, H, W = x.shape
        # Chuẩn hóa kích thước nếu khác 256x448
        if H != 256 or W != 448:
            x = F.interpolate(x, size=(256, 448), mode="bilinear", align_corners=False)
            H, W = 256, 448

        x = x.to(self.device)
        cls_token, patch_spatial = extract_tokens(self.backbone, x, patch_size=self.patch_size)
        # patch_spatial: (B, H_patches, W_patches, embed_dim)
        B_cur, Hp, Wp, D = patch_spatial.shape
        patch_flat = patch_spatial.view(B_cur, Hp * Wp, D)  # (B, 448, embed_dim)

        # 1. Chuẩn hóa L2 trong không gian DINO
        patch_norm = F.normalize(patch_flat, p=2, dim=-1)

        # 2. Chiếu PCA
        patch_centered = patch_norm - self.pca_mean.view(1, 1, -1)
        patch_pca = torch.matmul(patch_centered, self.pca_components)  # (B, 448, pca_dim)

        # 3. Chuẩn hóa L2 trong không gian PCA
        u = F.normalize(patch_pca, p=2, dim=-1)
        return u
