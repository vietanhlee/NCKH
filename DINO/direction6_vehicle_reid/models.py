"""
=============================================================================
 Hướng 6: Delta-Guided Unsupervised Vehicle Re-Identification Across Cameras
 Module: Model (Mô hình nhận dạng phương tiện với kiến trúc BNNeck)
=============================================================================
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class VehicleReIDModel(nn.Module):
    """
    Mô hình Vehicle Re-ID sử dụng Vision Foundation Backbone (DINO ViT)
    kết hợp cấu trúc cổ chai Batch Normalization Neck (BNNeck).
    Mô hình tạo ra vector đặc trưng danh tính phương tiện chuẩn hóa L2 (L2-normalized identity embedding)
    có tính bất biến cao với góc chiếu sáng và phối cảnh camera khác nhau.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        reid_dim: int = 256,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = True,
    ):
        """
        Khởi tạo VehicleReIDModel.

        Args:
            backbone: ViT backbone (DINOv3/DINOv2).
            embed_dim: Số chiều vector đặc trưng thô từ ViT CLS token.
            reid_dim: Số chiều không gian đặc trưng nhận dạng Re-ID (mặc định 256).
            dropout_rate: Tỷ lệ dropout ngăn ngừa quá khớp.
            freeze_backbone: Đóng băng trọng số ViT để chỉ tối ưu projector.
        """
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.reid_dim = reid_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Cấu trúc BNNeck chuẩn quốc tế trong bài toán Re-ID (Luo et al., CVPR Workshops 2019)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, reid_dim, bias=False),
            nn.BatchNorm1d(reid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(reid_dim, reid_dim, bias=False),
        )
        self.bnneck = nn.BatchNorm1d(reid_dim)
        self.bnneck.bias.requires_grad_(False)  # Giữ bias cố định

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất đặc trưng thô từ backbone ViT."""
        if hasattr(self.backbone, "get_intermediate_layers"):
            out = self.backbone.get_intermediate_layers(x, n=1, return_class_token=True)
            cls_token = out[0][1] if isinstance(out[0], tuple) else out[0][:, 0]
        elif hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)
            cls_token = feat["x_norm_clstoken"] if isinstance(feat, dict) else feat[:, 0]
        else:
            cls_token = self.backbone(x)
            if cls_token.dim() > 2:
                cls_token = cls_token.mean(dim=(2, 3))
        return cls_token

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Xử lý ảnh phương tiện đầu vào.

        Args:
            x: Tensor batch ảnh phương tiện (B, 3, H, W).

        Returns:
            feat_unnorm: Đặc trưng trước BNNeck phục vụ tính loss tương phản.
            feat_norm: Đặc trưng sau BNNeck đã chuẩn hóa L2 phục vụ truy vấn tìm kiếm cosine.
        """
        raw_cls = self.forward_features(x)
        feat_proj = self.projector(raw_cls)
        feat_bn = self.bnneck(feat_proj)

        # Chuẩn hóa L2 phục vụ khoảng cách Cosine
        feat_norm = F.normalize(feat_bn, dim=-1, p=2)
        return feat_proj, feat_norm

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Chế độ suy luận nhanh trích xuất vector đặc trưng chuẩn hóa L2."""
        self.eval()
        _, feat_norm = self.forward(x)
        return feat_norm
