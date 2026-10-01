"""
=============================================================================
 Hướng 5: Temporal Contrastive Learning for Traffic Density Estimation
 Module: Model (Mô hình mã hóa tương phản chuỗi thời gian giao thông)
=============================================================================
"""

import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaSpatialEncoder(nn.Module):
    """Mạng tích chập gọn nhẹ trích xuất đặc trưng không gian từ bản đồ sai khác Delta."""

    def __init__(self, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=4, padding=3),  # (B, 32, H/4, W/4)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),            # (B, 64, H/8, W/8)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),       # (B, out_dim, H/16, W/16)
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalTrafficEncoder(nn.Module):
    """
    Mô hình mã hóa tương phản thời gian (Temporal Traffic Encoder).
    Kết hợp:
      1. ViT Backbone: Trích xuất biểu diễn ngữ nghĩa của từng khung hình [CLS] token.
      2. DeltaSpatialEncoder: Trích xuất biểu diễn chuyển động quang học từ bản đồ sai khác Delta.
      3. Temporal Attention Transformer: Tổng hợp sự biến thiên thời gian qua K khung hình.
      4. Projection Head: Ánh xạ sang không gian tương phản (128-dim) phục vụ InfoNCE.
      5. Regression Head: Dự đoán mật độ lưu lượng phương tiện.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        delta_dim: int = 128,
        temporal_dim: int = 256,
        proj_dim: int = 128,
        num_temporal_heads: int = 4,
        max_seq_len: int = 16,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Nhánh mã hóa Delta
        self.delta_encoder = DeltaSpatialEncoder(in_channels=1, out_dim=delta_dim)

        # Hợp nhất đặc trưng mỗi khung hình: DINO CLS (embed_dim) + Delta (delta_dim)
        self.frame_fusion = nn.Sequential(
            nn.Linear(embed_dim + delta_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
        )

        # Mã hóa vị trí thời gian (Temporal Positional Embedding)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, temporal_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Temporal Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=num_temporal_heads,
            dim_feedforward=temporal_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Projection Head cho Contrastive Learning (chuẩn hóa L2)
        self.proj_head = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim),
            nn.ReLU(inplace=True),
            nn.Linear(temporal_dim, proj_dim),
        )

        # Downstream Regression Head (ước lượng mật độ và chỉ số ùn tắc)
        self.density_head = nn.Sequential(
            nn.Linear(temporal_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.ReLU(),  # Mật độ >= 0
        )

    def extract_frame_features(self, rgb_seq: torch.Tensor, delta_seq: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất đặc trưng cho chuỗi K khung hình.
        Args:
            rgb_seq: (B, K, 3, H, W)
            delta_seq: (B, K, 1, H, W)
        Returns:
            fused_seq: (B, K, temporal_dim)
        """
        B, K, C, H, W = rgb_seq.shape

        # Trải phẳng chiều thời gian để đưa qua CNN/ViT một lần
        rgb_flat = rgb_seq.view(B * K, C, H, W)
        delta_flat = delta_seq.view(B * K, 1, H, W)

        # 1. Trích xuất DINO CLS
        with torch.set_grad_enabled(not next(self.backbone.parameters()).is_leaf or any(p.requires_grad for p in self.backbone.parameters())):
            if hasattr(self.backbone, "get_intermediate_layers"):
                out = self.backbone.get_intermediate_layers(rgb_flat, n=1, return_class_token=True)
                cls_token = out[0][1] if isinstance(out[0], tuple) else out[0][:, 0]
            elif hasattr(self.backbone, "forward_features"):
                feat = self.backbone.forward_features(rgb_flat)
                cls_token = feat["x_norm_clstoken"] if isinstance(feat, dict) else feat[:, 0]
            else:
                cls_token = self.backbone(rgb_flat)
                if cls_token.dim() > 2:
                    cls_token = cls_token.mean(dim=(2, 3))

        # 2. Trích xuất đặc trưng Delta
        delta_feat = self.delta_encoder(delta_flat)  # (B*K, delta_dim)

        # 3. Ghép nối và biến đổi tuyến tính
        combined = torch.cat([cls_token, delta_feat], dim=-1)  # (B*K, embed_dim + delta_dim)
        frame_feats = self.frame_fusion(combined)              # (B*K, temporal_dim)

        # Khôi phục kích thước chuỗi (B, K, temporal_dim)
        fused_seq = frame_feats.view(B, K, self.temporal_dim)
        return fused_seq

    def forward(
        self,
        rgb_seq: torch.Tensor,
        delta_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Quy trình xử lý thuận:
          - Mã hóa từng frame: ViT + Delta.
          - Biến đổi thời gian: Temporal Attention qua K khung hình.
          - Trích xuất vector biểu diễn thời gian h, vector tương phản z, và dự đoán mật độ.
        """
        B, K, _, _, _ = rgb_seq.shape

        # Trích xuất chuỗi đặc trưng các frame
        fused_seq = self.extract_frame_features(rgb_seq, delta_seq)  # (B, K, temporal_dim)

        # Thêm positional encoding thời gian
        pos = self.pos_embed[:, :K, :]
        fused_seq = fused_seq + pos

        # Tổng hợp qua Temporal Transformer
        trans_out = self.temporal_transformer(fused_seq)  # (B, K, temporal_dim)

        # Biểu diễn thời gian của toàn cửa sổ (pooling trung bình)
        h = trans_out.mean(dim=1)  # (B, temporal_dim)

        # Vector chiếu tương phản (chuẩn hóa L2)
        z = F.normalize(self.proj_head(h), dim=-1, p=2)

        # Ước lượng mật độ phương tiện
        pred_density = self.density_head(h).squeeze(-1)  # (B,)

        return {
            "temporal_feature": h,
            "proj_contrastive": z,
            "pred_density": pred_density,
        }
