"""
=============================================================================
 Hướng 5: Spatio-Temporal DINO for Continuous Road Space Occupancy 
          and Congestion Level of Service (LoS) Estimation
 Module: Model (Mạng Không-Thời gian ước lượng độ chiếm dụng và cấp độ dịch vụ)
=============================================================================
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaSpatialEncoder(nn.Module):
    """Mạng tích chập gọn nhẹ trích xuất vector đặc trưng chuyển động từ bản đồ sai khác Delta."""

    def __init__(self, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=4, padding=3),   # (B, 32, H/4, W/4)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),             # (B, 64, H/8, W/8)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),        # (B, out_dim, H/16, W/16)
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatioTemporalDensityNet(nn.Module):
    """
    Mạng Không-Thời gian SpatioTemporalDensityNet:
      1. ViT Backbone (DINOv3/DINOv2): Trích xuất biểu diễn ngữ nghĩa không gian của xe cộ và cảnh quan (CLS token).
      2. DeltaSpatialEncoder: Trích xuất tín hiệu sai khác quang học vật lý chuyển động.
      3. Frame Fusion Layer: Hợp nhất tri thức không gian (RGB) và quang học chuyển động (Delta).
      4. Temporal Bi-GRU: Học động học thay đổi mật độ phương tiện theo thời gian.
      5. Multi-Task Heads:
         - occupancy_head: Ước lượng độ chiếm dụng mặt đường liên tục rho_hat in [0, 1].
         - los_head: Phân loại 4 mức độ dịch vụ ùn tắc (Free-Flow, Moderate, Slow, Gridlock).
         - trend_head: Dự đoán xu hướng ùn tắc d(rho)/dt (Gia tăng hay Giải tỏa).
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        delta_dim: int = 128,
        temporal_dim: int = 256,
        num_los_classes: int = 4,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.delta_encoder = DeltaSpatialEncoder(in_channels=1, out_dim=delta_dim)

        # Tầng hợp nhất đặc trưng đơn khung hình
        self.frame_fusion = nn.Sequential(
            nn.Linear(embed_dim + delta_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Mạng tuần hoàn thời gian 2 chiều (Bidirectional GRU)
        self.temporal_gru = nn.GRU(
            input_size=temporal_dim,
            hidden_size=temporal_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Đầu ra 1: Ước lượng tỷ lệ chiếm dụng mặt đường liên tục rho in [0, 1]
        self.occupancy_head = nn.Sequential(
            nn.Linear(temporal_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Đảm bảo rho nằm trọn trong khoảng [0, 1]
        )

        # Đầu ra 2: Phân loại cấp độ dịch vụ giao thông LoS (4 lớp)
        self.los_head = nn.Sequential(
            nn.Linear(temporal_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_los_classes),
        )

        # Đầu ra 3: Dự đoán xu hướng biến thiên d(rho)/dt in [-1, 1]
        self.trend_head = nn.Sequential(
            nn.Linear(temporal_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh(),  # Âm: Giải tỏa ùn tắc; Dương: Ùn tắc đang gia tăng
        )

    def extract_frame_features(self, rgb_seq: torch.Tensor, delta_seq: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất và hợp nhất đặc trưng không gian cho toàn bộ chuỗi K khung hình.
        Args:
            rgb_seq: (B, K, 3, H, W)
            delta_seq: (B, K, 1, H, W)
        Returns:
            fused_seq: (B, K, temporal_dim)
        """
        B, K, C, H, W = rgb_seq.shape
        rgb_flat = rgb_seq.view(B * K, C, H, W)
        delta_flat = delta_seq.view(B * K, 1, H, W)

        # 1. Trích xuất CLS token từ ViT Backbone
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
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

        # 2. Trích xuất đặc trưng chuyển động Delta
        delta_feat = self.delta_encoder(delta_flat)  # (B*K, delta_dim)

        # 3. Hợp nhất không gian
        fused = self.frame_fusion(torch.cat([cls_token, delta_feat], dim=-1))  # (B*K, temporal_dim)
        return fused.view(B, K, self.temporal_dim)

    def forward(
        self,
        rgb_seq: torch.Tensor,
        delta_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Quy trình xử lý thuận:
          - Trích xuất đặc trưng không gian đa khung hình.
          - Mô hình hóa động học thời gian qua Bi-GRU.
          - Dự đoán đồng thời tỷ lệ chiếm dụng, cấp độ phục vụ LoS và xu hướng biến thiên.
        """
        B, K, _, _, _ = rgb_seq.shape

        fused_seq = self.extract_frame_features(rgb_seq, delta_seq)  # (B, K, temporal_dim)

        # Bi-GRU thời gian
        gru_out, _ = self.temporal_gru(fused_seq)  # (B, K, temporal_dim)

        # Đặc trưng của khung hình hiện tại (khung hình cuối cùng trong cửa sổ thời gian)
        current_feat = gru_out[:, -1, :]  # (B, temporal_dim)

        # Dự đoán tỷ lệ chiếm dụng cho toàn bộ chuỗi K khung hình
        pred_occupancy_seq = self.occupancy_head(gru_out).squeeze(-1)  # (B, K)
        pred_current_occupancy = pred_occupancy_seq[:, -1]             # (B,)

        # Dự đoán phân loại LoS hiện tại
        logits_los = self.los_head(current_feat)                       # (B, 4)

        # Dự đoán xu hướng biến thiên
        pred_trend = self.trend_head(current_feat).squeeze(-1)         # (B,)

        return {
            "pred_occupancy_seq": pred_occupancy_seq,
            "pred_occupancy": pred_current_occupancy,
            "logits_los": logits_los,
            "pred_trend": pred_trend,
            "temporal_feature": current_feat,
        }


# Tương thích ngược với tên gọi cũ
TemporalTrafficEncoder = SpatioTemporalDensityNet
