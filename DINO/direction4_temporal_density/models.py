"""
=============================================================================
 Hướng 4: Spatio-Temporal DINO for Continuous Road Space Occupancy 
          and Congestion Level Estimation
 Module: Model (Mạng Không-Thời gian với 2 chế độ: Nowcasting & Causal Forecasting)
 Chuẩn Q1: Tách rõ Nowcasting (BiGRU) và Cảnh báo sớm khởi phát kẹt xe (Causal 1-way GRU)
=============================================================================
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from common.backbone_loader import imagenet_normalize
except ImportError:
    def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
        if x.min() >= -0.05 and x.max() <= 1.05 and x.shape[1] >= 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            return (x[:, :3] - mean) / std
        return x


class DeltaSpatialEncoder(nn.Module):
    """Mạng tích chập gọn nhẹ trích xuất vector đặc trưng chuyển động từ bản đồ sai khác Delta."""

    def __init__(self, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
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
      1. ViT Backbone (DINOv3/DINOv2): Trích xuất biểu diễn ngữ nghĩa không gian của xe cộ (CLS token).
      2. DeltaSpatialEncoder: Trích xuất tín hiệu sai khác quang học vật lý chuyển động.
      3. Frame Fusion Layer: Hợp nhất tri thức không gian (RGB) và quang học chuyển động (Delta).
      4. Temporal Recurrent Engine:
         - Nowcasting Mode: Bi-directional GRU trên chuỗi quá khứ.
         - Forecasting Mode (Causal): 1-way Causal GRU (không nhìn tương lai) để cảnh báo sớm.
      5. Multi-Task Heads:
         - occupancy_head: Ước lượng độ chiếm dụng mặt đường liên tục rho_hat in [0, 1].
         - los_head: Phân loại 4 mức độ dịch vụ ùn tắc (Free-Flow, Moderate, Slow, Gridlock).
         - trend_head: Dự đoán xu hướng ùn tắc d(rho)/dt.
         - onset_head (Causal mode): Dự đoán xác suất khởi phát kẹt xe P(Gridlock).
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        delta_dim: int = 128,
        temporal_dim: int = 256,
        num_los_classes: int = 4,
        mode: str = "nowcasting",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        self.is_causal = (self.mode in ["forecasting", "causal"])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.delta_encoder = DeltaSpatialEncoder(in_channels=1, out_dim=delta_dim)

        self.frame_fusion = nn.Sequential(
            nn.Linear(embed_dim + delta_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Mạng tuần hoàn thời gian
        if self.is_causal:
            # 1-way Causal GRU (không nhìn thấy tương lai)
            self.temporal_gru = nn.GRU(
                input_size=temporal_dim,
                hidden_size=temporal_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=False,
            )
            gru_out_dim = temporal_dim
        else:
            # BiGRU cho Nowcasting
            self.temporal_gru = nn.GRU(
                input_size=temporal_dim,
                hidden_size=temporal_dim // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            gru_out_dim = temporal_dim

        # Đầu ra 1: Ước lượng tỷ lệ chiếm dụng mặt đường liên tục rho in [0, 1]
        self.occupancy_head = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Đầu ra 2: Phân loại cấp độ dịch vụ giao thông LoS (4 lớp)
        self.los_head = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_los_classes),
        )

        # Đầu ra 3: Dự đoán xu hướng d(rho)/dt
        self.trend_head = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Đầu ra 4: Cảnh báo sớm khởi phát kẹt xe P(Onset Jam) trong tương lai h bước
        self.onset_head = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = imagenet_normalize(x)
        out = self.backbone(x_norm)
        if isinstance(out, dict):
            cls_token = out.get("x_norm_clstoken", list(out.values())[0])
        elif isinstance(out, torch.Tensor):
            cls_token = out[:, 0] if out.dim() == 3 else out
        else:
            cls_token = out[0]
        return cls_token

    def forward(
        self,
        rgb_seq: torch.Tensor,
        delta_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward chuỗi thời gian qua SpatioTemporalDensityNet.

        Args:
            rgb_seq: Tensor khung hình màu (B, K, 3, H, W).
            delta_seq: Tensor bản đồ sai khác (B, K, 1, H, W).
        """
        B, K, C, H, W = rgb_seq.shape

        # Trải phẳng không gian thời gian (B * K)
        rgb_flat = rgb_seq.view(B * K, C, H, W)
        delta_flat = delta_seq.view(B * K, 1, H, W)

        # 1. Trích xuất đặc trưng không gian
        feat_rgb = self.extract_backbone_features(rgb_flat)   # (B * K, embed_dim)
        feat_delta = self.delta_encoder(delta_flat)           # (B * K, delta_dim)

        # 2. Hợp nhất đặc trưng
        fused = self.frame_fusion(torch.cat([feat_rgb, feat_delta], dim=-1)) # (B * K, temporal_dim)
        fused_seq = fused.view(B, K, self.temporal_dim)                      # (B, K, temporal_dim)

        # 3. Mô hình hóa chuỗi thời gian
        gru_out, _ = self.temporal_gru(fused_seq)                            # (B, K, gru_out_dim)

        # 4. Dự đoán chuỗi Occupancy cho từng bước thời gian k
        pred_occ_seq = self.occupancy_head(gru_out).squeeze(-1)              # (B, K)

        # Dự đoán tại khung hình hiện tại (bước cuối cùng t = K - 1)
        last_hidden = gru_out[:, -1, :]                                      # (B, gru_out_dim)
        pred_current_occ = self.occupancy_head(last_hidden).squeeze(-1)      # (B,)
        pred_los_logits = self.los_head(last_hidden)                         # (B, num_classes)
        pred_trend = self.trend_head(last_hidden).squeeze(-1)                # (B,)
        pred_onset = self.onset_head(last_hidden).squeeze(-1)                # (B,)

        return {
            "pred_occupancy_seq": pred_occ_seq,
            "pred_occupancy": pred_current_occ,
            "logits_los": pred_los_logits,
            "pred_trend": pred_trend,
            "pred_onset": pred_onset,
        }
