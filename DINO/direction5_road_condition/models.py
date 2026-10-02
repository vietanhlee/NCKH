"""
=============================================================================
 Hướng 8: Self-Supervised Road Surface Condition Estimation
 Module: Model (Mô hình phân tích đa thuộc tính tình trạng mặt đường)
=============================================================================
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoadConditionClassifier(nn.Module):
    """
    Mô hình ước lượng tình trạng mặt đường tự giám sát (Road Surface Condition Classifier).
    Tận dụng đặc trưng đại diện cấu trúc nền từ DINO ViT [CLS] token để dự đoán đồng thời:
      1. Wetness Index: Mức độ ướt/đọng nước của mặt đường [0.0 (khô ráo) -> 1.0 (ngập ướt/phản chiếu)].
      2. Illumination State: Điều kiện chiếu sáng môi trường (3 lớp: Đêm, Chạng vạng, Ngày).
      3. Road Degradation Index: Chỉ số hư hại kết cấu mặt đường (ổ gà, nứt nẻ, bề mặt bong tróc).
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        hidden_dim: int = 128,
        freeze_backbone: bool = True,
    ):
        """
        Khởi tạo RoadConditionClassifier.

        Args:
            backbone: ViT backbone (DINOv3/DINOv2).
            embed_dim: Số chiều embedding từ backbone.
            hidden_dim: Số chiều lớp ẩn cho các head phân loại.
            freeze_backbone: Đóng băng ViT backbone.
        """
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Head 1: Ước lượng mức độ ẩm ướt / đọng nước (Wetness Index in [0, 1])
        self.wetness_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Head 2: Phân loại trạng thái chiếu sáng (3 lớp: 0=Đêm, 1=Chạng vạng, 2=Ngày)
        self.illum_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

        # Head 3: Chỉ số hư hại/suy giảm bề mặt đường (Degradation Index in [0, 1])
        self.degradation_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất CLS token từ backbone."""
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Quy trình xử lý thuận.

        Args:
            x: Tensor batch ảnh nền mặt đường (B, 3, H, W).

        Returns:
            Dict chứa:
                - 'embedding': (B, embed_dim) đặc trưng mặt đường gốc.
                - 'embedding_norm': (B, embed_dim) vector đã chuẩn hóa L2 cho gom cụm.
                - 'pred_wetness': (B,) chỉ số ẩm ướt trong [0, 1].
                - 'logits_illum': (B, 3) logits chiếu sáng.
                - 'pred_degradation': (B,) chỉ số suy giảm kết cấu trong [0, 1].
        """
        cls_token = self.forward_features(x)
        cls_norm = F.normalize(cls_token, dim=-1, p=2)

        wetness = self.wetness_head(cls_token).squeeze(-1)
        logits_illum = self.illum_head(cls_token)
        degradation = self.degradation_head(cls_token).squeeze(-1)

        return {
            "embedding": cls_token,
            "embedding_norm": cls_norm,
            "pred_wetness": wetness,
            "logits_illum": logits_illum,
            "pred_degradation": degradation,
        }
