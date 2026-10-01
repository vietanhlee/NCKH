"""
=============================================================================
 Hướng 7: Open-Vocabulary Traffic Scene Understanding via Delta Proposals
 Module: Model (Mô hình căn chỉnh không gian DINO sang CLIP Text Space)
=============================================================================
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenVocabTrafficDetector(nn.Module):
    """
    Mô hình nhận diện đối tượng giao thông từ vựng mở (Open-Vocabulary Traffic Detector).
    Cơ chế hoạt động:
      1. ViT Backbone: Trích xuất vector biểu diễn thị giác dày dặn từ mỗi vùng đề xuất (Proposal Crop).
      2. Alignment Projection Layer: Ánh xạ không gian đặc trưng DINO sang không gian ngữ nghĩa văn bản CLIP (512-dim).
      3. Zero-shot Similarity Matching: Tính toán xác suất phân loại thông qua độ tương đồng Cosine
         giữa vector vùng ảnh và các vector prompt văn bản tự nhiên.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        clip_dim: int = 512,
        temperature: float = 0.05,
        freeze_backbone: bool = True,
    ):
        """
        Khởi tạo OpenVocabTrafficDetector.

        Args:
            backbone: ViT backbone (DINOv3/DINOv2).
            embed_dim: Số chiều vector đặc trưng DINO CLS token.
            clip_dim: Số chiều không gian nhúng CLIP (mặc định 512).
            temperature: Nhiệt độ điều tiết độ nhọn phân phối softmax.
            freeze_backbone: Đóng băng trọng số ViT.
        """
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.clip_dim = clip_dim
        self.temperature = temperature

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Tầng căn chỉnh không gian đặc trưng (Vision-Language Alignment Projector)
        self.align_projector = nn.Sequential(
            nn.Linear(embed_dim, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.GELU(),
            nn.Linear(clip_dim, clip_dim),
        )

    def extract_visual_features(self, crops_tensor: torch.Tensor) -> torch.Tensor:
        """Trích xuất và chiếu vector đặc trưng thị giác sang không gian CLIP."""
        if hasattr(self.backbone, "get_intermediate_layers"):
            out = self.backbone.get_intermediate_layers(crops_tensor, n=1, return_class_token=True)
            cls_token = out[0][1] if isinstance(out[0], tuple) else out[0][:, 0]
        elif hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(crops_tensor)
            cls_token = feat["x_norm_clstoken"] if isinstance(feat, dict) else feat[:, 0]
        else:
            cls_token = self.backbone(crops_tensor)
            if cls_token.dim() > 2:
                cls_token = cls_token.mean(dim=(2, 3))

        # Chiếu sang không gian CLIP
        vis_proj = self.align_projector(cls_token)
        # Chuẩn hóa L2
        vis_norm = F.normalize(vis_proj, dim=-1, p=2)
        return vis_norm

    def forward(
        self,
        crops_tensor: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Dự đoán xác suất phân loại mở cho một lô vùng đề xuất.

        Args:
            crops_tensor: Tensor các vùng cắt ảnh đối tượng (B, 3, H, W).
            text_embeddings: Tensor các vector nhúng văn bản danh mục (N_classes, clip_dim).

        Returns:
            Dict chứa:
                - 'visual_embeddings': (B, clip_dim)
                - 'similarity_matrix': (B, N_classes)
                - 'probabilities': (B, N_classes)
                - 'predicted_class_ids': (B,)
        """
        vis_feats = self.extract_visual_features(crops_tensor)  # (B, clip_dim)
        text_norm = F.normalize(text_embeddings, dim=-1, p=2)   # (N_classes, clip_dim)

        # Tính độ tương đồng Cosine: S = V * T^T
        sims = torch.matmul(vis_feats, text_norm.T)             # (B, N_classes)

        # Xác suất Softmax theo nhiệt độ
        probs = F.softmax(sims / self.temperature, dim=-1)

        pred_ids = torch.argmax(probs, dim=-1)

        return {
            "visual_embeddings": vis_feats,
            "similarity_matrix": sims,
            "probabilities": probs,
            "predicted_class_ids": pred_ids,
        }
