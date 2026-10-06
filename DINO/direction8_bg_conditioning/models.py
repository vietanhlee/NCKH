"""
=============================================================================
 Hướng 8: Background Conditioning — Model Architecture
 Mô hình Thích ứng Camera Mới có Điều kiện hóa Thống kê Toàn cục Background
 Hỗ trợ 3 cơ chế (FiLM / Prompt / Cross-Attn) và Kỹ thuật Huấn luyện Bền vững (Bg-Dropout)
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from direction8_bg_conditioning.conditioning import (
    FiLMConditioningLayer,
    PromptTokenConditioningLayer,
    CrossAttentionConditioningLayer,
)

try:
    from common.backbone_loader import imagenet_normalize
except ImportError:
    def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
        if x.min() >= -0.05 and x.max() <= 1.05 and x.shape[1] >= 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            return (x[:, :3] - mean) / std
        return x


class BackgroundConditionedModel(nn.Module):
    """
    Mô hình thích ứng camera chưa thấy qua thống kê toàn cục background:
      - Backbone: DINOv3 ViT.
      - Conditioning Layer: FiLM (mặc định), Prompt Tokens, hoặc Cross-Attention.
      - Robustness: Background-Dropout (thay z bằng z_null với xác suất p_drop)
        ngăn chặn overfitting và cho phép suy luận ngay cả khi camera mới chưa có background.
      - Heads:
          1. density_head: Dự đoán mật độ đếm xe (Regression).
          2. congestion_head: Dự đoán 4 cấp độ ùn tắc (Classification).
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 384,
        descriptor_dim: int = 1152,  # 3 * feature_dim
        conditioning_mode: str = "film",  # "film", "prompt", "cross_attn", "none"
        bg_dropout_prob: float = 0.25,
        num_classes: int = 4,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.descriptor_dim = descriptor_dim
        self.mode = conditioning_mode.lower()
        self.bg_dropout_prob = bg_dropout_prob

        # Token null đại diện cho background rỗng khi dropout hoặc thiếu background
        self.null_z = nn.Parameter(torch.zeros(1, descriptor_dim))

        # Khởi tạo khối điều kiện hóa theo mode được chọn
        if self.mode == "film":
            self.conditioning = FiLMConditioningLayer(descriptor_dim, feature_dim)
        elif self.mode == "prompt":
            self.conditioning = PromptTokenConditioningLayer(descriptor_dim, feature_dim, num_prompts=4)
        elif self.mode == "cross_attn":
            self.conditioning = CrossAttentionConditioningLayer(feature_dim, num_heads=4)
        else:
            self.conditioning = None

        # Tầng tổng hợp đặc trưng toàn cục (Global Average Pooling)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 1. Head đếm mật độ xe (Vehicle Counting Regression Head)
        self.density_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.ReLU(),  # Số lượng xe không âm
        )

        # 2. Head phân loại mức ùn tắc (Congestion Classification Head)
        self.congestion_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        rgb_frame: torch.Tensor,
        z_descriptor: Optional[torch.Tensor] = None,
        bg_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb_frame: (B, 3, H, W)
            z_descriptor: (B, descriptor_dim) từ RobustSceneDescriptorExtractor
            bg_tokens: (B, M, D) tokens patch của background (chỉ cần nếu mode == 'cross_attn')
            
        Returns:
            Dict chứa:
              - "count": (B, 1) số lượng xe dự đoán
              - "logits": (B, num_classes) phân bố mức ùn tắc
        """
        B = rgb_frame.shape[0]

        # 1. Trích xuất đặc trưng tokens từ frame camera hiện tại
        rgb_norm = imagenet_normalize(rgb_frame)
        out = self.backbone(rgb_norm)
        if isinstance(out, dict):
            tokens = out.get("x_norm_patchtokens", list(out.values())[0])
        elif isinstance(out, torch.Tensor):
            tokens = out
        else:
            tokens = out[0]

        # 2. Xử lý Background Dropout trong quá trình huấn luyện
        if z_descriptor is not None:
            if self.training and self.bg_dropout_prob > 0:
                drop_mask = (torch.rand(B, 1, device=rgb_frame.device) < self.bg_dropout_prob).float()
                null_expanded = self.null_z.expand(B, -1)
                z_used = (1.0 - drop_mask) * z_descriptor + drop_mask * null_expanded
            else:
                z_used = z_descriptor
        else:
            z_used = self.null_z.expand(B, -1)

        # 3. Áp dụng cơ chế điều kiện hóa
        if self.conditioning is not None:
            if self.mode == "film":
                tokens = self.conditioning(tokens, z_used)
            elif self.mode == "prompt":
                tokens = self.conditioning(tokens, z_used)
            elif self.mode == "cross_attn" and bg_tokens is not None:
                tokens = self.conditioning(tokens, bg_tokens)

        # 4. Gom đặc trưng patch thành 1 vector biểu diễn toàn khung hình: (B, D)
        # tokens: (B, N, D) -> permute -> (B, D, N) -> pool -> (B, D, 1) -> squeeze -> (B, D)
        h_frame = self.pool(tokens.permute(0, 2, 1)).squeeze(-1)

        # 5. Các đầu ra dự đoán
        predicted_count = self.density_head(h_frame)
        predicted_logits = self.congestion_head(h_frame)

        return {
            "count": predicted_count,
            "logits": predicted_logits,
        }
