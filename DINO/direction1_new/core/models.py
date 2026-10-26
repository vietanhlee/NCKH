"""
=============================================================================
 Hướng 1 Mới (Direction 1 New): Kiến trúc Mô hình Student & Teacher
 Gồm DINOHead, DirectionGStudentModel và DirectionGTeacherModel
=============================================================================
"""

import os
import sys
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_dino_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from common.backbone_loader import (
    extract_tokens,
    extract_tokens_with_grad,
    imagenet_normalize,
)


class DINOHead(nn.Module):
    """Đầu MLP chiếu đa lớp + lớp Prototype chuẩn hóa L2 cho DINO / iBOT."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 16384,
        use_bn: bool = False,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.last_layer.weight, std=0.02)
        self.norm_last_layer = norm_last_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if len(shape) == 3:  # (B, N, D)
            B, N, D = shape
            x = x.reshape(B * N, D)
            feat = self.mlp(x)
            feat = F.normalize(feat, dim=-1, p=2)
            if self.norm_last_layer:
                w = F.normalize(self.last_layer.weight, dim=-1, p=2)
                out = F.linear(feat, w)
            else:
                out = self.last_layer(feat)
            return out.reshape(B, N, -1)
        else:  # (B, D)
            feat = self.mlp(x)
            feat = F.normalize(feat, dim=-1, p=2)
            if self.norm_last_layer:
                w = F.normalize(self.last_layer.weight, dim=-1, p=2)
                return F.linear(feat, w)
            else:
                return self.last_layer(feat)


class DirectionGStudentModel(nn.Module):
    """
    Wrapper gom toàn bộ luồng tính toán của Student (Backbone + DINO Head + iBOT Head)
    để phân bổ song song hoàn hảo trên toàn bộ các GPU qua nn.DataParallel.

    Lưu ý nghiệp vụ:
      - Đầu vào ở dải [0, 1]; chuẩn hóa ImageNet được thực hiện bên trong wrapper.
      - `masks` (B, N) bool từ AGM: patch bị che được thay bằng mask token của DINOv3,
        Student phải dự đoán phân phối prototype của Teacher tại các vị trí đó (iBOT MIM).
      - BẮT BUỘC dùng `extract_tokens_with_grad` để gradient chảy về backbone.
    """
    def __init__(self, backbone: nn.Module, dino_head: nn.Module, ibot_head: nn.Module, patch_size: int = 16):
        super().__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head
        self.patch_size = patch_size

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = imagenet_normalize(x)
        cls_token, patch_spatial = extract_tokens_with_grad(
            self.backbone, x, patch_size=self.patch_size, masks=masks
        )
        B, Hp, Wp, D = patch_spatial.shape
        patches = patch_spatial.reshape(B, Hp * Wp, D)
        dino_logits = self.dino_head(cls_token)
        ibot_logits = self.ibot_head(patches)
        return cls_token, patches, dino_logits, ibot_logits


class DirectionGTeacherModel(nn.Module):
    """
    Wrapper gom toàn bộ luồng suy luận của Teacher EMA để tận dụng tối đa Multi-GPU.
    Teacher luôn nhìn ảnh gốc KHÔNG che, KHÔNG gradient.
    """
    def __init__(self, backbone: nn.Module, dino_head: nn.Module, ibot_head: nn.Module, patch_size: int = 16):
        super().__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head
        self.patch_size = patch_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = imagenet_normalize(x)
        cls_token, patch_spatial = extract_tokens(self.backbone, x, patch_size=self.patch_size)
        B, Hp, Wp, D = patch_spatial.shape
        patches = patch_spatial.reshape(B, Hp * Wp, D)
        dino_logits = self.dino_head(cls_token)
        ibot_logits = self.ibot_head(patches)
        return cls_token, patches, dino_logits, ibot_logits
