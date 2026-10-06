"""
=============================================================================
 Hướng 1: BG-Guided DINO — Model Architecture & Projection Heads
 Đóng gói Student-Teacher Framework với Foreground-Aware Masking (FAM)
 Bổ sung Patch-Level iBOT Self-Distillation Head theo chuẩn Q1
=============================================================================
"""

import copy
from typing import List, Optional, Tuple, Union, Dict
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


class DINOHead(nn.Module):
    """
    3-layer Projection Head chuẩn Meta DINOv2 / DINOv3 với L2-normalization
    và cosine prototype classifier để chống sụp đổ biểu diễn (Dimensional Collapse).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        # Prototype classifier không dùng bias, chuẩn hóa vector trọng số
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.trunc_normal_(self.last_layer.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.mlp(x)
        z = F.normalize(z, dim=-1, p=2)
        w = F.normalize(self.last_layer.weight, dim=-1, p=2)
        logits = F.linear(z, w)
        return logits


class BGGuidedDINOModel(nn.Module):
    """
    Đóng gói cặp mạng Student và Teacher chuẩn Meta DINOv2/v3 + iBOT:
    - Student: Nhận cả view toàn vẹn và view bị che theo FAM, dự đoán cả CLS và Patch tokens.
    - Teacher: EMA Momentum Teacher cung cấp mục tiêu giám sát ngữ nghĩa toàn cục và cục bộ.
    """

    def __init__(
        self,
        student_backbone: nn.Module,
        embed_dim: int,
        out_dim: int = 4096,
        patch_out_dim: int = 4096,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.student_backbone = student_backbone
        self.embed_dim = embed_dim

        # Head cho CLS token (DINO Loss)
        self.student_head = DINOHead(
            in_dim=embed_dim,
            out_dim=out_dim,
            bottleneck_dim=bottleneck_dim,
        )
        # Head cho Patch tokens (iBOT Loss)
        self.student_ibot_head = DINOHead(
            in_dim=embed_dim,
            out_dim=patch_out_dim,
            bottleneck_dim=bottleneck_dim,
        )

        # Khởi tạo Teacher là bản sao độc lập của Student
        self.teacher_backbone = copy.deepcopy(student_backbone)
        self.teacher_head = copy.deepcopy(self.student_head)
        self.teacher_ibot_head = copy.deepcopy(self.student_ibot_head)

        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        for p in self.teacher_ibot_head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """Cập nhật trọng số Teacher theo EMA: theta_t <- m * theta_t + (1 - m) * theta_s"""
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data.mul_(momentum).add_((1.0 - momentum) * ps.detach().data)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(momentum).add_((1.0 - momentum) * ps.detach().data)
        for ps, pt in zip(self.student_ibot_head.parameters(), self.teacher_ibot_head.parameters()):
            pt.data.mul_(momentum).add_((1.0 - momentum) * ps.detach().data)

    def extract_features(self, backbone: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trích xuất đồng thời (cls_token, patch_tokens) từ backbone linh hoạt.
        Hỗ trợ đầy đủ chuẩn Meta DINOv2 / DINOv3 API (get_intermediate_layers, forward_features)
        đảm bảo giữ nguyên chuỗi gradient cho cả Student và Teacher.
        """
        x = imagenet_normalize(x)
        # 1. Chuẩn Meta DINOv2 / DINOv3 VisionTransformer (Torch Hub / Official)
        if hasattr(backbone, "get_intermediate_layers"):
            outputs = backbone.get_intermediate_layers(x, n=1, return_class_token=True)
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                if isinstance(outputs[0], tuple):
                    patch_toks, cls_tok = outputs[0]
                else:
                    patch_raw = outputs[0]
                    cls_tok = patch_raw[:, 0]
                    patch_toks = patch_raw[:, 1:]
                return cls_tok, patch_toks

        # 2. Chuẩn forward_features (timm / transformers)
        if hasattr(backbone, "forward_features"):
            feat = backbone.forward_features(x)
            if isinstance(feat, dict):
                cls_tok = feat.get("x_norm_clstoken", None)
                patch_toks = feat.get("x_norm_patchtokens", None)
                if patch_toks is None:
                    patch_toks = feat.get("x_prenorm", None)
                if cls_tok is not None and patch_toks is not None:
                    return cls_tok, patch_toks
            elif isinstance(feat, torch.Tensor):
                if feat.dim() == 3:
                    return feat[:, 0], feat[:, 1:]

        # 3. Standard forward fallback
        out = backbone(x)
        if isinstance(out, dict):
            cls_tok = out.get("x_norm_clstoken", list(out.values())[0])
            patch_toks = out.get("x_norm_patchtokens", None)
            if patch_toks is None:
                patch_toks = cls_tok.unsqueeze(1)
        elif isinstance(out, torch.Tensor):
            if out.dim() == 3:
                cls_tok = out[:, 0]
                patch_toks = out[:, 1:]
            else:
                cls_tok = out
                patch_toks = out.unsqueeze(1)
        else:
            cls_tok = out[0]
            patch_toks = out[1] if len(out) > 1 else cls_tok.unsqueeze(1)
        return cls_tok, patch_toks

    def forward_student(
        self,
        crops: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Chạy Student trên toàn bộ crops (2 Global + N Local).
        Nếu có mask, trả về thêm patch logits của view bị che (Global View 1).
        """
        sizes = [c.shape[-1] for c in crops]
        unique_sizes = list(set(sizes))

        all_cls_outputs = []
        masked_patch_logits = None

        for s in unique_sizes:
            indices = [i for i, sz in enumerate(sizes) if sz == s]
            batch_s = torch.cat([crops[i] for i in indices], dim=0)

            cls_tok, patch_toks = self.extract_features(self.student_backbone, batch_s)
            cls_logits = self.student_head(cls_tok)
            all_cls_outputs.append((indices, cls_logits))

            # Nếu batch chứa Global View 1 (indices có 0) và có mask
            if 0 in indices and mask is not None:
                # Patch tokens tương ứng view 0
                idx_in_batch = indices.index(0)
                B = len(crops[0])
                view0_patches = patch_toks[idx_in_batch * B : (idx_in_batch + 1) * B]  # (B, N_p, D)
                masked_patch_logits = self.student_ibot_head(view0_patches)

        ordered_outs = [None] * len(crops)
        for indices, out in all_cls_outputs:
            chunks = out.chunk(len(indices))
            for i, chunk in zip(indices, chunks):
                ordered_outs[i] = chunk

        total_cls_logits = torch.cat(ordered_outs, dim=0)
        return total_cls_logits, masked_patch_logits

    @torch.no_grad()
    def forward_teacher(
        self,
        global_crops: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chạy Teacher trên 2 Global unmasked views.
        Trả về (cls_logits, patch_logits của global view 1).
        """
        batch = torch.cat(global_crops, dim=0)
        cls_tok, patch_toks = self.extract_features(self.teacher_backbone, batch)
        cls_logits = self.teacher_head(cls_tok)

        B = len(global_crops[0])
        view0_patches = patch_toks[:B]
        patch_logits = self.teacher_ibot_head(view0_patches)
        return cls_logits, patch_logits

    def forward(
        self,
        crops: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        mode: str = "student"
    ):
        if mode == "teacher":
            return self.forward_teacher(crops)
        return self.forward_student(crops, mask)
