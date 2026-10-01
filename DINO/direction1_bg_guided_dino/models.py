"""
=============================================================================
 Hướng 1: BG-Guided DINO — Model Architecture & Projection Head
 Đóng gói Student-Teacher Framework với Dynamic Foreground Masking
=============================================================================
"""

import copy
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Khởi tạo prototype weights
        nn.init.trunc_normal_(self.last_layer.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Vector biểu diễn đặc trưng (B, in_dim)
        Returns:
            Logits phân bố xác suất trên prototypes (B, out_dim)
        """
        z = self.mlp(x)
        z = F.normalize(z, dim=-1, p=2)
        # Chuẩn hóa prototype weights để tính cosine similarity
        w = F.normalize(self.last_layer.weight, dim=-1, p=2)
        logits = F.linear(z, w)
        return logits


class BGGuidedDINOModel(nn.Module):
    """
    Đóng gói cặp mạng Student và Teacher.
    - Student: Nhận cả view đầy đủ lẫn view bị che theo Foreground-Aware Masking.
    - Teacher: Cập nhật trọng số qua Exponential Moving Average (EMA), nhận global unmasked views.
    """

    def __init__(
        self,
        student_backbone: nn.Module,
        embed_dim: int,
        out_dim: int = 4096,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.student_backbone = student_backbone
        self.student_head = DINOHead(
            in_dim=embed_dim,
            out_dim=out_dim,
            bottleneck_dim=bottleneck_dim,
        )

        # Khởi tạo Teacher là bản sao hoàn hảo của Student
        self.teacher_backbone = copy.deepcopy(student_backbone)
        self.teacher_head = copy.deepcopy(self.student_head)

        # Đóng băng gradient hoàn toàn cho Teacher
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """
        Cập nhật trọng số Teacher theo quy tắc EMA:
        $\\theta_t \\leftarrow m \\cdot \\theta_t + (1 - m) \\cdot \\theta_s$
        """
        for param_s, param_t in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)
        for param_s, param_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)

    def forward_student(self, crops: List[torch.Tensor]) -> torch.Tensor:
        """
        Chạy Student qua toàn bộ các crops (2 Global + N Local).
        Tối ưu hóa: Gom các views có cùng kích thước để forward theo batch.
        """
        # Phân nhóm theo resolution
        sizes = [c.shape[-1] for c in crops]
        unique_sizes = list(set(sizes))

        all_outputs = []
        for s in unique_sizes:
            indices = [i for i, sz in enumerate(sizes) if sz == s]
            batch_s = torch.cat([crops[i] for i in indices], dim=0)
            feat = self.student_backbone(batch_s)
            if hasattr(feat, "get") and isinstance(feat, dict):
                feat = feat.get("x_norm_clstoken", list(feat.values())[0])
            elif feat.dim() > 2:
                feat = feat[:, 0]  # Lấy CLS token
            out = self.student_head(feat)
            all_outputs.append((indices, out))

        # Khôi phục thứ tự ban đầu của crops
        ordered_outs = [None] * len(crops)
        for indices, out in all_outputs:
            batch_size = len(out) // len(indices)
            chunks = out.chunk(len(indices))
            for i, chunk in zip(indices, chunks):
                ordered_outs[i] = chunk

        return torch.cat(ordered_outs, dim=0)

    @torch.no_grad()
    def forward_teacher(self, global_crops: List[torch.Tensor]) -> torch.Tensor:
        """
        Chạy Teacher trên 2 Global unmasked views.
        """
        batch = torch.cat(global_crops, dim=0)
        feat = self.teacher_backbone(batch)
        if hasattr(feat, "get") and isinstance(feat, dict):
            feat = feat.get("x_norm_clstoken", list(feat.values())[0])
        elif feat.dim() > 2:
            feat = feat[:, 0]
        out = self.teacher_head(feat)
        return out

    def forward(self, crops: List[torch.Tensor], mode: str = "student") -> torch.Tensor:
        """
        Phương thức forward chuẩn tương thích với PyTorch nn.DataParallel & DDP.
        Args:
            crops: List tensor các view ảnh.
            mode: 'student' (toàn bộ crops) hoặc 'teacher' (2 global unmasked crops).
        """
        if mode == "teacher":
            return self.forward_teacher(crops)
        return self.forward_student(crops)
