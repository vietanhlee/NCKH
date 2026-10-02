"""
=============================================================================
 Hướng 1: BG-Guided DINO — Loss Functions
 Hàm mất mát Tự chưng cất liên hoàn kết hợp DINO CLS Loss và iBOT Patch Loss
 Theo chuẩn nghiên cứu Q1: L = L_DINO + lambda_ibot * L_iBOT
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class BGGuidedDINOLoss(nn.Module):
    """
    Hàm mất mát chưng cất tự thân đa tầng:
      1. Global DINO Loss trên [CLS] tokens giữa mọi views
      2. Patch-level iBOT Loss trên các patch bị che bởi FAM
    """

    def __init__(
        self,
        out_dim: int = 4096,
        patch_out_dim: int = 4096,
        ncrops: int = 6,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 10,
        nepochs: int = 50,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        lambda_ibot: float = 1.0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.lambda_ibot = lambda_ibot

        self.register_buffer("center_cls", torch.zeros(1, out_dim))
        self.register_buffer("center_patch", torch.zeros(1, 1, patch_out_dim))

        self.last_entropy = 0.0

        warmup_teacher_temp_epochs = min(warmup_teacher_temp_epochs, nepochs)
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(0, nepochs - warmup_teacher_temp_epochs)) * teacher_temp,
        ))

    def forward(
        self,
        student_cls: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        teacher_cls: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        student_patch: Optional[torch.Tensor] = None,
        teacher_patch: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        """
        Tính toán tổng mất mát DINO + iBOT.
        Hỗ trợ cả input dạng Tensor lẫn Tuple (cls_out, patch_out) từ forward_student/teacher.
        """
        if isinstance(student_cls, (tuple, list)):
            if len(student_cls) > 1 and student_patch is None:
                student_patch = student_cls[1]
            student_cls = student_cls[0]

        if isinstance(teacher_cls, (tuple, list)):
            if len(teacher_cls) > 1 and teacher_patch is None:
                teacher_patch = teacher_cls[1]
            teacher_cls = teacher_cls[0]

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]

        # 1. CLS DINO Distillation Loss
        student_out = (student_cls / self.student_temp).chunk(self.ncrops)
        teacher_centered = (teacher_cls.float() - self.center_cls) / temp
        teacher_probs = F.softmax(teacher_centered, dim=-1)

        with torch.no_grad():
            p = teacher_probs.detach()
            self.last_entropy = float(-(p * torch.log(p + 1e-12)).sum(-1).mean())

        teacher_chunks = teacher_probs.detach().chunk(2)

        cls_loss = 0.0
        n_terms = 0
        for i_t, t_prob in enumerate(teacher_chunks):
            for i_s, s_logit in enumerate(student_out):
                if i_s == i_t:
                    continue
                loss = torch.sum(-t_prob * F.log_softmax(s_logit, dim=-1), dim=-1)
                cls_loss += loss.mean()
                n_terms += 1
        cls_loss /= max(1, n_terms)

        # 2. Patch-level iBOT Loss
        patch_loss = torch.tensor(0.0, device=student_cls.device)
        if (
            student_patch is not None
            and teacher_patch is not None
            and mask is not None
            and mask.sum() > 0
        ):
            # student_patch: (B, N_p, D), teacher_patch: (B, N_p, D), mask: (B, N_p) bool
            t_patch_centered = (teacher_patch.float() - self.center_patch) / temp
            t_patch_prob = F.softmax(t_patch_centered, dim=-1).detach()
            s_patch_log_prob = F.log_softmax(student_patch / self.student_temp, dim=-1)

            loss_per_patch = torch.sum(-t_patch_prob * s_patch_log_prob, dim=-1)  # (B, N_p)
            if mask.dtype == torch.bool:
                masked_loss = loss_per_patch[mask]
            else:
                masked_loss = loss_per_patch[mask > 0.5]

            if masked_loss.numel() > 0:
                patch_loss = masked_loss.mean()

            # Cập nhật center_patch
            with torch.no_grad():
                self.center_patch = (
                    self.center_momentum * self.center_patch
                    + (1.0 - self.center_momentum) * teacher_patch.mean(dim=(0, 1), keepdim=True)
                )

        # Cập nhật center_cls
        with torch.no_grad():
            self.center_cls = (
                self.center_momentum * self.center_cls
                + (1.0 - self.center_momentum) * teacher_cls.mean(dim=0, keepdim=True)
            )

        total_loss = cls_loss + self.lambda_ibot * patch_loss
        return total_loss
