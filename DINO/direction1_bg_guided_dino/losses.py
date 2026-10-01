"""
=============================================================================
 Hướng 1: BG-Guided DINO — Loss Functions
 Hàm mất mát Tự chưng cất (Self-Distillation) với Teacher Centering & Sharpening
 Kết hợp Foreground-Guided Consistency Loss
=============================================================================
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BGGuidedDINOLoss(nn.Module):
    """
    Hàm mất mát Cross-Entropy chưng cất tự thân của DINO kết hợp:
      - Teacher Centering: $C \\leftarrow m \\cdot C + (1-m) \\cdot \\text{Mean}(g_t)$
      - Teacher Sharpening (Nhiệt độ $\\tau_t$ thấp): Tạo phân bố tập trung
      - Student Temperature ($\\tau_s = 0.1$)
      - Đảm bảo toán học chống sụp đổ biểu diễn (Mode Collapse / Dimensional Collapse)
    """

    def __init__(
        self,
        out_dim: int = 4096,
        ncrops: int = 6,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 10,
        nepochs: int = 50,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Thống kê chẩn đoán sụp đổ biểu diễn
        self.last_entropy = 0.0

        # Lịch trình tăng nhiệt độ của Teacher (Cosine Warmup)
        warmup_teacher_temp_epochs = min(warmup_teacher_temp_epochs, nepochs)
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(0, nepochs - warmup_teacher_temp_epochs)) * teacher_temp,
        ))

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """
        Tính toán Cross-Entropy Loss giữa phân bố xác suất Student và Teacher.

        Args:
            student_output: Tensor logits từ Student (ncrops * B, out_dim).
            teacher_output: Tensor logits từ Teacher (2 * B, out_dim).
            epoch: Epoch hiện tại để áp dụng lịch trình nhiệt độ.

        Returns:
            total_loss: Giá trị loss trung bình.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]

        # Áp dụng Centering và Sharpening cho Teacher
        teacher_centered = (teacher_output.float() - self.center) / temp
        teacher_probs = F.softmax(teacher_centered, dim=-1)

        # Chẩn đoán trạng thái hội tụ (Entropy đo độ đồng đều)
        with torch.no_grad():
            p = teacher_probs.detach()
            self.last_entropy = float(-(p * torch.log(p + 1e-12)).sum(-1).mean())

        teacher_out_chunks = teacher_probs.detach().chunk(2)

        total_loss = 0.0
        n_terms = 0

        # Đối sánh 2 view của Teacher với tất cả các view của Student
        for i_t, t_prob in enumerate(teacher_out_chunks):
            for i_s, s_logit in enumerate(student_out):
                # Bỏ qua nếu là cùng 1 view
                if i_s == i_t:
                    continue
                loss = torch.sum(-t_prob * F.log_softmax(s_logit, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_terms += 1

        total_loss /= max(1, n_terms)

        # Cập nhật center vector cho Teacher
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """Cập nhật giá trị trung bình động (EMA) của vector trung tâm."""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True) / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)
