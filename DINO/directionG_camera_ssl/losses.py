"""
=============================================================================
 Hướng G: Losses
 Tập hợp các hàm mất mát tự giám sát:
 1. DINOLoss (CLS token distillation with centering & sharpening)
 2. iBOTLoss (MIM loss trên các patch AGM bị che với trọng số beta*pi)
 3. KoLeoLoss (Entropy regularizer)
 4. RICLoss (Regional Instance Contrastive giữa các frame cùng camera)
=============================================================================
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko differential entropy estimator (KoLeo).
    Khuyến khích các feature phân bố đều trên mặt cầu đơn vị, chống sụp đổ biểu diễn.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, student_cls: torch.Tensor) -> torch.Tensor:
        # student_cls: (B, D), giả định đã chuẩn hóa L2
        x = F.normalize(student_cls, p=2, dim=-1)
        B = x.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # Khoảng cách Euclidean giữa các cặp điểm: ||x_i - x_j||^2 = 2 - 2 * <x_i, x_j>
        dots = torch.matmul(x, x.T)
        dist_sq = 2.0 - 2.0 * dots
        # Bỏ đường chéo
        dist_sq.fill_diagonal_(float("inf"))
        # Láng giềng gần nhất
        min_dist = torch.sqrt(torch.clamp(dist_sq.min(dim=1)[0], min=self.eps))
        loss = -torch.mean(torch.log(min_dist + self.eps))
        return loss


class DINOLoss(nn.Module):
    """
    DINO Cross-Entropy Loss giữa Student và Teacher với Centering & Sharpening.
    """
    def __init__(
        self,
        out_dim: int = 16384,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 10,
        nepochs: int = 100,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Lịch tăng nhiệt độ teacher
        self.teacher_temp_schedule = [
            warmup_teacher_temp + (teacher_temp - warmup_teacher_temp) * (i / max(1, warmup_teacher_temp_epochs))
            if i < warmup_teacher_temp_epochs else teacher_temp
            for i in range(nepochs)
        ]

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        """
        student_output: (B_student, out_dim)
        teacher_output: (B_teacher, out_dim)
        """
        curr_teacher_temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]

        # Student probabilities
        s_out = student_output / self.student_temp
        s_logprobs = F.log_softmax(s_out, dim=-1)

        # Teacher probabilities có trừ center
        t_out = F.softmax((teacher_output - self.center) / curr_teacher_temp, dim=-1).detach()

        # Cross entropy loss
        loss = -torch.sum(t_out * s_logprobs, dim=-1).mean()

        # Cập nhật center
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center.copy_(self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum))


class iBOTPatchLoss(nn.Module):
    """
    MIM Loss trên các patch token bị che bởi AGM.
    Cho phép nhân hệ số trọng số tiền cảnh (1 + beta * pi).
    """
    def __init__(
        self,
        out_dim: int = 16384,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_patch_logits: torch.Tensor,
        teacher_patch_logits: torch.Tensor,
        mask: torch.Tensor,
        pi: Optional[torch.Tensor] = None,
        beta: float = 0.0,
    ) -> torch.Tensor:
        """
        student_patch_logits: (B, N_patches, out_dim)
        teacher_patch_logits: (B, N_patches, out_dim)
        mask: (B, N_patches) bool, True là patch bị che
        pi: (B, N_patches) xác suất tiền cảnh từ TAM
        """
        if not mask.any():
            return torch.tensor(0.0, device=student_patch_logits.device, requires_grad=True)

        # Lọc các patch bị che
        s_masked = student_patch_logits[mask]  # (K_masked, out_dim)
        t_masked = teacher_patch_logits[mask]  # (K_masked, out_dim)

        s_logprobs = F.log_softmax(s_masked / self.student_temp, dim=-1)
        t_probs = F.softmax((t_masked - self.center) / self.teacher_temp, dim=-1).detach()

        ce_per_patch = -torch.sum(t_probs * s_logprobs, dim=-1)  # (K_masked,)

        if beta > 0.0 and pi is not None:
            pi_masked = pi[mask].detach()
            weights = 1.0 + beta * pi_masked
            loss = torch.mean(ce_per_patch * weights)
        else:
            loss = torch.mean(ce_per_patch)

        # Cập nhật center
        self.update_center(t_masked)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_patches: torch.Tensor):
        if teacher_patches.shape[0] > 0:
            batch_center = torch.mean(teacher_patches, dim=0, keepdim=True)
            self.center.copy_(self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum))


class RICLoss(nn.Module):
    """
    Regional Instance Contrastive (RIC) Loss.
    Tập trung vector vùng xe z_i (gộp theo trọng số pi) của các view khác nhau cùng 1 frame (cặp dương)
    và tương phản với các frame khác của cùng camera (cặp âm).
    """
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        cids: torch.Tensor,
    ) -> torch.Tensor:
        """
        z1: (B, D) Vector vùng xe từ view 1 (đã normalize L2)
        z2: (B, D) Vector vùng xe từ view 2 (đã normalize L2)
        cids: (B,) Camera IDs
        """
        B = z1.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        sim_pos = torch.sum(z1 * z2, dim=-1) / self.temperature  # (B,)
        
        # Ma trận tương đồng toàn phần z1 x z2.T
        all_sim = torch.matmul(z1, z2.T) / self.temperature     # (B, B)

        # Mask âm: Các frame khác (j != i) nhưng CÙNG CAMERA
        same_cam = (cids.unsqueeze(1) == cids.unsqueeze(0))     # (B, B) bool
        diff_frame = ~torch.eye(B, dtype=torch.bool, device=z1.device)
        hard_neg_mask = same_cam & diff_frame

        if not hard_neg_mask.any():
            # InfoNCE tiêu chuẩn
            labels = torch.arange(B, device=z1.device)
            return F.cross_entropy(all_sim, labels)

        # Mẫu số InfoNCE
        exp_sim = torch.exp(all_sim)
        exp_pos = torch.exp(sim_pos)
        # Chỉ gom pos + các negatives cùng camera
        denom = exp_pos + (exp_sim * hard_neg_mask.float()).sum(dim=-1)
        loss = -torch.log(exp_pos / denom.clamp(min=1e-8)).mean()
        return loss
