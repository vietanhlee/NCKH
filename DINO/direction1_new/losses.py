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

from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko differential entropy estimator (KoLeo) (DINOv2/v3).
    Khuyến khích các feature CLS phân bố đều trên mặt cầu đơn vị, chống sụp đổ biểu diễn (Mode Collapse).
    Bảo toàn gradient liên tục: Chuẩn hóa khoảng cách cực đại rho / 2.0 <= 1.0 trên mặt cầu đơn vị S^{D-1}.
    Khi đó -log(rho / 2.0) luôn >= 0 tự nhiên, loại bỏ hoàn toàn việc clamp(min=0.0) làm triệt tiêu gradient.
    """
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, student_cls: torch.Tensor) -> torch.Tensor:
        # student_cls: (B, D), chuẩn hóa L2 lên mặt cầu đơn vị
        x = F.normalize(student_cls, p=2, dim=-1)
        B = x.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # Khoảng cách Euclidean giữa các cặp điểm: ||x_i - x_j||^2 = 2 - 2 * <x_i, x_j>
        dots = torch.matmul(x, x.T)
        dist_sq = 2.0 - 2.0 * dots
        # Bỏ đường chéo
        dist_sq.fill_diagonal_(float("inf"))
        # Láng giềng gần nhất với clamp chặn an toàn epsilon
        min_dist_sq = dist_sq.min(dim=1)[0]
        min_dist = torch.sqrt(torch.clamp(min_dist_sq, min=self.eps ** 2))
        min_dist = torch.clamp(min_dist, min=self.eps, max=2.0)
        # Chuẩn hóa khoảng cách theo đường kính tối đa 2.0 của mặt cầu đơn vị S^{D-1}
        # dist_norm = min_dist / 2.0 in (0, 1] => -log(dist_norm) >= 0 luôn dương và đạo hàm liên tục!
        dist_norm = min_dist / 2.0
        loss = -torch.mean(torch.log(dist_norm))
        return torch.clamp(loss, max=5.0)


class DINOLoss(nn.Module):
    """
    DINO Cross-Entropy Loss giữa Student và Teacher với Centering & Sharpening.
    """
    def __init__(
        self,
        out_dim: int = 16384,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: Optional[int] = None,
        nepochs: int = 100,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Tự động cân đối số epoch warmup nhiệt độ Teacher phù hợp với tổng số epoch kế hoạch
        if warmup_teacher_temp_epochs is None:
            warmup_teacher_temp_epochs = min(10, max(2, int(nepochs * 0.35)))

        # Lịch tăng nhiệt độ teacher
        self.teacher_temp_schedule = [
            warmup_teacher_temp + (teacher_temp - warmup_teacher_temp) * (i / max(1, warmup_teacher_temp_epochs))
            if i < warmup_teacher_temp_epochs else teacher_temp
            for i in range(max(nepochs, warmup_teacher_temp_epochs))
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
    def compute_distribution_metrics(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        Tính toán các chỉ số giám sát phân bố xác suất để phát hiện sớm phân kỳ hoặc sụp đổ biểu diễn:
        - teacher_entropy: Entropy phân bố Teacher (đo độ phân tán)
        - student_entropy: Entropy phân bố Student
        - kl_div: Khoảng cách phân bố KL(Teacher || Student)
        - active_prototypes: Số lượng prototype có kích hoạt thực tế
        """
        curr_teacher_temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        s_out = student_output / self.student_temp
        s_logprobs = F.log_softmax(s_out, dim=-1)
        s_probs = F.softmax(s_out, dim=-1)
        t_probs = F.softmax((teacher_output - self.center) / curr_teacher_temp, dim=-1)

        t_entropy = -(t_probs * torch.log(t_probs.clamp(min=1e-9))).sum(dim=-1).mean().item()
        s_entropy = -(s_probs * s_logprobs).sum(dim=-1).mean().item()
        kl_div = F.kl_div(s_logprobs, t_probs, reduction="batchmean").item()

        out_dim = teacher_output.shape[-1]
        active_cnt = (t_probs.mean(dim=0) > (1.0 / out_dim)).sum().item()

        return {
            "teacher_entropy": float(f"{t_entropy:.4f}"),
            "student_entropy": float(f"{s_entropy:.4f}"),
            "kl_div": float(f"{kl_div:.4f}"),
            "active_prototypes": int(active_cnt),
            "teacher_temp": float(f"{curr_teacher_temp:.4f}"),
        }

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
    Regional Instance Contrastive (RIC) Loss cải tiến chống bão hòa (Anti-Saturation InfoNCE).
    Gộp đặc trưng vùng phương tiện z_veh theo xác suất tiền cảnh pi.
    Tương phản giữa Student và Teacher của cùng một frame (cặp dương)
    với tập mẫu âm gồm toàn bộ batch, trong đó các mẫu âm cùng camera được tăng trọng số phạt (hard negatives).
    """
    def __init__(self, temperature: float = 0.07, hard_cam_weight: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.hard_cam_weight = hard_cam_weight

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
        
        # Ma trận tương đồng liên view z1 x z2.T và nội view z1 x z1.T
        sim_cross = torch.matmul(z1, z2.T) / self.temperature     # (B, B)
        sim_within = torch.matmul(z1, z1.T) / self.temperature    # (B, B)

        # Mask âm: Các frame khác j != i
        diff_frame = ~torch.eye(B, dtype=torch.bool, device=z1.device)
        same_cam = (cids.unsqueeze(1) == cids.unsqueeze(0)) & diff_frame     # (B, B) cùng cam khác frame

        # Trọng số phạt mẫu âm: cùng camera được phạt mạnh hơn gấp hard_cam_weight lần
        neg_weights = torch.ones((B, B), device=z1.device, dtype=z1.dtype)
        if same_cam.any():
            neg_weights = torch.where(
                same_cam,
                torch.tensor(self.hard_cam_weight, device=z1.device, dtype=z1.dtype),
                neg_weights,
            )
        neg_weights = neg_weights * diff_frame.float()

        # Mẫu số InfoNCE gồm cặp dương + toàn bộ mẫu âm trong batch (cả cross-view và within-view)
        exp_cross = torch.exp(sim_cross) * neg_weights
        exp_within = torch.exp(sim_within) * neg_weights
        exp_pos = torch.exp(sim_pos)

        denom = exp_pos + exp_cross.sum(dim=-1) + exp_within.sum(dim=-1)
        loss = -torch.log(exp_pos / denom.clamp(min=1e-8)).mean()
        return loss


def pool_vehicle_embedding(
    patch_tokens: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Gộp đặc trưng vùng phương tiện (Attention-Weighted Vehicle Pooling) theo bản đồ xác suất tiền cảnh pi.
    Buộc vector đại diện z_veh tập trung hoàn toàn vào các patch chứa xe thay vì nền tĩnh.

    Args:
        patch_tokens: (B, P, D) Đặc trưng patch từ backbone.
        pi: (B, P) hoặc (B, Hp, Wp) Xác suất tiền cảnh phương tiện trong [0, 1].
        eps: Hằng số chống chia cho 0.

    Returns:
        z_veh: (B, D) Vector vùng phương tiện đã chuẩn hóa L2.
    """
    B, P, D = patch_tokens.shape
    w = pi.reshape(B, P, 1).detach().clamp(min=0.0, max=1.0)
    # Cộng thêm một sàn nhỏ 1e-3 để nếu frame hoàn toàn vắng xe, phép gộp vẫn ổn định số học
    w_safe = w + 1e-3
    pooled = (patch_tokens * w_safe).sum(dim=1) / (w_safe.sum(dim=1) + eps)
    return F.normalize(pooled, p=2, dim=-1)


def compute_dann_alpha(progress: float, max_alpha: float = 0.1, gamma: float = 10.0) -> float:
    """
    Lịch trình tăng mượt mà hệ số đảo đạo hàm Ganin et al. (DANN):
    alpha(p) = max_alpha * (2.0 / (1.0 + exp(-gamma * p)) - 1.0)
    Tại p = 0.0 (đầu buổi train): alpha = 0.0 (bảo vệ backbone, tránh xung đột sớm).
    Tại p = 1.0 (cuối buổi train): alpha = max_alpha (0.1).
    """
    p = float(np.clip(progress, 0.0, 1.0))
    schedule = (2.0 / (1.0 + np.exp(-gamma * p))) - 1.0
    return float(max_alpha * schedule)


class _GradientReversalFn(torch.autograd.Function):
    """
    Hàm tự động đảo ngược dấu đạo hàm (Gradient Reversal Layer - Ganin et al.).
    Chiều thuận (Forward): giữ nguyên đầu vào.
    Chiều ngược (Backward): nhân gradient với -alpha để ép backbone xóa thông tin phân biệt miền/camera.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


class CameraAdversarialHead(nn.Module):
    """
    Đầu phân loại đối kháng Camera ID (Camera-Invariance Adversarial Discriminator).
    Sử dụng Gradient Reversal Layer (GRL) để phạt trực tiếp việc mã hóa Camera ID trong CLS token,
    triệt tiêu hoàn toàn Camera ID Shortcut Trap.
    """
    def __init__(self, in_dim: int, num_cams: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.num_cams = max(2, num_cams)
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_cams),
        )

    def ensure_cam_capacity(self, max_cid: int):
        """Mở rộng tầng phân loại nếu gặp camera ID vượt quá num_cams hiện tại."""
        if max_cid < self.num_cams:
            return
        new_cams = max(max_cid + 1, self.num_cams * 2)
        old_linear = self.net[-1]
        device = old_linear.weight.device
        dtype = old_linear.weight.dtype
        new_linear = nn.Linear(self.hidden_dim, new_cams, device=device, dtype=dtype)
        with torch.no_grad():
            new_linear.weight[:self.num_cams].copy_(old_linear.weight)
            new_linear.bias[:self.num_cams].copy_(old_linear.bias)
        self.net[-1] = new_linear
        self.num_cams = new_cams

    def forward(self, cls_feat: torch.Tensor, cids: torch.Tensor, grl_alpha: float = 0.5) -> torch.Tensor:
        """
        cls_feat: (B, D) Đặc trưng CLS từ Student.
        cids: (B,) Nhãn Camera ID.
        grl_alpha: Hệ số cường độ đảo chiều gradient (mặc định tăng lên để phản kháng hiệu dụng).
        """
        max_cid = int(cids.max().item()) if cids.numel() > 0 else 0
        self.ensure_cam_capacity(max_cid)
        x_norm = F.normalize(cls_feat, p=2, dim=-1)
        x_rev = _GradientReversalFn.apply(x_norm, grl_alpha)
        logits = self.net(x_rev)
        return F.cross_entropy(logits, cids.long())


class PatchVarianceRegularizer(nn.Module):
    """
    Điều hòa phương sai mức Patch (VICReg-style Variance Regularizer):
    Ngăn chặn triệt để hiện tượng Sụp đổ Chiều (Patch Dimensional Collapse),
    ép các chiều đặc trưng của patch tokens phải phân tán độc lập,
    tránh việc biểu diễn co cụm về mặt phẳng 2D-3D (Top-3 PC chiếm 88% variance).
    """
    def __init__(self, target_std: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.target_std = target_std
        self.eps = eps

    def forward(self, patch_tokens: torch.Tensor, pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        patch_tokens: (B, N, D) Ma trận patch tokens từ Student.
        pi: (B, N) Bản đồ tiền cảnh TAM (ưu tiên duy trì phương sai vùng phương tiện).
        """
        B, N, D = patch_tokens.shape
        flat_tokens = patch_tokens.view(B * N, D)

        if pi is not None:
            # Lọc các patch tiền cảnh có xác suất cao
            fg_mask = (pi.view(-1) > 0.25)
            if fg_mask.sum() >= max(16, D // 4):
                flat_tokens = flat_tokens[fg_mask]

        # Tính độ lệch chuẩn theo từng chiều đặc trưng D
        std = torch.sqrt(flat_tokens.var(dim=0) + self.eps)
        # Hinge loss: phạt nếu std nhỏ hơn target_std
        std_loss = torch.mean(F.relu(self.target_std - std))
        return std_loss


class FeatureAnchorLoss(nn.Module):
    """
    Neo giữ không gian đặc trưng (Feature Anchoring Loss):
    Ngăn chặn Student bị trôi dạt quá xa (representation drift) khỏi
    không gian biểu diễn ngữ nghĩa trù mật ban đầu của DINOv3 Pretrained.
    """
    def __init__(self):
        super().__init__()

    def forward(self, student_patches: torch.Tensor, pretrained_patches: torch.Tensor) -> torch.Tensor:
        """
        student_patches: (B, N, D) Đặc trưng từ Student.
        pretrained_patches: (B, N, D) Đặc trưng từ Vanilla DINOv3 Pretrained đóng băng (đã detach).
        """
        s_norm = F.normalize(student_patches, p=2, dim=-1)
        p_norm = F.normalize(pretrained_patches.detach(), p=2, dim=-1)
        cos_sim = (s_norm * p_norm).sum(dim=-1)  # (B, N)
        return torch.mean(1.0 - cos_sim)


