"""
=============================================================================
 Hướng 1 Mới (Direction 1 New): Lịch trình Learning Rate liên tục C^0
=============================================================================
"""

import numpy as np
import torch


def build_smooth_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 1,
    eta_min: float = 1e-6,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Xây dựng Lịch trình Learning Rate mượt mà kết hợp Linear Warmup và Cosine Annealing dài hạn.
    Đảm bảo tính liên tục tuyệt đối C^0 Continuity qua mọi phiên làm việc (Resume Sessions):
    - Tại bất kỳ epoch t trong [0, total_epochs - 1], LR là hàm đơn trị phụ thuộc vào total_epochs kế hoạch.
    - Khi resume session tiếp theo (start_epoch), LR xuất phát chính xác từ giá trị tiếp nối,
      triệt tiêu hoàn toàn hiện tượng sốc gradient (LR Shock / Kick).
    """
    base_lr = float(optimizer.param_groups[0]["initial_lr"] if "initial_lr" in optimizer.param_groups[0] else optimizer.param_groups[0]["lr"])
    eta_min_ratio = float(eta_min) / max(1e-12, base_lr)

    def lr_lambda(current_epoch: int) -> float:
        if warmup_epochs > 0 and current_epoch < warmup_epochs:
            alpha = float(current_epoch + 1) / float(warmup_epochs + 1)
            return eta_min_ratio + (1.0 - eta_min_ratio) * alpha
        else:
            cosine_steps = max(1, total_epochs - max(0, warmup_epochs))
            progress = float(current_epoch - max(0, warmup_epochs)) / float(cosine_steps)
            progress = min(max(progress, 0.0), 1.0)
            return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + np.cos(np.pi * progress))

    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
