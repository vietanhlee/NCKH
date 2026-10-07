"""
=============================================================================
 Hướng 2 Mới: solve_ell.py
 Giải mã ánh sáng ell_t in R^J cho Chế độ suy luận T (camera đã có SceneBasis)
 Sử dụng Weighted Least Squares (IRLS) trên các pixel tĩnh (thời gian < 5 ms)
=============================================================================
"""

from typing import Optional
import torch
import torch.nn.functional as F


def solve_ell(
    I: torch.Tensor,
    E0: torch.Tensor,
    Ej: torch.Tensor,
    w_static: Optional[torch.Tensor] = None,
    iters: int = 4,
) -> torch.Tensor:
    """
    Giải hệ mã ánh sáng ell in R^J cho một frame I:
      min_ell sum_p w(p) || (I - E0) - sum_j ell_j Ej ||^2

    Args:
        I: (3, H, W) Frame ảnh mới
        E0: (3, H, W) Cảnh tĩnh trung bình
        Ej: (J, 3, H_half, W_half) hoặc (J, 3, H, W)
        w_static: (1, H, W) Trọng số tĩnh ban đầu (ví dụ 1 - pi). Nếu None, gán 1.
        iters: Số vòng lặp IRLS (mặc định 4)

    Returns:
        ell: (J,) Vector mã ánh sáng
    """
    device = I.device
    C, H, W = I.shape
    J = Ej.shape[0]

    # Nội suy Ej lên đầy đủ H x W nếu cần
    if Ej.shape[-2:] != (H, W):
        Ej_up = F.interpolate(Ej, size=(H, W), mode="bilinear", align_corners=False)  # (J, 3, H, W)
    else:
        Ej_up = Ej

    # Chuẩn bị ma trận thiết kế:
    # A: (C * H * W, J)
    # y: (C * H * W,) = (I - E0)
    A = Ej_up.permute(1, 2, 3, 0).reshape(-1, J)  # (3*H*W, J)
    y = (I - E0).view(-1)                          # (3*H*W,)

    if w_static is None:
        w = torch.ones((H, W), device=device)
    else:
        w = w_static.squeeze()

    w_flat = w.unsqueeze(0).expand(3, -1, -1).reshape(-1)  # (3*H*W,)
    ell = torch.zeros(J, device=device)

    # Vòng lặp IRLS
    for _ in range(iters):
        # Trọng số căn bậc hai
        sqrt_w = torch.sqrt(w_flat.clamp(min=1e-4)).unsqueeze(-1)  # (3*H*W, 1)
        A_w = A * sqrt_w
        y_w = y * sqrt_w.squeeze(-1)

        # Giải bình phương tối thiểu: (A_w^T A_w + lambda I)^(-1) A_w^T y_w
        AtA = torch.matmul(A_w.T, A_w) + 1e-4 * torch.eye(J, device=device)
        Aty = torch.matmul(A_w.T, y_w)
        ell = torch.linalg.solve(AtA, Aty)

        # Cập nhật phần dư và trọng số IRLS
        pred_delta = torch.matmul(A, ell)
        res = torch.abs(y - pred_delta)
        # Huber / Bisquare weighting
        w_flat = 1.0 / (1.0 + (res / 0.05) ** 2)

    return ell
