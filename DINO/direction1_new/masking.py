"""
=============================================================================
 Hướng G: AGM (Atypicality-Guided Masking)
 Chiến lược che phân tầng có trọng số theo bản đồ khác thường TAM
 Kiểm soát độc lập tỷ lệ ngân sách che vùng xe (phi) và giới hạn che tối đa (q_max)
=============================================================================
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def sample_without_replacement(
    indices: torch.Tensor,
    k: int,
    weights: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Lấy mẫu không hoàn lại k phần tử từ tập indices bằng Gumbel-max trick nếu có weights.
    """
    if k <= 0 or indices.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=indices.device)
    k = min(k, indices.numel())

    if weights is None:
        perm = torch.randperm(indices.numel(), generator=generator, device=indices.device)
        return indices[perm[:k]]
    else:
        # Gumbel top-k: key = log(w) + Gumbel(0, 1) = log(w) - log(-log(U))
        w = weights.clamp(min=1e-6)
        u = torch.rand(w.shape, generator=generator, device=w.device).clamp(min=1e-6, max=1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(u))
        keys = torch.log(w) + gumbel
        topk_idx = torch.topk(keys, k=k, dim=0)[1]
        return indices[topk_idx]


def agm_sample(
    pi: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    ratio: float = 0.35,
    phi: float = 0.5,
    q_max: float = 0.6,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Lấy mẫu mask cho 1 crop view theo chính sách AGM phân tầng.

    Args:
        pi: Tensor (N,) xác suất tiền cảnh của crop view (ví dụ N = 196 cho 14x14).
        valid: Tensor (N,) bool, True nếu vị trí có TAM xác định.
        ratio: Tỷ lệ tổng số patch bị che (ví dụ ratio in [0.1, 0.5]).
        phi: Tỷ lệ ngân sách che dành cho vùng xe (mặc định 0.5).
        q_max: Tỷ lệ tối đa patch xe được phép che (mặc định 0.6, giữ lại >= 40% làm ngữ cảnh).
        generator: Bộ sinh số ngẫu nhiên torch.Generator.

    Returns:
        mask: Tensor (N,) bool, True là patch bị che.
    """
    N = pi.numel()
    device = pi.device
    if valid is None:
        valid = torch.ones(N, dtype=torch.bool, device=device)

    n_mask = int(round(ratio * N))
    if n_mask <= 0:
        return torch.zeros(N, dtype=torch.bool, device=device)

    F_mask = (pi > 0.5) & valid       # Vùng xe
    B_mask = ~F_mask                   # Vùng nền & không xác định

    n_fg_total = int(F_mask.sum().item())
    n_bg_total = int(B_mask.sum().item())

    # 1. Phân bổ ngân sách che vùng xe k_fg
    # Giới hạn bởi phi * n_mask và trần q_max * n_fg_total
    k_fg = min(int(round(phi * n_mask)), int(math.floor(q_max * n_fg_total)))
    k_fg = max(0, k_fg)

    # 2. Ngân sách che vùng nền k_bg
    k_bg = min(n_mask - k_fg, n_bg_total)
    # Nếu nền không đủ, bù lại cho xe nếu còn hạn mức q_max
    if k_fg + k_bg < n_mask and n_fg_total > k_fg:
        extra_fg = min(n_mask - (k_fg + k_bg), int(math.floor(q_max * n_fg_total)) - k_fg)
        k_fg += max(0, extra_fg)

    fg_indices = torch.where(F_mask)[0]
    bg_indices = torch.where(B_mask)[0]

    fg_weights = pi[fg_indices] if n_fg_total > 0 else None
    chosen_fg = sample_without_replacement(fg_indices, k_fg, weights=fg_weights, generator=generator)
    chosen_bg = sample_without_replacement(bg_indices, k_bg, weights=None, generator=generator)

    mask = torch.zeros(N, dtype=torch.bool, device=device)
    if chosen_fg.numel() > 0:
        mask[chosen_fg] = True
    if chosen_bg.numel() > 0:
        mask[chosen_bg] = True

    return mask


def map_pi_to_crop(
    pi_full: torch.Tensor,
    crop_box: Tuple[int, int, int, int],
    orig_size: Tuple[int, int] = (256, 448),
    target_grid: Tuple[int, int] = (14, 14),
    horizontal_flip: bool = False,
) -> torch.Tensor:
    """
    Ánh xạ ma trận xác suất tiền cảnh pi_full từ khung hình đầy đủ (Hp, Wp)
    lên vùng RandomResizedCrop của Global View, bảo đảm đồng bộ không gian chính xác.

    Args:
        pi_full: Tensor (Hp, Wp) ví dụ (16, 28)
        crop_box: (top, left, height, width) tính trên kích thước pixel ảnh gốc
        orig_size: (H_orig, W_orig), mặc định (256, 448)
        target_grid: (target_Hp, target_Wp), mặc định (14, 14) cho crop 224x224
        horizontal_flip: True nếu crop đã bị lật ngang ngẫu nhiên

    Returns:
        pi_crop: Tensor (target_Hp * target_Wp,)
    """
    H_orig, W_orig = orig_size
    top, left, h, w = crop_box
    device = pi_full.device

    # Chuẩn hóa về tọa độ grid_sample [-1, 1]
    # Lấy mẫu pi_full (1, 1, Hp, Wp)
    pi_4d = pi_full.view(1, 1, pi_full.shape[0], pi_full.shape[1])

    # Tọa độ bounding box trong không gian [-1, 1]
    y0 = 2.0 * top / H_orig - 1.0
    y1 = 2.0 * (top + h) / H_orig - 1.0
    x0 = 2.0 * left / W_orig - 1.0
    x1 = 2.0 * (left + w) / W_orig - 1.0

    if horizontal_flip:
        x0, x1 = x1, x0

    t_hp, t_wp = target_grid
    lin_y = torch.linspace(y0, y1, t_hp, device=device)
    lin_x = torch.linspace(x0, x1, t_wp, device=device)
    grid_y, grid_x = torch.meshgrid(lin_y, lin_x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, t_hp, t_wp, 2)

    pi_sampled = F.grid_sample(pi_4d, grid, mode="bilinear", align_corners=False)  # (1, 1, t_hp, t_wp)
    pi_crop = pi_sampled.view(-1)  # (t_hp * t_wp,)
    return pi_crop.clamp(0.0, 1.0)
