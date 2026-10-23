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
    mode: str = "patch",
    grid_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Lấy mẫu mask cho 1 crop view theo chính sách AGM phân tầng (hỗ trợ patch-wise, block-wise và random).

    Args:
        pi: Tensor (N,) hoặc (Hp, Wp) xác suất tiền cảnh của crop view.
        valid: Tensor (N,) bool, True nếu vị trí có TAM xác định.
        ratio: Tỷ lệ tổng số patch bị che (ví dụ ratio in [0.1, 0.5]).
        phi: Tỷ lệ ngân sách che dành cho vùng xe (mặc định 0.5).
        q_max: Tỷ lệ tối đa patch xe được phép che (mặc định 0.6, giữ lại >= 40% làm ngữ cảnh).
        generator: Bộ sinh số ngẫu nhiên torch.Generator.
        mode: Chế độ che {'patch', 'block', 'random'}.
              - 'patch': Lấy mẫu phân tầng có trọng số Gumbel-max.
              - 'block': Gom cụm che theo khối 2x2 quanh tâm phương tiện để tránh nội suy cục bộ dễ dàng.
              - 'random': Che ngẫu nhiên đồng nhất (dùng cho Ablation Study).
        grid_shape: Kích thước lưới 2D (Hp, Wp), mặc định tự suy ra (16, 28) nếu N=448 hoặc (14, 14) nếu N=196.

    Returns:
        mask: Tensor (N,) bool, True là patch bị che.
    """
    orig_pi_shape = pi.shape
    pi_flat = pi.reshape(-1)
    N = pi_flat.numel()
    device = pi_flat.device
    if valid is None:
        valid_flat = torch.ones(N, dtype=torch.bool, device=device)
    else:
        valid_flat = valid.reshape(-1)

    n_mask = int(round(ratio * N))
    if n_mask <= 0:
        return torch.zeros(N, dtype=torch.bool, device=device)

    # Nhánh 1: Che ngẫu nhiên thuần túy (phục vụ Ablation AGM vs Random Mask)
    if mode == "random":
        all_indices = torch.arange(N, device=device)
        chosen = sample_without_replacement(all_indices, n_mask, weights=None, generator=generator)
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[chosen] = True
        return mask

    # Ngưỡng phân tầng xe linh hoạt: nếu ban đêm hoặc xe nhỏ khiến ít patch đạt > 0.5,
    # tự động hạ ngưỡng xuống 0.35 để không bị rơi vào tình trạng bỏ sót xe
    thr_fg = 0.50
    if (pi_flat >= 0.50).sum() < 12 and (pi_flat >= 0.35).sum() >= 12:
        thr_fg = 0.35

    F_mask = (pi_flat >= thr_fg) & valid_flat  # Vùng xe
    B_mask = ~F_mask                           # Vùng nền & không xác định

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

    # Nhánh 2: Che theo khối (Block-wise Masking chuẩn iBOT / SimMIM)
    if mode == "block":
        if grid_shape is None:
            if len(orig_pi_shape) == 2:
                Hp, Wp = orig_pi_shape
            elif N == 448:
                Hp, Wp = 16, 28
            else:
                side = int(math.isqrt(N))
                Hp, Wp = (side, N // max(1, side))
        else:
            Hp, Wp = grid_shape

        if Hp * Wp == N:
            mask_2d = torch.zeros((Hp, Wp), dtype=torch.bool, device=device)

            # A. Gieo khối 2x2 cho vùng phương tiện (k_fg):
            # Luôn giữ trọn vẹn khối 2x2 (4 patches) bao quanh xe và vi ngữ cảnh lân cận,
            # đồng thời tôn trọng nghiêm ngặt trần q_max của vùng xe.
            max_fg_patches = int(math.floor(q_max * n_fg_total))
            if k_fg > 0 and fg_indices.numel() > 0 and max_fg_patches > 0:
                fg_weights = pi_flat[fg_indices].clone()
                # Gieo từng khối 2x2 cho đến khi đạt k_fg hoặc chạm trần max_fg_patches
                while mask_2d.view(-1)[fg_indices].sum().item() < k_fg:
                    unmasked_fg = fg_indices[~mask_2d.view(-1)[fg_indices]]
                    if unmasked_fg.numel() == 0:
                        break
                    w = fg_weights[~mask_2d.view(-1)[fg_indices]]
                    seed = sample_without_replacement(unmasked_fg, 1, weights=w, generator=generator)[0].item()
                    r = min(int(seed // Wp), Hp - 2)
                    c = min(int(seed % Wp), Wp - 2)
                    candidate = mask_2d.clone()
                    candidate[r:r+2, c:c+2] = True
                    if candidate.view(-1)[fg_indices].sum().item() <= max_fg_patches:
                        mask_2d[r:r+2, c:c+2] = True
                    else:
                        break

            # B. Gieo khối 2x2 cho vùng nền (k_bg) thay vì rải rác muối tiêu 1x1:
            cur_masked = int(mask_2d.sum().item())
            while cur_masked + 3 < n_mask:
                avail_bg_mask = B_mask.view(Hp, Wp) & (~mask_2d)
                avail_bg_indices = torch.where(avail_bg_mask.view(-1))[0]
                if avail_bg_indices.numel() == 0:
                    break
                seed = sample_without_replacement(avail_bg_indices, 1, weights=None, generator=generator)[0].item()
                r = min(int(seed // Wp), Hp - 2)
                c = min(int(seed % Wp), Wp - 2)
                mask_2d[r:r+2, c:c+2] = True
                cur_masked = int(mask_2d.sum().item())

            # C. Tinh chỉnh chính xác số lượng patch bị che về n_mask:
            # Bù các patch còn thiếu (1-3 patch) bằng cách mở rộng tiếp giáp khối hiện hữu,
            # tuyệt đối không rắc 1x1 ngẫu nhiên để tránh sinh hạt muối tiêu cô lập.
            cur_masked = int(mask_2d.sum().item())
            if cur_masked < n_mask:
                n_add = n_mask - cur_masked
                # Tìm các ô chưa che tiếp giáp trực tiếp (4 hướng) với các ô đã che
                adj_mask = torch.zeros((Hp, Wp), dtype=torch.bool, device=device)
                adj_mask[:-1, :] |= mask_2d[1:, :]
                adj_mask[1:, :] |= mask_2d[:-1, :]
                adj_mask[:, :-1] |= mask_2d[:, 1:]
                adj_mask[:, 1:] |= mask_2d[:, :-1]
                adj_unmasked = adj_mask & (~mask_2d)
                adj_indices = torch.where(adj_unmasked.view(-1))[0]
                if adj_indices.numel() >= n_add:
                    chosen = sample_without_replacement(adj_indices, n_add, weights=None, generator=generator)
                    mask_2d.view(-1)[chosen] = True
                else:
                    if adj_indices.numel() > 0:
                        mask_2d.view(-1)[adj_indices] = True
                    remain = n_mask - int(mask_2d.sum().item())
                    if remain > 0:
                        rem_idx = torch.where(~mask_2d.view(-1))[0]
                        chosen = sample_without_replacement(rem_idx, remain, weights=None, generator=generator)
                        mask_2d.view(-1)[chosen] = True
            elif cur_masked > n_mask:
                # Nếu thừa 1-3 patch do khối 2x2: gỡ bỏ an toàn từ các ô có >= 2 láng giềng
                n_drop = cur_masked - n_mask
                for _ in range(n_drop):
                    nbr_count = torch.zeros((Hp, Wp), dtype=torch.int32, device=device)
                    nbr_count[:-1, :] += mask_2d[1:, :].int()
                    nbr_count[1:, :] += mask_2d[:-1, :].int()
                    nbr_count[:, :-1] += mask_2d[:, 1:].int()
                    nbr_count[:, 1:] += mask_2d[:, :-1].int()
                    nbr_count = nbr_count * mask_2d.int()

                    safe_candidates = (mask_2d) & (nbr_count >= 2)
                    safe_idx = torch.where(safe_candidates.view(-1))[0]
                    if safe_idx.numel() > 0:
                        max_nbrs = nbr_count.view(-1)[safe_idx].max()
                        best_candidates = safe_idx[nbr_count.view(-1)[safe_idx] == max_nbrs]
                        drop_one = best_candidates[0].item()
                        mask_2d.view(-1)[drop_one] = False
                    else:
                        rem_m = torch.where(mask_2d.view(-1))[0]
                        mask_2d.view(-1)[rem_m[0]] = False

            return mask_2d.view(-1)
        else:
            fg_weights = pi_flat[fg_indices] if n_fg_total > 0 else None
            chosen_fg = sample_without_replacement(fg_indices, k_fg, weights=fg_weights, generator=generator)
            chosen_bg = sample_without_replacement(bg_indices, k_bg, weights=None, generator=generator)
    else:
        fg_weights = pi_flat[fg_indices] if n_fg_total > 0 else None
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
