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

    # Nhánh 2: Che theo khối (Block-wise Masking chuẩn iBOT / SimMIM dựa trên Super-Block Grid)
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

        # Khi kích thước lưới chia hết cho 2 (chuẩn ViT-S 16x28):
        # Sử dụng mô hình Super-Block Grid 2x2 chuẩn mực của SimMIM & iBOT:
        # Gom 448 patches thành 112 siêu khối 2x2 không chồng lấn.
        # Vector hóa hoàn toàn, triệt tiêu 100% vòng lặp tuần tự CPU.
        if Hp * Wp == N and Hp % 2 == 0 and Wp % 2 == 0:
            Hb, Wb = Hp // 2, Wp // 2
            n_super_blocks = Hb * Wb

            # 1. Mức độ tiền cảnh của từng siêu khối 2x2 = MAX của 4 patch (không dùng trung bình):
            # Xe máy chiếm đúng 1 patch (pi cao) vẫn đưa cả siêu khối vào tầng xe. Nếu dùng trung bình,
            # 1 patch xe + 3 patch nền cho giá trị ~0.25 < ngưỡng -> tầng xe gần như không có xe máy.
            pi_2d = pi_flat.view(Hp, Wp)
            pi_block = F.max_pool2d(pi_2d.view(1, 1, Hp, Wp), kernel_size=2, stride=2).view(Hb, Wb)
            pi_block_flat = pi_block.reshape(-1)

            # 2. Tổng số siêu khối 2x2 cần che (mỗi khối = 4 patches)
            n_b_mask = max(1, int(round(n_mask / 4.0)))

            # 3. Phân tầng siêu khối xe vs siêu khối nền (cùng ngưỡng thr_fg với mức patch)
            is_fg_block = pi_block_flat >= thr_fg
            fg_b_indices = torch.where(is_fg_block)[0]
            bg_b_indices = torch.where(~is_fg_block)[0]

            n_fg_b_total = int(fg_b_indices.numel())
            n_bg_b_total = int(bg_b_indices.numel())

            # 4. Phân bổ ngân sách khối: tôn trọng trần q_max và tỷ lệ phi
            max_fg_b = int(math.floor(q_max * n_fg_b_total))
            k_fg_b = min(int(round(phi * n_b_mask)), max_fg_b)
            k_fg_b = max(0, k_fg_b)

            k_bg_b = min(n_b_mask - k_fg_b, n_bg_b_total)
            if k_fg_b + k_bg_b < n_b_mask and n_fg_b_total > k_fg_b:
                extra = min(n_b_mask - (k_fg_b + k_bg_b), max_fg_b - k_fg_b)
                k_fg_b += max(0, extra)

            # 5. Lấy mẫu siêu khối: 1 lần duy nhất cho xe (có trọng số) và nền (ngẫu nhiên)
            block_mask_flat = torch.zeros(n_super_blocks, dtype=torch.bool, device=device)
            if k_fg_b > 0 and n_fg_b_total > 0:
                fg_w = pi_block_flat[fg_b_indices]
                chosen_fg_b = sample_without_replacement(fg_b_indices, k_fg_b, weights=fg_w, generator=generator)
                block_mask_flat[chosen_fg_b] = True

            if k_bg_b > 0 and n_bg_b_total > 0:
                chosen_bg_b = sample_without_replacement(bg_b_indices, k_bg_b, weights=None, generator=generator)
                block_mask_flat[chosen_bg_b] = True

            # 6. Bung khối siêu tốc thành lưới (Hp, Wp) bằng repeat_interleave
            mask_2d = block_mask_flat.view(Hb, Wb).repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

            # 7. Tinh chỉnh chính xác số lượng patch (nếu có chênh lệch 1-2 patch do phép chia 4):
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
                n_drop = cur_masked - n_mask
                nbr_count = torch.zeros((Hp, Wp), dtype=torch.int32, device=device)
                nbr_count[:-1, :] += mask_2d[1:, :].int()
                nbr_count[1:, :] += mask_2d[:-1, :].int()
                nbr_count[:, :-1] += mask_2d[:, 1:].int()
                nbr_count[:, 1:] += mask_2d[:, :-1].int()
                nbr_count = nbr_count * mask_2d.int()

                safe_candidates = (mask_2d) & (nbr_count >= 2)
                safe_idx = torch.where(safe_candidates.view(-1))[0]
                if safe_idx.numel() >= n_drop:
                    drop_chosen = sample_without_replacement(safe_idx, n_drop, weights=None, generator=generator)
                    mask_2d.view(-1)[drop_chosen] = False
                elif safe_idx.numel() > 0:
                    mask_2d.view(-1)[safe_idx] = False
                    rem_drop = n_mask - int(mask_2d.sum().item())
                    if rem_drop > 0:
                        all_m = torch.where(mask_2d.view(-1))[0]
                        drop_chosen = sample_without_replacement(all_m, rem_drop, weights=None, generator=generator)
                        mask_2d.view(-1)[drop_chosen] = False

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


def _gumbel_noise(shape, device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sinh nhiễu Gumbel(0, 1) chuẩn: g = -log(-log(U))."""
    u = torch.rand(shape, device=device, generator=generator).clamp(min=1e-6, max=1.0 - 1e-6)
    return -torch.log(-torch.log(u))


def _rank_desc(keys: torch.Tensor) -> torch.Tensor:
    """Hạng (0 = lớn nhất) của từng phần tử theo chiều cuối. Dùng cho Gumbel top-k với k thay đổi theo mẫu."""
    order = torch.argsort(keys, dim=-1, descending=True)
    return torch.argsort(order, dim=-1)


def batch_agm_sample(
    pi: torch.Tensor,
    ratio: float = 0.35,
    phi: float = 0.5,
    q_max: float = 0.6,
    mode: str = "block",
    grid_shape: Tuple[int, int] = (16, 28),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sinh mặt nạ AGM cho toàn bộ batch, VECTOR HÓA HOÀN TOÀN ở chế độ 'block' (không vòng lặp Python, không .item()).

    Thuật toán (Super-Block Grid 2x2 chuẩn SimMIM / iBOT):
      1. pi_block = MaxPool2d(pi, 2): siêu khối là "xe" nếu có ít nhất 1 patch xe (giữ được xe máy 1 patch).
      2. Ngưỡng xe động theo từng mẫu: 0.5, hạ về 0.35 khi ít hơn 12 patch đạt 0.5 (ban đêm / xe nhỏ).
      3. Ngân sách khối k_fg, k_bg là tensor (B,) - tuân thủ trần q_max và tỷ lệ phi như agm_sample.
      4. Lấy mẫu không hoàn lại có trọng số bằng Gumbel top-k với k thay đổi theo mẫu:
         key = log(w) + Gumbel; chọn các phần tử có hạng < k_b.
      5. Bung siêu khối bằng repeat_interleave -> mọi khối là 2x2 nguyên vẹn.
      6. Bù phần dư (n_mask - 4 * n_blocks) bằng Gumbel top-r, ưu tiên ô KỀ khối đã che
         (không tạo patch cô lập) -> mỗi mẫu có chính xác n_mask patch bị che.

    Các chế độ 'patch' / 'random' hoặc lưới lẻ dùng lại agm_sample theo từng mẫu (tương thích ngược).

    Args:
        pi: (B, N) hoặc (B, Hp, Wp) xác suất tiền cảnh; (N,) sẽ chuyển sang agm_sample.
        ratio, phi, q_max: như agm_sample.
        mode: 'block' | 'patch' | 'random'.
        grid_shape: (Hp, Wp), mặc định (16, 28).
    Returns:
        masks: (B, N) bool.
    """
    if pi.dim() == 1:
        return agm_sample(pi=pi, ratio=ratio, phi=phi, q_max=q_max, mode=mode,
                          grid_shape=grid_shape, generator=generator)

    B = pi.shape[0]
    Hp, Wp = grid_shape
    N = Hp * Wp
    device = pi.device

    if mode != "block" or Hp % 2 != 0 or Wp % 2 != 0 or pi[0].numel() != N:
        masks = [
            agm_sample(pi=pi[b].reshape(-1), ratio=ratio, phi=phi, q_max=q_max, mode=mode,
                       grid_shape=grid_shape, generator=generator)
            for b in range(B)
        ]
        return torch.stack(masks, dim=0)

    n_mask = int(round(ratio * N))
    if n_mask <= 0:
        return torch.zeros((B, N), dtype=torch.bool, device=device)

    pi_2d = pi.reshape(B, Hp, Wp).float().clamp(0.0, 1.0)
    Hb, Wb = Hp // 2, Wp // 2
    nb = Hb * Wb

    # 1-2. Mức xe của siêu khối (max-pool) và ngưỡng động theo mẫu
    pi_blk = F.max_pool2d(pi_2d.unsqueeze(1), kernel_size=2, stride=2).reshape(B, nb)
    n05 = (pi_2d >= 0.50).flatten(1).sum(dim=1)
    n035 = (pi_2d >= 0.35).flatten(1).sum(dim=1)
    thr = torch.where((n05 < 12) & (n035 >= 12),
                      torch.full((B,), 0.35, device=device),
                      torch.full((B,), 0.50, device=device))
    is_fg = pi_blk >= thr.unsqueeze(1)  # (B, nb)

    # 3. Ngân sách khối (tensor theo mẫu)
    n_b = max(1, n_mask // 4)  # 4 * n_b <= n_mask; phần dư được bù ở bước 6
    n_fg = is_fg.sum(dim=1)
    n_bg = nb - n_fg
    max_fg = torch.floor(q_max * n_fg.float()).long()
    k_fg = torch.minimum(torch.full_like(n_fg, int(round(phi * n_b))), max_fg).clamp(min=0)
    k_bg = torch.minimum(n_b - k_fg, n_bg).clamp(min=0)
    extra = torch.minimum((n_b - (k_fg + k_bg)).clamp(min=0), (max_fg - k_fg).clamp(min=0))
    k_fg = k_fg + extra

    # 4. Gumbel top-k có trọng số pi cho tầng xe, đồng nhất cho tầng nền
    neg_inf = torch.tensor(float("-inf"), device=device)
    key_fg = torch.where(is_fg, torch.log(pi_blk.clamp(min=1e-6)) + _gumbel_noise((B, nb), device, generator), neg_inf)
    key_bg = torch.where(~is_fg, _gumbel_noise((B, nb), device, generator), neg_inf)
    sel = (is_fg & (_rank_desc(key_fg) < k_fg.unsqueeze(1))) | ((~is_fg) & (_rank_desc(key_bg) < k_bg.unsqueeze(1)))

    # 5. Bung siêu khối 2x2
    mask = sel.view(B, Hb, Wb).repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)  # (B, Hp, Wp)

    # 6. Bù phần dư để mỗi mẫu có đúng n_mask patch, ưu tiên ô kề 4 hướng với vùng đã che
    deficit = n_mask - mask.flatten(1).sum(dim=1)  # (B,)
    if bool((deficit > 0).any()):
        m = mask.unsqueeze(1).float()
        cross = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]], device=device).view(1, 1, 3, 3)
        adj = (F.conv2d(m, cross, padding=1) > 0).squeeze(1) & (~mask)
        g = _gumbel_noise((B, Hp, Wp), device, generator)
        key_add = torch.where(adj, g + 100.0, torch.where(~mask, g, neg_inf)).flatten(1)
        add = (_rank_desc(key_add) < deficit.clamp(min=0).unsqueeze(1)).view(B, Hp, Wp) & (~mask)
        mask = mask | add

    return mask.reshape(B, N)

