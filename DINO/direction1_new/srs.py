"""
=============================================================================
 Hướng G: SRS (Static-Region Swap)
 Hoán đổi các patch vùng tĩnh giữa các ngày khác nhau của cùng camera
 Buộc biểu diễn bất biến với ánh sáng/nền và dồn toàn bộ chú ý vào xe
=============================================================================
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def dilate_mask_2d(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """
    Giãn vùng mask 2D (Hp, Wp) với bán kính radius patch.
    """
    if radius <= 0:
        return mask
    kernel_size = 2 * radius + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=torch.float32)
    mask_4d = mask.float().view(1, 1, mask.shape[0], mask.shape[1])
    dilated = F.conv2d(mask_4d, kernel, padding=radius) > 0.0
    return dilated.view(mask.shape[0], mask.shape[1])


def static_region_swap(
    x: torch.Tensor,
    x2: torch.Tensor,
    pi: torch.Tensor,
    pi2: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    valid2: Optional[torch.Tensor] = None,
    ratio: float = 0.6,
    generator: Optional[torch.Generator] = None,
    thr: float = 0.2,
    feather: int = 4,
    patch_size: int = 16,
    continuous_bg: bool = False,
    color_match: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hoán đổi các khối patch vùng tĩnh từ x2 sang x (hỗ trợ cả patch-wise rời rạc và continuous background).

    Args:
        x: Tensor (3, H, W) Frame ngày d (mặc định H=256, W=448).
        x2: Tensor (3, H, W) Frame ngày d' (cùng hoặc khác camera).
        pi: Tensor (Hp, Wp) Bản đồ xác suất tiền cảnh frame x (16, 28).
        pi2: Tensor (Hp, Wp) Bản đồ xác suất tiền cảnh frame x2 (16, 28).
        valid: Tensor (Hp, Wp) bool
        valid2: Tensor (Hp, Wp) bool
        ratio: Tỷ lệ các ô tĩnh được chọn để hoán đổi in [0.3, 1.0].
        generator: Bộ sinh ngẫu nhiên.
        thr: Ngưỡng xác định vùng tĩnh (mặc định pi < 0.2).
        feather: Độ rộng dải làm mềm biên pixel (mặc định 4 px).
        patch_size: Kích thước patch (mặc định 16).
        continuous_bg: Nếu True, thay thế liền mạch toàn bộ vùng nền tĩnh (1 - pi đã giãn) thay vì bốc patch rời rạc.
        color_match: Nếu True, chuẩn hóa nhẹ mean/std của x2 theo x (hữu ích cho Cross-Camera SRS).

    Returns:
        x_tilde: (3, H, W) Ảnh sau khi hoán đổi nền.
        swap_mask_patch: (Hp, Wp) bool, các patch thực tế bị hoán đổi.
    """
    device = x.device
    Hp, Wp = pi.shape
    C, H, W = x.shape

    if valid is None:
        valid = torch.ones_like(pi, dtype=torch.bool)
    if valid2 is None:
        valid2 = torch.ones_like(pi2, dtype=torch.bool)

    # 1. Bảo vệ toàn vẹn các phương tiện giao thông của frame x1:
    # Nhận diện vùng xe của frame 1 (pi >= 0.40) và giãn nở 1 patch nhẹ để bảo vệ đường viền xe
    fg1 = (pi >= 0.40) & valid
    fg1_dilated = dilate_mask_2d(fg1, radius=1)

    # 2. Loại trừ các phương tiện của frame x2 (pi2 >= 0.45) để không chép đè xe "ma" vào nền mới
    fg2 = (pi2 >= 0.45) & valid2

    # 3. Tập các patch nền tĩnh chung hợp lệ để hoán đổi:
    # Là vùng nền của frame 1 VÀ là vùng nền của frame 2
    eligible_swap = (~fg1_dilated) & (~fg2) & valid & valid2

    n_eligible = int(eligible_swap.sum().item())
    min_eligible_needed = max(16, int(0.15 * Hp * Wp))  # Tối thiểu 15% diện tích khung hình (~67 patches)

    # Cơ chế Ngưỡng Thích ứng Dự phòng (Adaptive Fallback Guarantee):
    # Nếu khung hình kẹt xe quá dày đặc, nới lỏng bán kính giãn nở về 0 để bảo đảm diện tích swap
    if n_eligible < min_eligible_needed:
        eligible_swap_fb1 = (~fg1) & (~fg2) & valid & valid2
        if int(eligible_swap_fb1.sum().item()) >= min_eligible_needed:
            eligible_swap = eligible_swap_fb1
            n_eligible = int(eligible_swap.sum().item())
        else:
            # Fallback cấp 2: Dùng phân vị động để tìm 40% diện tích có xác suất xe thấp nhất
            combined_fg = torch.maximum(pi, pi2)
            flat_fg = combined_fg[valid & valid2].view(-1)
            if flat_fg.numel() > 0:
                k_val = max(1, min(int(flat_fg.numel() * 0.40), flat_fg.numel() - 1))
                adapt_thr = float(torch.kthvalue(flat_fg, k_val).values.item())
                adapt_thr = max(thr, min(adapt_thr, 0.55))
                fg_core1 = (pi >= 0.65)
                fg_core2 = (pi2 >= 0.65)
                eligible_swap_fb2 = (pi <= adapt_thr) & (pi2 <= adapt_thr) & valid & valid2 & (~fg_core1) & (~fg_core2)
                if int(eligible_swap_fb2.sum().item()) > 0:
                    eligible_swap = eligible_swap_fb2
                    n_eligible = int(eligible_swap.sum().item())

    if n_eligible == 0 or ratio <= 0.0:
        return x.clone(), torch.zeros((Hp, Wp), dtype=torch.bool, device=device)

    # 4. Xác định mặt nạ hoán đổi (liền mạch hoặc lấy mẫu con theo ratio)
    if continuous_bg or ratio >= 0.99:
        swap_mask_patch = eligible_swap.clone()
    else:
        k_swap = max(1, int(round(ratio * n_eligible)))
        eligible_indices = torch.where(eligible_swap.view(-1))[0]
        perm = torch.randperm(eligible_indices.numel(), generator=generator, device=device)
        chosen_indices = eligible_indices[perm[:k_swap]]

        swap_mask_patch = torch.zeros(Hp * Wp, dtype=torch.bool, device=device)
        swap_mask_patch[chosen_indices] = True
        swap_mask_patch = swap_mask_patch.view(Hp, Wp)

    # 5. Phóng đại mask patch lên kích thước pixel (H, W)
    # Phóng đại trực tiếp swap_mask_patch với bilinear interpolation để đạt trọng số đầy đủ w in [0, 1]
    pixel_mask = F.interpolate(
        swap_mask_patch.float().view(1, 1, Hp, Wp),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )

    # 6. Làm mềm biên với dải feather pixel bằng average pool để ghép tự nhiên
    if feather > 0:
        k_size = 2 * feather + 1
        pixel_weight = F.avg_pool2d(pixel_mask, kernel_size=k_size, stride=1, padding=feather)
    else:
        pixel_weight = pixel_mask

    if x.dim() == 3:
        pixel_weight = pixel_weight.squeeze(0)  # (1, H, W)

    x2_src = x2
    if color_match:
        # Cân bằng nhẹ độ sáng / tương phản của x2 theo x để ghép khác camera tự nhiên
        mean_x = x.mean(dim=(1, 2), keepdim=True)
        std_x = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-4)
        mean_x2 = x2.mean(dim=(1, 2), keepdim=True)
        std_x2 = x2.std(dim=(1, 2), keepdim=True).clamp(min=1e-4)
        x2_matched = (x2 - mean_x2) * (std_x / std_x2) + mean_x
        x2_src = (0.5 * x2 + 0.5 * x2_matched).clamp(0.0, 1.0)

    # 6. Ghép tuyến tính: x_tilde = (1 - w) * x + w * x2_src
    x_tilde = (1.0 - pixel_weight) * x + pixel_weight * x2_src
    return x_tilde.clamp(0.0, 1.0), swap_mask_patch


def batch_static_region_swap(
    x1: torch.Tensor,
    x2: torch.Tensor,
    pi1: torch.Tensor,
    pi2: torch.Tensor,
    p_srs: float = 0.5,
    ratio: float = 0.6,
    feather: int = 4,
    continuous_bg: bool = True,
    color_match: bool = True,
    generator: Optional[torch.Generator] = None,
    thr: float = 0.2,
    max_lum_log_ratio: float = 0.47,
    exclude_top_rows: int = 1,
    exclude_bottom_rows: int = 0,
    min_swap_frac: float = 0.10,
    return_stats: bool = False,
):
    """
    Hoán đổi vùng tĩnh (SRS) song song toàn bộ batch (B, 3, H, W) trực tiếp trên GPU.

    Quy tắc nghiệp vụ (True Background Swap có kiểm soát):
      1. Bảo vệ xe frame 1: patch có MaxPool3x3(pi1) >= thr (xe giãn 1 patch) không bao giờ bị thay.
         Ngưỡng thấp (mặc định 0.2) để xe máy có pi ~0.3 không bị thay bằng mặt đường ở view student
         trong khi teacher vẫn thấy xe (gây mục tiêu học mâu thuẫn).
      2. Loại xe frame 2: patch có pi2 >= thr + 0.1 không được chép sang (tránh "xe ma").
      3. Loại banner timestamp: exclude_top_rows / exclude_bottom_rows hàng patch không bao giờ bị swap.
         Banner là chữ tương phản cao; nếu đổi theo x2, nó trở thành tín hiệu lộ "đã swap".
      4. Gate điều kiện sáng: chỉ swap khi |log(L1 / L2)| <= max_lum_log_ratio (mặc định log 1.6 ~ 0.47).
         Tránh ảnh ghép ngày - đêm phi vật lý mà student không bao giờ gặp lúc test.
      5. Khớp màu (color_match, mặc định bật cả với cặp cùng camera): chuẩn hóa mean/std theo kênh của x2
         về x1, ước lượng CHỈ trên vùng nền chung được swap -> làm mờ biên ghép.
      6. Mẫu có diện tích hợp lệ < min_swap_frac sẽ không swap (tránh vá vụn gây artifact).

    Args:
        x1, x2: (B, 3, H, W) trong [0, 1].
        pi1, pi2: (B, Hp, Wp) bản đồ tiền cảnh.
        p_srs: Xác suất áp dụng hoán nền cho mỗi mẫu.
        ratio, continuous_bg: Giữ cho tương thích chữ ký (batch luôn swap liền mạch toàn vùng nền hợp lệ).
        feather: Bán kính làm mềm đường viền (pixels).
        return_stats: Nếu True trả thêm dict thống kê (applied, swap_frac, lum_ok) theo mẫu.

    Returns:
        x_student: (B, 3, H, W)
        swap_mask_patch: (B, Hp, Wp) bool - patch THỰC SỰ bị swap (False ở mẫu không áp dụng).
        stats (tùy chọn): dict các tensor (B,).
    """
    B, C, H, W = x1.shape
    device = x1.device
    if pi1.dim() == 2:
        pi1 = pi1.unsqueeze(0).expand(B, -1, -1)
    if pi2.dim() == 2:
        pi2 = pi2.unsqueeze(0).expand(B, -1, -1)
    pi1 = pi1.float()
    pi2 = pi2.float()
    Hp, Wp = pi1.shape[-2], pi1.shape[-1]

    # 1. Bảo vệ xe frame 1 (dilation radius 1 bằng max-pool trên pi)
    pi1_dil = F.max_pool2d(pi1.unsqueeze(1), kernel_size=3, stride=1, padding=1)  # (B, 1, Hp, Wp)
    protect1 = pi1_dil >= thr
    # 2. Loại xe frame 2
    fg2 = (pi2 >= (thr + 0.1)).unsqueeze(1)
    eligible = (~protect1) & (~fg2)  # (B, 1, Hp, Wp)

    # 3. Loại các hàng banner
    if exclude_top_rows > 0:
        eligible[:, :, :exclude_top_rows, :] = False
    if exclude_bottom_rows > 0:
        eligible[:, :, Hp - exclude_bottom_rows:, :] = False

    swap_frac = eligible.float().mean(dim=(1, 2, 3))  # (B,)

    # 4. Gate điều kiện sáng (luminance Rec.601)
    lum_w = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=x1.dtype).view(1, 3, 1, 1)
    L1 = (x1 * lum_w).sum(dim=1).mean(dim=(1, 2)).clamp(min=1e-3)
    L2 = (x2 * lum_w).sum(dim=1).mean(dim=(1, 2)).clamp(min=1e-3)
    lum_ok = torch.abs(torch.log(L1 / L2)) <= max_lum_log_ratio

    # 5. Quyết định áp dụng theo mẫu
    rand_ok = torch.rand((B,), device=device, generator=generator) < p_srs
    applied = rand_ok & lum_ok & (swap_frac >= min_swap_frac)
    swap_mask_patch = eligible.squeeze(1) & applied.view(B, 1, 1)

    # 6. Trọng số pixel: phóng đại + feather (KHÔNG làm mềm vượt vào vùng bảo vệ quá feather px)
    pixel_mask = F.interpolate(eligible.float(), size=(H, W), mode="nearest")
    if feather > 0:
        k_size = 2 * feather + 1
        pixel_weight = F.avg_pool2d(pixel_mask, kernel_size=k_size, stride=1, padding=feather,
                                    count_include_pad=False)
    else:
        pixel_weight = pixel_mask

    # 7. Khớp màu theo kênh trên vùng nền chung
    x2_src = x2
    if color_match:
        wsum = pixel_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        m1 = (x1 * pixel_mask).sum(dim=(2, 3), keepdim=True) / wsum
        m2 = (x2 * pixel_mask).sum(dim=(2, 3), keepdim=True) / wsum
        s1 = (((x1 - m1) ** 2 * pixel_mask).sum(dim=(2, 3), keepdim=True) / wsum).sqrt().clamp(min=1e-3)
        s2 = (((x2 - m2) ** 2 * pixel_mask).sum(dim=(2, 3), keepdim=True) / wsum).sqrt().clamp(min=1e-3)
        x2_src = ((x2 - m2) * (s1 / s2) + m1).clamp(0.0, 1.0)

    # 8. Trộn
    effective_w = pixel_weight * applied.view(B, 1, 1, 1).to(pixel_weight.dtype)
    x_student = ((1.0 - effective_w) * x1 + effective_w * x2_src).clamp(0.0, 1.0)

    if return_stats:
        stats = {
            "applied": applied,
            "applied_rate": float(applied.float().mean().item()),
            "lum_ok": lum_ok,
            "lum_ok_all": bool(lum_ok.all().item()),
            "swap_frac": swap_frac * applied.float(),
            "swap_frac_mean": float((swap_frac * applied.float()).mean().item()),
        }
        return x_student, swap_mask_patch, stats
    return x_student, swap_mask_patch


