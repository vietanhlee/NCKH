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

    # 1. Xác định tập các patch tĩnh ở cả 2 frame theo ngưỡng mặc định
    static_common = (pi < thr) & (pi2 < thr) & valid & valid2

    # 2. Loại bỏ các patch kề cận vùng xe của cả hai frame để không chép dính nửa chiếc xe
    fg_union = (pi >= 0.5) | (pi2 >= 0.5)
    fg_dilated = dilate_mask_2d(fg_union, radius=1)
    eligible_swap = static_common & (~fg_dilated)

    n_eligible = int(eligible_swap.sum().item())
    min_eligible_needed = max(16, int(0.12 * Hp * Wp))  # Tối thiểu 12% diện tích khung hình (~54 patches)

    # 2b. Cơ chế Ngưỡng Thích ứng Dự phòng (Adaptive Fallback Guarantee):
    # Khắc phục triệt để hiện tượng n_eligible == 0 khi pi bị nhiễu hoặc mật độ xe cao
    if n_eligible < min_eligible_needed:
        # Thử nghiệm fallback cấp 1: Giảm bán kính giãn nở về 0 (chỉ loại bỏ trực tiếp xe, không giãn biên)
        eligible_swap_fb1 = static_common & (~fg_union)
        if int(eligible_swap_fb1.sum().item()) >= min_eligible_needed:
            eligible_swap = eligible_swap_fb1
            n_eligible = int(eligible_swap.sum().item())
        else:
            # Fallback cấp 2: Nới lỏng ngưỡng tĩnh dựa trên phân vị động của ảnh
            # Tìm các patch có xác suất xe thấp nhất trong cả 2 khung hình
            combined_fg = torch.maximum(pi, pi2)
            # Ngưỡng thích ứng tại phân vị 35% thấp nhất
            flat_fg = combined_fg[valid & valid2].view(-1)
            if flat_fg.numel() > 0:
                k_val = max(1, min(int(flat_fg.numel() * 0.35), flat_fg.numel() - 1))
                adapt_thr = float(torch.kthvalue(flat_fg, k_val).values.item())
                adapt_thr = max(thr, min(adapt_thr, 0.45))
                # Chỉ coi là xe chắc chắn nếu pi >= 0.65 (tránh false positive mặt đường)
                fg_core = (pi >= 0.65) | (pi2 >= 0.65)
                eligible_swap_fb2 = (pi <= adapt_thr) & (pi2 <= adapt_thr) & valid & valid2 & (~fg_core)
                if int(eligible_swap_fb2.sum().item()) > 0:
                    eligible_swap = eligible_swap_fb2
                    n_eligible = int(eligible_swap.sum().item())

    if n_eligible == 0 or ratio <= 0.0:
        return x.clone(), torch.zeros((Hp, Wp), dtype=torch.bool, device=device)

    # 3. Xác định mặt nạ hoán đổi (liền mạch hoặc lấy mẫu con theo ratio)
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

    # 4. Phóng đại mask patch lên kích thước pixel (H, W)
    # Nếu continuous_bg, dùng bilinear trên trọng số mềm (1 - pi) kết hợp eligible_swap để biên cực mượt
    if continuous_bg:
        soft_patch_mask = eligible_swap.float() * (1.0 - pi.clamp(0.0, 1.0))
        pixel_mask = F.interpolate(
            soft_patch_mask.view(1, 1, Hp, Wp),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
    else:
        pixel_mask = F.interpolate(
            swap_mask_patch.float().view(1, 1, Hp, Wp),
            size=(H, W),
            mode="nearest",
        )  # (1, 1, H, W)

    # 5. Làm mềm biên với dải feather pixel bằng average pool
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

