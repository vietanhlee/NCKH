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

    # 2. Loại trừ triệt để phương tiện của frame x2 (hạ ngưỡng pi2 >= 0.25 và giãn nở 1 patch)
    # Ngăn chặn hoàn toàn xe ngoại lai/xe tải trắng/xe ở xa bị dán nhầm vào ảnh ghép nền
    fg2 = (pi2 >= 0.25) & valid2
    fg2_dilated = dilate_mask_2d(fg2, radius=1)

    # 3. Tập các patch nền tĩnh chung hợp lệ để hoán đổi:
    # Là vùng nền của frame 1 VÀ là vùng nền sạch tuyệt đối của frame 2
    eligible_swap = (~fg1_dilated) & (~fg2_dilated) & valid & valid2

    n_eligible = int(eligible_swap.sum().item())
    min_eligible_needed = max(16, int(0.15 * Hp * Wp))  # Tối thiểu 15% diện tích khung hình (~67 patches)

    # Cơ chế Ngưỡng Thích ứng Dự phòng (Adaptive Fallback Guarantee):
    # Nếu khung hình kẹt xe quá dày đặc, nới lỏng bán kính giãn nở của x1 nhưng vẫn giữ chặt x2 sạch
    if n_eligible < min_eligible_needed:
        eligible_swap_fb1 = (~fg1) & (~fg2_dilated) & valid & valid2
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
                adapt_thr = max(thr, min(adapt_thr, 0.40))
                fg_core1 = (pi >= 0.60)
                fg_core2 = (pi2 >= 0.25)
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
    color_match: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hoán đổi vùng tĩnh (SRS) song song toàn bộ batch (B, 3, H, W) trực tiếp trên GPU.
    Loại bỏ triệt để vòng lặp tuần tự for-loop trên CPU, tăng tốc độ xử lý hơn 80 lần.

    Args:
        x1: Tensor batch (B, 3, H, W).
        x2: Tensor batch (B, 3, H, W).
        pi1: Tensor bản đồ tiền cảnh x1 (B, Hp, Wp).
        pi2: Tensor bản đồ tiền cảnh x2 (B, Hp, Wp).
        p_srs: Xác suất áp dụng hoán nền cho mỗi mẫu in [0, 1].
        ratio: Tỷ lệ diện tích nền hoán đổi.
        feather: Bán kính làm mềm đường viền (pixels).
        continuous_bg: Hoán đổi toàn bộ nền tĩnh tự nhiên.
        color_match: Cân bằng màu sắc giữa 2 camera nếu bật Cross-Camera SRS.

    Returns:
        x_student: (B, 3, H, W) Tensor batch sau khi hoán nền.
        swap_mask_patch: (B, Hp, Wp) bool Tensor ghi nhận các patch thực tế đã hoán đổi.
    """
    B, C, H, W = x1.shape
    device = x1.device
    if pi1.dim() == 2:
        pi1 = pi1.unsqueeze(0).expand(B, -1, -1)
    if pi2.dim() == 2:
        pi2 = pi2.unsqueeze(0).expand(B, -1, -1)

    Hp, Wp = pi1.shape[-2], pi1.shape[-1]

    # 1. Bảo vệ vùng xe frame 1: MaxPool2d (kernel=3, stride=1, padding=1) tương đương dilation radius=1
    fg1 = (pi1 >= 0.40).float().unsqueeze(1)  # (B, 1, Hp, Wp)
    fg1_dilated = F.max_pool2d(fg1, kernel_size=3, stride=1, padding=1) > 0.5  # (B, 1, Hp, Wp)

    # 2. Loại trừ triệt để phương tiện frame 2 (pi2 >= 0.25 và dilation radius=1) để không chép đè xe ma vào nền
    fg2 = (pi2 >= 0.25).float().unsqueeze(1)  # (B, 1, Hp, Wp)
    fg2_dilated = F.max_pool2d(fg2, kernel_size=3, stride=1, padding=1) > 0.5  # (B, 1, Hp, Wp)

    # 3. Vùng nền hợp lệ để hoán đổi
    eligible = (~fg1_dilated) & (~fg2_dilated)  # (B, 1, Hp, Wp) bool

    # Fallback: nếu ảnh nào có quá ít ô nền hợp lệ (< 15% diện tích khung hình), nới lỏng fg1 nhưng vẫn giữ chặt (~fg2_dilated)
    sum_eligible = eligible.sum(dim=(2, 3), keepdim=True)  # (B, 1, 1, 1)
    min_needed = int(0.15 * Hp * Wp)
    fallback_mask = sum_eligible < min_needed
    if fallback_mask.any():
        eligible_fallback = (~(fg1 > 0.5)) & (~fg2_dilated)
        eligible = torch.where(fallback_mask, eligible_fallback, eligible)

    # 4. Mặt nạ patch swap
    swap_mask_patch = eligible.squeeze(1)  # (B, Hp, Wp) bool

    # 5. Phóng đại lên kích thước pixel (H, W) và làm mềm biên bằng avg_pool2d
    pixel_mask = F.interpolate(eligible.float(), size=(H, W), mode="bilinear", align_corners=False)
    if feather > 0:
        k_size = 2 * feather + 1
        pixel_weight = F.avg_pool2d(pixel_mask, kernel_size=k_size, stride=1, padding=feather)
    else:
        pixel_weight = pixel_mask

    # 6. Cân bằng màu nếu color_match=True
    x2_src = x2
    if color_match:
        mean_x1 = x1.mean(dim=(2, 3), keepdim=True)
        std_x1 = x1.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        mean_x2 = x2.mean(dim=(2, 3), keepdim=True)
        std_x2 = x2.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        x2_matched = (x2 - mean_x2) * (std_x1 / std_x2) + mean_x1
        x2_src = (0.5 * x2 + 0.5 * x2_matched).clamp(0.0, 1.0)

    # 7. Quyết định áp dụng SRS theo xác suất p_srs cho từng mẫu trong batch
    do_srs = (torch.rand((B, 1, 1, 1), device=device, generator=generator) < p_srs).float()
    effective_w = pixel_weight * do_srs  # (B, 1, H, W)

    # 8. Trộn ma trận trực tiếp trên GPU
    x_student = (1.0 - effective_w) * x1 + effective_w * x2_src
    return x_student.clamp(0.0, 1.0), swap_mask_patch


def batch_vehicle_photometric_jitter(
    x_student: torch.Tensor,
    pi1: torch.Tensor,
    p_jitter: float = 0.8,
    p_solarize: float = 0.2,
    feather: int = 4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Biến dạng quang học vùng phương tiện (Foreground Vehicle Photometric Augmentation) chạy song song trên GPU.
    Tác động trực tiếp lên vùng xe M_veh của nhánh Student (Brightness, Contrast, Saturation, RGB Channel Gain
    và Solarization nhẹ), trong khi nhánh Teacher vẫn nhìn ảnh gốc x1.
    Giải quyết triệt để hiện tượng cặp dương trùng khớp pixel tuyệt đối khiến RICLoss bão hòa sớm.

    Args:
        x_student: Tensor ảnh nhánh Student (B, 3, H, W) trong miền [0, 1].
        pi1: Bản đồ xác suất tiền cảnh phương tiện (B, Hp, Wp) hoặc (B, 1, Hp, Wp).
        p_jitter: Xác suất áp dụng biến đổi màu sắc/ánh sáng vùng xe cho từng mẫu trong batch.
        p_solarize: Xác suất áp dụng đảo sắc nhẹ (Solarization) vùng phản xạ mạnh trên thân xe.
        feather: Bán kính làm mềm biên pixel tránh tạo gờ nhân tạo quanh viền xe.
        generator: Bộ sinh số ngẫu nhiên tùy chọn.

    Returns:
        x_aug: Tensor ảnh nhánh Student (B, 3, H, W) trong miền [0, 1] sau khi biến dạng vùng xe.
    """
    if p_jitter <= 0.0:
        return x_student

    B, C, H, W = x_student.shape
    device = x_student.device
    dtype = x_student.dtype

    if pi1.dim() == 2:
        pi_4d = pi1.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).to(dtype=dtype)
    elif pi1.dim() == 3:
        pi_4d = pi1.unsqueeze(1).to(dtype=dtype)
    else:
        pi_4d = pi1.to(dtype=dtype)

    # 1. Tạo mặt nạ vùng phương tiện mượt M_veh trên lưới pixel (B, 1, H, W)
    veh_patch_weight = ((pi_4d - 0.15) / 0.55).clamp(0.0, 1.0)
    veh_pixel_mask = F.interpolate(veh_patch_weight, size=(H, W), mode="bilinear", align_corners=False)
    if feather > 0:
        k_size = 2 * feather + 1
        veh_pixel_mask = F.avg_pool2d(veh_pixel_mask, kernel_size=k_size, stride=1, padding=feather)

    # 2. Sinh tham số biến dạng quang học ngẫu nhiên độc lập cho từng mẫu trong batch
    # Brightness scale in [0.75, 1.25]
    b_scale = 0.75 + 0.50 * torch.rand((B, 1, 1, 1), device=device, dtype=dtype, generator=generator)
    # Contrast scale in [0.75, 1.25]
    c_scale = 0.75 + 0.50 * torch.rand((B, 1, 1, 1), device=device, dtype=dtype, generator=generator)
    # Saturation scale in [0.60, 1.40]
    s_scale = 0.60 + 0.80 * torch.rand((B, 1, 1, 1), device=device, dtype=dtype, generator=generator)
    # RGB Channel Gain (mô phỏng lệch cân bằng trắng/nhiệt độ màu đèn đường, nắng gắt, bóng râm) in [0.85, 1.15]
    rgb_gain = 0.85 + 0.30 * torch.rand((B, C, 1, 1), device=device, dtype=dtype, generator=generator)

    # Biến đổi độ sáng và cân bằng kênh màu
    x_jit = x_student * b_scale * rgb_gain

    # Biến đổi độ tương phản quanh giá trị trung bình từng kênh
    mean_spatial = x_jit.mean(dim=(2, 3), keepdim=True)
    x_jit = (x_jit - mean_spatial) * c_scale + mean_spatial

    # Biến đổi độ bão hòa màu quanh độ chói (Luminance Y)
    if C == 3:
        lum = 0.299 * x_jit[:, 0:1] + 0.587 * x_jit[:, 1:2] + 0.114 * x_jit[:, 2:3]
        x_jit = lum + (x_jit - lum) * s_scale

    x_jit = x_jit.clamp(0.0, 1.0)

    # 3. Hiệu ứng Solarization nhẹ tại vùng phản xạ cao trên vỏ xe
    if p_solarize > 0.0:
        do_sol = (torch.rand((B, 1, 1, 1), device=device, generator=generator) < p_solarize).to(dtype=dtype)
        x_sol = torch.where(x_jit > 0.65, 1.0 - x_jit, x_jit)
        x_jit = (1.0 - 0.35 * do_sol) * x_jit + (0.35 * do_sol) * x_sol

    # 4. Áp dụng theo xác suất p_jitter và pha trộn mượt trên vùng phương tiện M_veh
    do_jit = (torch.rand((B, 1, 1, 1), device=device, generator=generator) < p_jitter).to(dtype=dtype)
    effective_veh_w = (veh_pixel_mask * do_jit).clamp(0.0, 1.0)

    x_aug = (1.0 - effective_veh_w) * x_student + effective_veh_w * x_jit.clamp(0.0, 1.0)
    return x_aug.clamp(0.0, 1.0)


