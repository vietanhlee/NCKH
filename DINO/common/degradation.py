"""
=============================================================================
 Common Utility: Background Degradation Benchmark (BDB)
 Bộ công cụ mô phỏng suy thoái chất lượng ảnh nền có kiểm soát
 6 loại nhiễu x 5 mức độ (Severity 1 -> 5) theo chuẩn nghiên cứu Q1
 Phục vụ trả lời câu hỏi phản biện: "Nếu background xấu đi thì mô hình ra sao?"
=============================================================================
"""

import os
import io
import math
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional, Union, List, Dict


class BackgroundDegradationBenchmark:
    """
    Bộ suy thoái ảnh nền BDB áp dụng tại thời điểm kiểm thử (Test-time) hoặc huấn luyện (Robust training).
    Bao gồm 6 loại nhiễu mô phỏng lỗi thực tế của hệ thống camera đô thị.
    """

    NOISE_TYPES = [
        "time_shift",          # Sai slot giờ (lệch chiếu sáng, bóng đổ)
        "camera_shift",        # Camera rung lắc hoặc bị chỉnh góc xoay
        "ghost_injection",     # Xe kẹt bị median nuốt thành bóng ma trong nền
        "optical_change",      # Mưa, nắng gắt, thay đổi phơi sáng
        "noise_compression",   # Cảm biến nhiễu hạt, nén JPEG truyền dẫn
        "cross_camera_swap"    # Thay bằng background của camera khác
    ]

    @staticmethod
    def apply_camera_shift(
        image: Image.Image,
        severity: int = 1
    ) -> Image.Image:
        """
        Mô phỏng camera bị rung hoặc bị công nhân chỉnh góc: dịch chuyển và xoay nhẹ.
        Severity 1 -> 5 tương ứng dịch: 2, 4, 8, 16, 32 pixel; xoay 0.5, 1.0, 1.5, 2.0, 3.0 độ.
        """
        shifts = [2, 4, 8, 16, 32]
        angles = [0.5, 1.0, 1.5, 2.0, 3.0]
        s_idx = max(0, min(severity - 1, 4))
        d_px = shifts[s_idx]
        deg = angles[s_idx] * (1 if random.random() > 0.5 else -1)

        # Xoay và dịch
        rot_img = image.rotate(deg, resample=Image.BILINEAR)
        dx = d_px * (1 if random.random() > 0.5 else -1)
        dy = d_px * (1 if random.random() > 0.5 else -1)

        # Áp dịch chuyển (affine translation)
        trans_matrix = (1, 0, dx, 0, 1, dy)
        shifted = rot_img.transform(image.size, Image.AFFINE, trans_matrix, resample=Image.BILINEAR)
        return shifted

    @staticmethod
    def apply_optical_change(
        image: Image.Image,
        severity: int = 1
    ) -> Image.Image:
        """
        Mô phỏng thay đổi quang học: độ phơi sáng, tương phản và gamma.
        Severity 1 -> 5: gamma [0.9, 0.8, 0.7, 0.6, 0.5] hoặc sáng ±10% -> ±40%.
        """
        factors = [0.10, 0.20, 0.30, 0.40, 0.50]
        s_idx = max(0, min(severity - 1, 4))
        delta = factors[s_idx]

        # Thay đổi độ sáng
        brightness_factor = 1.0 + (delta if random.random() > 0.5 else -delta)
        enhancer = ImageEnhance.Brightness(image)
        img_bright = enhancer.enhance(brightness_factor)

        # Thay đổi tương phản
        contrast_factor = 1.0 + (delta * 0.5 if random.random() > 0.5 else -delta * 0.5)
        enhancer_c = ImageEnhance.Contrast(img_bright)
        return enhancer_c.enhance(contrast_factor)

    @staticmethod
    def apply_noise_compression(
        image: Image.Image,
        severity: int = 1
    ) -> Image.Image:
        """
        Mô phỏng cảm biến chất lượng thấp và nén JPEG mạnh qua mạng 4G/CCTV.
        Severity 1 -> 5: JPEG quality từ 85 -> 15; Gaussian noise sigma từ 5 -> 35.
        """
        qualities = [80, 60, 40, 25, 10]
        sigmas = [5.0, 10.0, 18.0, 26.0, 35.0]
        s_idx = max(0, min(severity - 1, 4))

        arr = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, sigmas[s_idx], arr.shape)
        arr_noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img_noisy = Image.fromarray(arr_noisy)

        # Nén JPEG vào bộ đệm RAM
        buffer = io.BytesIO()
        img_noisy.save(buffer, format="JPEG", quality=qualities[s_idx])
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    @staticmethod
    def apply_ghost_injection(
        background: Image.Image,
        severity: int = 1,
        road_mask: Optional[np.ndarray] = None,
        vehicle_crops: Optional[List[Image.Image]] = None
    ) -> Image.Image:
        """
        Mô phỏng lỗi chí mạng của Median Filter: dán xe bóng ma (ghost vehicles) lên lòng đường.
        Severity 1 -> 5: Che phủ diện tích mặt đường lần lượt 5%, 10%, 20%, 30%, 50%.
        """
        bg_copy = background.copy()
        W, H = bg_copy.size
        coverage_targets = [0.05, 0.10, 0.20, 0.30, 0.50]
        s_idx = max(0, min(severity - 1, 4))
        target_cov = coverage_targets[s_idx]

        if road_mask is None:
            # Mặc định nửa dưới bức ảnh là lòng đường
            road_mask = np.zeros((H, W), dtype=bool)
            road_mask[int(H * 0.35):, :] = True

        road_area = np.sum(road_mask)
        target_ghost_pixels = road_area * target_cov

        cur_ghost_pixels = 0
        road_coords = np.argwhere(road_mask)  # (N, 2): [y, x]
        if len(road_coords) == 0:
            return bg_copy

        # Số lần dán tối đa để đạt diện tích mục tiêu
        max_attempts = 30
        attempt = 0
        while cur_ghost_pixels < target_ghost_pixels and attempt < max_attempts:
            attempt += 1
            # Chọn tọa độ ngẫu nhiên trong lòng đường
            idx = random.randint(0, len(road_coords) - 1)
            cy, cx = road_coords[idx]

            # Kích thước box xe giả lập (phù hợp phối cảnh camera: gần to, xa nhỏ)
            perspective_scale = 0.5 + 0.8 * (cy / H)
            veh_w = int(max(20, 45 * perspective_scale))
            veh_h = int(max(15, 35 * perspective_scale))

            x1 = max(0, cx - veh_w // 2)
            y1 = max(0, cy - veh_h // 2)
            x2 = min(W, x1 + veh_w)
            y2 = min(H, y1 + veh_h)

            if vehicle_crops and len(vehicle_crops) > 0:
                crop = random.choice(vehicle_crops).resize((x2 - x1, y2 - y1), Image.BILINEAR)
                # Dán với độ trong suốt bán phần (ghost vehicle mờ ảo do median)
                alpha = random.uniform(0.6, 0.9)
                bg_roi = np.array(bg_copy.crop((x1, y1, x2, y2)), dtype=np.float32)
                crop_arr = np.array(crop, dtype=np.float32)
                blended = (alpha * crop_arr + (1 - alpha) * bg_roi).astype(np.uint8)
                bg_copy.paste(Image.fromarray(blended), (x1, y1))
            else:
                # Tạo một khối bóng ma xe có viền và kết cấu màu ngẫu nhiên (xe máy/ô tô)
                veh_color = (random.randint(40, 220), random.randint(40, 220), random.randint(40, 220))
                ghost_block = Image.new("RGB", (x2 - x1, y2 - y1), veh_color)
                # Làm mờ biên nhẹ
                ghost_block = ghost_block.filter(ImageFilter.GaussianBlur(radius=1.5))
                bg_copy.paste(ghost_block, (x1, y1))

            cur_ghost_pixels += (x2 - x1) * (y2 - y1)

        return bg_copy

    @classmethod
    def apply_bdb(
        cls,
        background: Image.Image,
        noise_type: str,
        severity: int = 1,
        road_mask: Optional[np.ndarray] = None,
        other_camera_bg: Optional[Image.Image] = None,
        vehicle_crops: Optional[List[Image.Image]] = None
    ) -> Image.Image:
        """
        Áp dụng kiểm thử suy thoái BDB tổng hợp theo loại nhiễu và mức độ.

        Args:
            background: Ảnh nền chuẩn PIL.Image.
            noise_type: Một trong các giá trị trong NOISE_TYPES.
            severity: Mức độ nhiễu 1 -> 5.
            road_mask: Mặt nạ lòng đường bool numpy (H, W).
            other_camera_bg: Ảnh nền của camera khác (cho cross_camera_swap).
            vehicle_crops: Danh sách crop xe để dán ghost chân thực.

        Returns:
            degraded_background: Ảnh nền đã qua xử lý suy thoái có kiểm soát.
        """
        if noise_type == "camera_shift":
            return cls.apply_camera_shift(background, severity)
        elif noise_type == "optical_change":
            return cls.apply_optical_change(background, severity)
        elif noise_type == "noise_compression":
            return cls.apply_noise_compression(background, severity)
        elif noise_type == "ghost_injection":
            return cls.apply_ghost_injection(background, severity, road_mask, vehicle_crops)
        elif noise_type == "cross_camera_swap":
            if other_camera_bg is not None:
                return other_camera_bg.resize(background.size, Image.BILINEAR)
            else:
                # Đảo ngược màu hoặc dịch lớn
                return cls.apply_camera_shift(background, severity=5)
        elif noise_type == "time_shift":
            # Thay đổi ánh sáng mô phỏng slot giờ khác (ngày vs đêm)
            if severity >= 4:
                # Đảo ngày <-> đêm: giảm độ sáng mạnh
                enhancer = ImageEnhance.Brightness(background)
                return enhancer.enhance(0.25)
            else:
                return cls.apply_optical_change(background, severity)
        else:
            return background

    @classmethod
    def apply_degradation(
        cls,
        background: Union[torch.Tensor, Image.Image, np.ndarray],
        noise_type: str,
        severity: int = 1,
        road_mask: Optional[np.ndarray] = None,
    ) -> Union[torch.Tensor, Image.Image]:
        """
        Giao diện linh hoạt nhận cả torch.Tensor (3, H, W) hoặc PIL.Image và trả về cùng kiểu.
        """
        is_torch = isinstance(background, torch.Tensor)
        device = background.device if is_torch else None

        if is_torch:
            t = background.detach().cpu()
            if t.dim() == 4:
                t = t.squeeze(0)
            arr = (t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(arr)
        elif isinstance(background, np.ndarray):
            pil_img = Image.fromarray(background)
        else:
            pil_img = background

        degraded_pil = cls.apply_bdb(pil_img, noise_type, severity=severity, road_mask=road_mask)

        if is_torch:
            deg_arr = np.array(degraded_pil, dtype=np.float32) / 255.0
            out_tensor = torch.from_numpy(deg_arr).permute(2, 0, 1).to(device)
            return out_tensor
        return degraded_pil

