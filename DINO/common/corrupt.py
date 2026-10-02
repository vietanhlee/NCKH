"""
=============================================================================
 Common Utility: Frame Corruption Suite (FCS)
 Bộ thử nghiệm độ bền khung hình ngoại cảnh (8 loại nhiễu x 5 mức độ)
 Phục vụ đánh giá tính bền bỉ tổng quát hóa (Robustness Evaluation) chuẩn Q1
=============================================================================
"""

import io
import math
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional, List, Union


class FrameCorruptionSuite:
    """
    Tập hợp các phép biến đổi mô phỏng hư hao chất lượng ảnh ngoại cảnh giao thông thực tế.
    """

    CORRUPTION_TYPES = [
        "gaussian_noise",
        "motion_blur",
        "defocus_blur",
        "jpeg_compression",
        "low_light",
        "fog",
        "rain",
        "glare"
    ]

    @staticmethod
    def apply_gaussian_noise(image: Image.Image, severity: int = 1) -> Image.Image:
        sigmas = [8.0, 15.0, 25.0, 40.0, 60.0]
        s = sigmas[max(0, min(severity - 1, 4))]
        arr = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, s, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    @staticmethod
    def apply_motion_blur(image: Image.Image, severity: int = 1) -> Image.Image:
        radii = [1, 2, 3, 5, 8]
        r = radii[max(0, min(severity - 1, 4))]
        # Mô phỏng motion blur theo phương ngang của luồng giao thông
        kernel = np.zeros((2 * r + 1, 2 * r + 1))
        kernel[r, :] = 1.0 / (2 * r + 1)
        # Fallback dùng Gaussian blur nhẹ theo bán kính
        return image.filter(ImageFilter.BoxBlur(r))

    @staticmethod
    def apply_defocus_blur(image: Image.Image, severity: int = 1) -> Image.Image:
        radii = [1.0, 2.0, 3.5, 5.0, 7.5]
        r = radii[max(0, min(severity - 1, 4))]
        return image.filter(ImageFilter.GaussianBlur(radius=r))

    @staticmethod
    def apply_jpeg_compression(image: Image.Image, severity: int = 1) -> Image.Image:
        qualities = [75, 50, 30, 15, 5]
        q = qualities[max(0, min(severity - 1, 4))]
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @staticmethod
    def apply_low_light(image: Image.Image, severity: int = 1) -> Image.Image:
        dark_factors = [0.75, 0.55, 0.40, 0.25, 0.12]
        factor = dark_factors[max(0, min(severity - 1, 4))]
        enhancer = ImageEnhance.Brightness(image)
        dark = enhancer.enhance(factor)
        # Thêm nhiễu hạt ban đêm
        arr = np.array(dark, dtype=np.float32)
        noise = np.random.normal(0, 10.0 * severity, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_fog(image: Image.Image, severity: int = 1) -> Image.Image:
        alphas = [0.15, 0.25, 0.40, 0.55, 0.70]
        alpha = alphas[max(0, min(severity - 1, 4))]
        arr = np.array(image, dtype=np.float32)
        fog = np.full_like(arr, 220.0)  # Lớp sương màu xám trắng
        blended = (1 - alpha) * arr + alpha * fog
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_rain(image: Image.Image, severity: int = 1) -> Image.Image:
        streak_counts = [50, 120, 250, 450, 800]
        num_streaks = streak_counts[max(0, min(severity - 1, 4))]
        arr = np.array(image).copy()
        H, W = arr.shape[:2]
        for _ in range(num_streaks):
            x = random.randint(0, W - 15)
            y = random.randint(0, H - 25)
            length = random.randint(10, 25)
            for k in range(length):
                if 0 <= y + k < H and 0 <= x + k // 3 < W:
                    arr[y + k, x + k // 3] = np.clip(arr[y + k, x + k // 3].astype(np.int32) + 60, 0, 255)
        return Image.fromarray(arr)

    @staticmethod
    def apply_glare(image: Image.Image, severity: int = 1) -> Image.Image:
        num_glares = [1, 2, 3, 5, 7]
        n = num_glares[max(0, min(severity - 1, 4))]
        arr = np.array(image, dtype=np.float32)
        H, W = arr.shape[:2]
        y_grid, x_grid = np.ogrid[:H, :W]

        for _ in range(n):
            cx = random.randint(int(W * 0.1), int(W * 0.9))
            cy = random.randint(int(H * 0.4), int(H * 0.9))
            radius = random.uniform(15, 45) * (0.8 + 0.3 * severity)
            dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
            intensity = np.exp(-dist_sq / (2 * (radius ** 2)))[..., np.newaxis]
            arr = arr + intensity * 230.0

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    @classmethod
    def corrupt(cls, image: Image.Image, corruption_type: str, severity: int = 1) -> Image.Image:
        fn_map = {
            "gaussian_noise": cls.apply_gaussian_noise,
            "motion_blur": cls.apply_motion_blur,
            "defocus_blur": cls.apply_defocus_blur,
            "jpeg_compression": cls.apply_jpeg_compression,
            "low_light": cls.apply_low_light,
            "fog": cls.apply_fog,
            "rain": cls.apply_rain,
            "glare": cls.apply_glare,
        }
        fn = fn_map.get(corruption_type)
        if fn:
            return fn(image, severity)
        return image

    @classmethod
    def apply_corruption(
        cls,
        image: Union[torch.Tensor, Image.Image, np.ndarray],
        corruption_type: str,
        severity: int = 1,
    ) -> Union[torch.Tensor, Image.Image]:
        """
        Giao diện linh hoạt nhận cả torch.Tensor (3, H, W) hoặc PIL.Image và trả về cùng kiểu.
        Hỗ trợ alias tên như 'rain_streaks' -> 'rain'.
        """
        # Ánh xạ alias tên
        alias_map = {
            "rain_streaks": "rain",
            "gaussian": "gaussian_noise",
            "jpeg": "jpeg_compression",
            "blur": "motion_blur",
        }
        ctype = alias_map.get(corruption_type, corruption_type)

        is_torch = isinstance(image, torch.Tensor)
        device = image.device if is_torch else None

        if is_torch:
            t = image.detach().cpu()
            if t.dim() == 4:
                t = t.squeeze(0)
            arr = (t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(arr)
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        corrupted_pil = cls.corrupt(pil_img, ctype, severity=severity)

        if is_torch:
            c_arr = np.array(corrupted_pil, dtype=np.float32) / 255.0
            out_tensor = torch.from_numpy(c_arr).permute(2, 0, 1).to(device)
            return out_tensor
        return corrupted_pil

