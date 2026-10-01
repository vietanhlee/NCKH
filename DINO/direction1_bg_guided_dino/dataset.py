"""
=============================================================================
 Hướng 1: BG-Guided DINO — Dataset & Data Augmentation
 Foreground-Aware Masking (FAM) và Multi-Crop cho Self-Supervised Learning
=============================================================================
"""

import os
import sys
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Nạp module tiện ích dùng chung
dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dino_root not in sys.path:
    sys.path.insert(0, dino_root)
from common.matcher import TrafficPairMatcher
from common.subtraction import BackgroundSubtractor


class GaussianBlur:
    """Áp dụng Gaussian Blur ngẫu nhiên với xác suất p."""
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class Solarization:
    """Áp dụng Solarization ngẫu nhiên với xác suất p."""
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class MultiCropBGGuidedAugmentation:
    """
    Chiến lược tăng cường Multi-Crop kết hợp Foreground-Aware Masking:
      - 2 Global views (kích thước lớn 224x224): Dành cho Teacher & Student.
      - N Local views (kích thước nhỏ 96x96): Tập trung chi tiết fine-grained cho Student.
      - Sinh ra mặt nạ nhị phân theo patch (Patch Mask) ưu tiên che vùng phương tiện.
    """

    def __init__(
        self,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 4,
        size_global: int = 224,
        size_local: int = 96,
        patch_size: int = 16,
        mask_ratio: float = 0.5,
        alpha_fg: float = 0.75,
    ):
        # Tự động căn chỉnh kích thước crop luôn là bội số của patch_size
        if size_global % patch_size != 0:
            size_global = max(patch_size, round(size_global / patch_size) * patch_size)
        if size_local % patch_size != 0:
            size_local = max(patch_size, round(size_local / patch_size) * patch_size)

        self.local_crops_number = local_crops_number
        self.size_global = size_global
        self.size_local = size_local
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.alpha_fg = alpha_fg

        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 1. Global View 1 (Gaussian Blur, không Solarize)
        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(size_global, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            transforms.ToTensor(),
            normalize,
        ])

        # 2. Global View 2 (Light Blur + Solarization)
        self.global_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(size_global, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        # 3. Local Views
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(size_local, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

    def generate_fg_mask(self, patch_weights: np.ndarray) -> torch.Tensor:
        """
        Sinh mặt nạ nhị phân dạng boolean tensor (num_patches,) dựa trên phân bố xác suất foreground.
        True = patch bị che (masked), False = patch giữ lại (unmasked).
        """
        flat_weights = patch_weights.flatten()
        n_patches = len(flat_weights)
        n_mask = int(n_patches * self.mask_ratio)

        # Lấy mẫu các chỉ số patch bị che dựa trên xác suất p = flat_weights
        flat_probs = flat_weights / (flat_weights.sum() + 1e-12)
        masked_indices = np.random.choice(n_patches, size=n_mask, replace=False, p=flat_probs)

        mask = np.zeros(n_patches, dtype=bool)
        mask[masked_indices] = True
        return torch.from_numpy(mask)


class BGGuidedDINODataset(Dataset):
    """
    Dataset nạp đồng thời ảnh Origin và Background, trích xuất ma trận xác suất patch
    và chuẩn bị Multi-Crop views cho quá trình huấn luyện tự giám sát DINOv3.
    """

    def __init__(
        self,
        bg_dir: str,
        origin_dir: str,
        match_strategy: str = "route_hourly",
        patch_size: int = 16,
        size_global: int = 224,
        size_local: int = 96,
        local_crops_number: int = 4,
        mask_ratio: float = 0.5,
        alpha_fg: float = 0.75,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.matcher = TrafficPairMatcher(
            bg_dir=bg_dir,
            origin_dir=origin_dir,
            match_strategy=match_strategy,
        )
        self.pairs = self.matcher.discover_pairs(max_pairs=max_samples)
        if not self.pairs:
            raise RuntimeError(
                f"Không thể tìm thấy cặp ảnh nào giữa origin_dir='{origin_dir}' và bg_dir='{bg_dir}'. "
                "Vui lòng kiểm tra lại đường dẫn và chiến lược matching."
            )

        # Tự động chuẩn hóa kích thước crop theo patch_size
        if size_global % patch_size != 0:
            size_global = max(patch_size, round(size_global / patch_size) * patch_size)
        if size_local % patch_size != 0:
            size_local = max(patch_size, round(size_local / patch_size) * patch_size)

        self.subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)
        self.patch_size = patch_size
        self.size_global = size_global
        self.size_local = size_local
        self.alpha_fg = alpha_fg

        self.augmentor = MultiCropBGGuidedAugmentation(
            size_global=size_global,
            size_local=size_local,
            patch_size=patch_size,
            local_crops_number=local_crops_number,
            mask_ratio=mask_ratio,
            alpha_fg=alpha_fg,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Union[List[torch.Tensor], torch.Tensor, str]]:
        item = self.pairs[idx]
        try:
            origin_img = Image.open(item["origin_path"]).convert("RGB")
            bg_img = Image.open(item["bg_path"]).convert("RGB")
        except Exception as e:
            # Fallback nếu gặp file lỗi
            origin_img = Image.new("RGB", (self.size_global, self.size_global), (128, 128, 128))
            bg_img = Image.new("RGB", (self.size_global, self.size_global), (128, 128, 128))

        # 1. Trừ nền trích xuất delta map
        delta_norm, _ = self.subtractor.compute_delta(origin_img, bg_img)

        # 2. Tính ma trận trọng số xác suất patch cho global crop (H_patches, W_patches)
        patch_probs = self.subtractor.compute_patch_mask_weights(
            delta_norm=delta_norm,
            img_size=self.size_global,
            patch_size=self.patch_size,
            alpha=self.alpha_fg,
        )

        # 3. Tạo Multi-Crops
        crops = []
        crops.append(self.augmentor.global_transform_1(origin_img))
        crops.append(self.augmentor.global_transform_2(origin_img))
        for _ in range(self.augmentor.local_crops_number):
            crops.append(self.augmentor.local_transform(origin_img))

        # 4. Sinh foreground mask cho Student Global View 1
        fg_mask = self.augmentor.generate_fg_mask(patch_probs)

        return {
            "crops": crops,                       # List gồm 2 Global + N Local tensors
            "fg_mask": fg_mask,                   # Boolean tensor (N_patches,)
            "origin_path": item["origin_path"],
            "route_id": item["route_id"],
        }
