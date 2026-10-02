"""
=============================================================================
 Hướng 1: BG-Guided DINO — Dataset & Data Augmentation
 Foreground-Aware Masking (FAM) chuẩn xác suất theo Softmax nhiệt độ T,
 Chuẩn hóa Rank-based chống lóa và Cổng tin cậy r_i theo chuẩn Q1
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
from common.reliability import estimate_background_reliability


class GaussianBlur:
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
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class MultiCropBGGuidedAugmentation:
    """
    Chiến lược tăng cường Multi-Crop kết hợp Foreground-Aware Masking chuẩn toán học:
    - 2 Global views (224x224): Dành cho Teacher & Student.
    - N Local views (96x96): Tập trung chi tiết fine-grained cho Student.
    - Chuẩn hóa Softmax nhiệt độ T kết hợp Rank Normalization chống nhiễu lóa đèn ban đêm.
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
        alpha_max: float = 0.75,
        temperature: float = 0.20,
    ):
        if size_global % patch_size != 0:
            size_global = max(patch_size, round(size_global / patch_size) * patch_size)
        if size_local % patch_size != 0:
            size_local = max(patch_size, round(size_local / patch_size) * patch_size)

        self.local_crops_number = local_crops_number
        self.size_global = size_global
        self.size_local = size_local
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.alpha_max = alpha_max
        self.temperature = temperature

        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(size_global, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            transforms.ToTensor(),
            normalize,
        ])

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

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(size_local, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

    def generate_fg_mask(
        self,
        patch_weights: np.ndarray,
        reliability: float = 1.0
    ) -> torch.Tensor:
        """
        Sinh mặt nạ nhị phân boolean (N_patches,) theo đặc tả I.2:
        P_mask(p) = alpha_i * (exp(w_tilde_p / T) / sum exp(w_tilde_q / T)) + (1 - alpha_i) / N
        với w_tilde là thứ hạng (rank) chuẩn hóa trong [0, 1] và alpha_i = alpha_max * r_i.
        """
        flat_weights = patch_weights.flatten().astype(np.float64)
        N = len(flat_weights)
        n_mask = int(N * self.mask_ratio)

        # 1. Cổng tin cậy theo từng ảnh
        alpha_i = float(self.alpha_max * max(0.0, min(1.0, reliability)))

        # 2. Rank normalization (thứ hạng chuẩn hóa) để chống nhiễu đè bẹp bởi điểm lóa đèn
        ranks = np.argsort(np.argsort(flat_weights)).astype(np.float64)
        w_tilde = ranks / max(1.0, N - 1.0)  # dải [0, 1]

        # 3. Phân phối xác suất Softmax có điều khiển nhiệt độ T
        exp_w = np.exp(w_tilde / max(1e-4, self.temperature))
        p_softmax = exp_w / np.sum(exp_w)

        # 4. Nội suy với phân phối ngẫu nhiên đồng đều
        p_mask = alpha_i * p_softmax + (1.0 - alpha_i) * (1.0 / N)
        p_mask = p_mask / np.sum(p_mask)  # chuẩn hóa bảo đảm tổng bằng 1.0

        # Lấy mẫu không hoàn lại (sampling without replacement)
        masked_indices = np.random.choice(N, size=n_mask, replace=False, p=p_mask)

        mask = np.zeros(N, dtype=bool)
        mask[masked_indices] = True
        return torch.from_numpy(mask)


class BGGuidedDINODataset(Dataset):
    """
    Dataset nạp đồng thời ảnh Origin và Background, tính toán r_i và sinh FAM patch mask.
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
        alpha_max: float = 0.75,
        temperature: float = 0.20,
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
                f"Không thể tìm thấy cặp ảnh nào giữa origin_dir='{origin_dir}' và bg_dir='{bg_dir}'."
            )

        if size_global % patch_size != 0:
            size_global = max(patch_size, round(size_global / patch_size) * patch_size)
        if size_local % patch_size != 0:
            size_local = max(patch_size, round(size_local / patch_size) * patch_size)

        self.subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)
        self.patch_size = patch_size
        self.size_global = size_global
        self.size_local = size_local

        self.augmentor = MultiCropBGGuidedAugmentation(
            size_global=size_global,
            size_local=size_local,
            patch_size=patch_size,
            local_crops_number=local_crops_number,
            mask_ratio=mask_ratio,
            alpha_max=alpha_max,
            temperature=temperature,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Union[List[torch.Tensor], torch.Tensor, float, str]]:
        item = self.pairs[idx]
        try:
            origin_img = Image.open(item["origin_path"]).convert("RGB")
            bg_img = Image.open(item["bg_path"]).convert("RGB")
        except Exception:
            origin_img = Image.new("RGB", (self.size_global, self.size_global), (128, 128, 128))
            bg_img = Image.new("RGB", (self.size_global, self.size_global), (128, 128, 128))

        # 1. Đo độ tin cậy r_i trên vùng tĩnh
        r_i, _ = estimate_background_reliability(origin_img, bg_img)

        # 2. Trừ nền trích xuất delta map
        delta_norm, _ = self.subtractor.compute_delta(origin_img, bg_img)

        # 3. Tính ma trận trọng số sai khác theo patch
        patch_weights = self.subtractor.compute_patch_mask_weights(
            delta_norm=delta_norm,
            img_size=self.size_global,
            patch_size=self.patch_size,
            alpha=1.0,  # Lấy raw delta weights cho bước rank sau đó
        )

        # 4. Multi-Crops
        crops = []
        crops.append(self.augmentor.global_transform_1(origin_img))
        crops.append(self.augmentor.global_transform_2(origin_img))
        for _ in range(self.augmentor.local_crops_number):
            crops.append(self.augmentor.local_transform(origin_img))

        # 5. Sinh mặt nạ FAM với cổng tin cậy r_i
        fg_mask = self.augmentor.generate_fg_mask(patch_weights, reliability=r_i)

        return {
            "crops": crops,
            "fg_mask": fg_mask,
            "reliability": float(r_i),
            "origin_path": item["origin_path"],
            "route_id": item["route_id"],
        }
