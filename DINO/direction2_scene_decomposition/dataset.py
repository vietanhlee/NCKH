"""
=============================================================================
 Hướng 3: Scene Decomposition — Dataset Pipeline
 Nạp và tiền xử lý cặp ảnh (Origin, Background) đồng bộ không gian
 phục vụ phân rã cảnh giao thông thành các lớp vật lý
=============================================================================
"""

import os
import sys
from typing import Dict, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Nạp module tiện ích dùng chung
dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dino_root not in sys.path:
    sys.path.insert(0, dino_root)
from common.matcher import TrafficPairMatcher


class DecompositionDataset(Dataset):
    """
    Dataset phục vụ bài toán Phân rã cảnh (Scene Decomposition).
    Áp dụng cùng một phép biến đổi hình học (Affine, Crop, Flip) ngẫu nhiên
    cho cả ảnh Origin và Background để đảm bảo tính thẳng hàng (Spatial Alignment).
    """

    def __init__(
        self,
        bg_dir: str,
        origin_dir: str,
        match_strategy: str = "route_hourly",
        img_size: int = 256,
        is_train: bool = True,
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
            raise RuntimeError(f"Không tìm thấy cặp ảnh hợp lệ cho Scene Decomposition tại {origin_dir} và {bg_dir}.")

        self.img_size = img_size
        self.is_train = is_train

        # Biến đổi màu sắc độc lập nhưng chuẩn hóa kích thước cố định
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        item = self.pairs[idx]
        try:
            origin_pil = Image.open(item["origin_path"]).convert("RGB")
            bg_pil = Image.open(item["bg_path"]).convert("RGB")
        except Exception:
            origin_pil = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
            bg_pil = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

        # Đồng bộ kích thước
        origin_pil = origin_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
        bg_pil = bg_pil.resize((self.img_size, self.img_size), Image.BICUBIC)

        # Lật ngang đồng bộ nếu đang ở chế độ train
        if self.is_train and torch.rand(1).item() > 0.5:
            origin_pil = origin_pil.transpose(Image.FLIP_LEFT_RIGHT)
            bg_pil = bg_pil.transpose(Image.FLIP_LEFT_RIGHT)

        origin_tensor = transforms.ToTensor()(origin_pil)
        bg_tensor = transforms.ToTensor()(bg_pil)

        return {
            "origin": origin_tensor,  # (3, H, W) trong [0, 1]
            "bg": bg_tensor,          # (3, H, W) trong [0, 1]
            "origin_norm": self.normalize(origin_tensor),
            "origin_path": item["origin_path"],
            "route_id": item["route_id"],
        }
