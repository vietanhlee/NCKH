"""
=============================================================================
 Hướng 4: Foreground-Enhanced Counting — Dataset Module
 Nạp dữ liệu ảnh giao thông, ảnh nền tĩnh và nhãn đếm phương tiện từ CSV
 Hỗ trợ phân vùng Spatial Disjoint Camera Split chống rò rỉ dữ liệu
=============================================================================
"""

import os
import re
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Nạp module tiện ích dùng chung
dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dino_root not in sys.path:
    sys.path.insert(0, dino_root)
from common.matcher import TrafficPairMatcher
from common.subtraction import BackgroundSubtractor


class FGCountingDataset(Dataset):
    """
    Dataset phục vụ bài toán Ước lượng lưu lượng phương tiện tăng cường Foreground:
      - Nạp ảnh Origin RGB và Background tương ứng.
      - Trích xuất bản đồ sai khác $\\Delta$ (1 kênh chuẩn hóa).
      - Ghép với nhãn đếm phương tiện ground-truth (xe máy, ô tô, tổng).
    """

    def __init__(
        self,
        csv_file: str,
        origin_dir: str,
        bg_dir: str,
        match_strategy: str = "route_hourly",
        img_size: int = 224,
        is_train: bool = True,
        camera_id_list: Optional[List[str]] = None,
        few_shot_ratio: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.img_size = img_size
        self.is_train = is_train

        # 1. Đọc file nhãn CSV
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"Không tìm thấy file nhãn CSV tại: {csv_file}")

        df = pd.read_csv(csv_file)
        # Chuẩn hóa tên cột chính xác chống nhầm lẫn chuỗi con (ví dụ 'tong' chứa 'to')
        col_map = {}
        for c in df.columns:
            c_low = c.lower().strip()
            if "file" in c_low or "image" in c_low:
                col_map[c] = "filename"
            elif "tong" in c_low or "total" in c_low:
                col_map[c] = "tong"
            elif "may" in c_low or "bike" in c_low or "moto" in c_low:
                col_map[c] = "xe_may"
            elif "o_to" in c_low or "oto" in c_low or "car" in c_low or c_low == "to":
                col_map[c] = "o_to"
        df = df.rename(columns=col_map)

        # Lọc theo danh sách camera ID để đảm bảo phân vùng Spatial Disjoint
        if camera_id_list is not None:
            cam_set = set([str(int(c)) if str(c).isdigit() else str(c) for c in camera_id_list])
            def extract_cam(fname):
                m = re.search(r"^(\d+)_", str(fname))
                return str(int(m.group(1))) if m else ""
            df["cam_id"] = df["filename"].apply(extract_cam)
            df = df[df["cam_id"].isin(cam_set)].reset_index(drop=True)

        # Few-shot subsampling
        if is_train and few_shot_ratio < 1.0:
            df = df.sample(frac=few_shot_ratio, random_state=seed).reset_index(drop=True)

        self.df = df
        self.matcher = TrafficPairMatcher(bg_dir=bg_dir, origin_dir=origin_dir, match_strategy=match_strategy)
        self.matcher.build_background_index()
        self.subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)

        # Augmentation và Normalization
        self.normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        row = self.df.iloc[idx]
        fname_raw = str(row["filename"]).strip()
        fname_base = os.path.basename(fname_raw)

        # Hỗ trợ cả full path hoặc chỉ tên file trong origin_dir
        origin_path = os.path.join(self.matcher.origin_dir, fname_base)
        if not os.path.isfile(origin_path) and os.path.isfile(fname_raw):
            origin_path = fname_raw

        # Nhãn targets: [xe_may, o_to, tong]
        counts = torch.tensor([
            float(row.get("xe_may", 0)),
            float(row.get("o_to", 0)),
            float(row.get("tong", 0)),
        ], dtype=torch.float32)

        # Nạp ảnh Origin
        try:
            origin_pil = Image.open(origin_path).convert("RGB")
        except Exception:
            origin_pil = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

        # Tìm ảnh Background phù hợp
        route_id, _, hour = self.matcher.parse_origin_filename(fname_base)
        search_hour = hour if hour is not None else 12
        bg_match = self.matcher.find_best_background(route_id, search_hour) if route_id else None

        if bg_match:
            try:
                bg_pil = Image.open(bg_match[0]).convert("RGB")
            except Exception:
                bg_pil = Image.new("RGB", origin_pil.size, (128, 128, 128))
        else:
            bg_pil = Image.new("RGB", origin_pil.size, (128, 128, 128))

        # Resize đồng bộ
        origin_pil = origin_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
        bg_pil = bg_pil.resize((self.img_size, self.img_size), Image.BICUBIC)

        origin_np = np.array(origin_pil)
        bg_np = np.array(bg_pil)

        # Tính toán bản đồ Delta
        delta_norm, _ = self.subtractor.compute_delta(origin_np, bg_np)

        # Chuẩn bị Tensor
        # 1. Ảnh RGB 3 kênh
        rgb_tensor = transforms.ToTensor()(origin_pil)
        rgb_norm = self.normalize_rgb(rgb_tensor)

        # 2. Delta 1 kênh chuẩn hóa [-1, 1]
        delta_tensor = torch.from_numpy(delta_norm).unsqueeze(0).float()
        delta_normed = (delta_tensor - 0.5) / 0.5

        # 3. Kết hợp 4 kênh (RGB + Delta)
        tensor_4ch = torch.cat([rgb_norm, delta_normed], dim=0)

        return {
            "rgb": rgb_norm,          # (3, H, W)
            "input_4ch": tensor_4ch,  # (4, H, W)
            "delta": delta_tensor,    # (1, H, W) trong [0, 1]
            "counts": counts,         # (3,)
            "filename": fname_base,
            "route_id": str(route_id) if route_id else "unknown",
        }
