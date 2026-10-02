"""
=============================================================================
 Hướng 2 (II.A): Scene Decomposition — Dataset Pipeline
 Quản lý dữ liệu phân rã cảnh với cơ chế Nhóm K-Frame Khác Ngày (Multi-day Grouping)
 Đảm bảo nền đường mặt nhựa cố định, đối tượng xe thay đổi vị trí tự nhiên
=============================================================================
"""

import os
import sys
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dino_root not in sys.path:
    sys.path.insert(0, dino_root)
from common.matcher import TrafficPairMatcher
from common.reliability import check_camera_alignment_phase_correlation, estimate_background_reliability


class DecompositionDataset(Dataset):
    """
    Dataset hỗ trợ cả chế độ nạp cặp đơn lẻ (Single Pair) và nhóm K-frame khác ngày (Multi-day Group)
    theo đặc tả A.6 phục vụ ràng buộc nền dùng chung (L_shared).
    """

    def __init__(
        self,
        bg_dir: str,
        origin_dir: str,
        match_strategy: str = "route_hourly",
        img_size: int = 256,
        is_train: bool = True,
        k_frames_per_group: int = 4,
        group_by_camera_slot: bool = False,
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
        self.k_frames = k_frames_per_group
        self.group_by_camera_slot = group_by_camera_slot
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Tổ chức nhóm theo (route_id, slot_hour)
        if self.group_by_camera_slot:
            self.groups = defaultdict(list)
            for p in self.pairs:
                key = (p.get("route_id", "route_unknown"), p.get("slot_hour", "slot_unknown"))
                self.groups[key].append(p)
            self.group_keys = list(self.groups.keys())

    def __len__(self) -> int:
        if self.group_by_camera_slot:
            return len(self.group_keys)
        return len(self.pairs)

    def _load_single(self, origin_path: str, bg_path: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
        try:
            origin_pil = Image.open(origin_path).convert("RGB")
            bg_pil = Image.open(bg_path).convert("RGB")
        except Exception:
            origin_pil = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
            bg_pil = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

        # Ước lượng độ tin cậy r_i
        r_i, _ = estimate_background_reliability(origin_pil, bg_pil)

        origin_pil = origin_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
        bg_pil = bg_pil.resize((self.img_size, self.img_size), Image.BICUBIC)

        if self.is_train and random.random() > 0.5:
            origin_pil = origin_pil.transpose(Image.FLIP_LEFT_RIGHT)
            bg_pil = bg_pil.transpose(Image.FLIP_LEFT_RIGHT)

        origin_t = transforms.ToTensor()(origin_pil)
        bg_t = transforms.ToTensor()(bg_pil)
        return origin_t, bg_t, r_i

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, str, List[torch.Tensor]]]:
        if not self.group_by_camera_slot:
            item = self.pairs[idx]
            origin_t, bg_t, r_i = self._load_single(item["origin_path"], item["bg_path"])
            return {
                "origin": origin_t,
                "bg": bg_t,
                "origin_norm": self.normalize(origin_t),
                "reliability": float(r_i),
                "origin_path": item["origin_path"],
                "route_id": item["route_id"],
            }
        else:
            # Nhóm K frames của cùng (route, slot)
            key = self.group_keys[idx]
            pool = self.groups[key]
            # Lấy ngẫu nhiên K items
            selected = random.choices(pool, k=self.k_frames) if len(pool) < self.k_frames else random.sample(pool, self.k_frames)

            origins = []
            bgs = []
            rel_list = []
            for item in selected:
                o_t, b_t, r_i = self._load_single(item["origin_path"], item["bg_path"])
                origins.append(o_t)
                bgs.append(b_t)
                rel_list.append(r_i)

            return {
                "origins": torch.stack(origins),      # (K, 3, H, W)
                "prior_bg": bgs[0],                    # (3, H, W) đại diện
                "prior_bgs": torch.stack(bgs),        # (K, 3, H, W)
                "reliabilities": torch.tensor(rel_list, dtype=torch.float32),
                "route_id": key[0],
                "slot_hour": key[1],
            }
