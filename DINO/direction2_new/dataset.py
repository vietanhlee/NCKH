"""
=============================================================================
 Hướng 2 (II.A): Scene Decomposition — Dataset Pipeline
 Quản lý dữ liệu phân rã cảnh với cơ chế Nhóm K-Frame Khác Ngày (Multi-day Grouping)
 Đảm bảo nền đường mặt nhựa cố định, đối tượng xe thay đổi vị trí tự nhiên
=============================================================================
"""

import os
import sys
import re
import random
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dino_root not in sys.path:
    sys.path.insert(0, dino_root)
from common.matcher import TrafficPairMatcher
from common.reliability import estimate_background_reliability


def extract_date_from_path(file_path: str) -> str:
    """Trích xuất chuỗi ngày YYYY-MM-DD từ tên file snapshot để nhóm các frame khác ngày."""
    fname = os.path.basename(file_path)
    base = os.path.splitext(fname)[0]
    parts = base.split("_", 1)
    if len(parts) >= 2:
        val = parts[1]
        # Unix timestamp 10 số
        if val.isdigit() and len(val) >= 9:
            try:
                ts = int(val[:10])
                dt = datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=7)))
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        # Chuỗi ngày dạng YYYYMMDD hoặc YYYY-MM-DD
        m_date = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", val)
        if m_date:
            return f"{m_date.group(1)}-{m_date.group(2)}-{m_date.group(3)}"
    return "date_unknown"


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
        compute_reliability: bool = False,
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
        self.compute_reliability = compute_reliability
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Tổ chức nhóm theo (route_id, slot_hour) và gom theo từng ngày riêng biệt
        if self.group_by_camera_slot:
            self.groups = defaultdict(list)
            self.groups_by_date = defaultdict(lambda: defaultdict(list))
            for p in self.pairs:
                key = (str(p.get("route_id", "route_unknown")), int(p.get("origin_hour", p.get("slot_hour", -1))))
                self.groups[key].append(p)
                d = extract_date_from_path(p["origin_path"])
                self.groups_by_date[key][d].append(p)
            self.group_keys = list(self.groups.keys())

    def __len__(self) -> int:
        if self.group_by_camera_slot:
            return len(self.group_keys)
        return len(self.pairs)

    def _load_single(
        self, origin_path: str, bg_path: str, do_flip: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        try:
            origin_pil = Image.open(origin_path).convert("RGB")
            bg_pil = Image.open(bg_path).convert("RGB")
        except Exception as e:
            logger.warning("Không thể đọc ảnh %s hoặc %s: %s", origin_path, bg_path, e)
            origin_pil = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
            bg_pil = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

        # Ước lượng độ tin cậy r_i (chỉ tính khi người dùng yêu cầu để tối đa hóa tốc độ I/O)
        if self.compute_reliability:
            r_i, _ = estimate_background_reliability(origin_pil, bg_pil)
        else:
            r_i = 1.0

        # Resize nhanh bằng Bilinear (nhanh hơn Bicubic 3x trên CPU)
        origin_pil = origin_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        bg_pil = bg_pil.resize((self.img_size, self.img_size), Image.BILINEAR)

        if do_flip:
            origin_pil = origin_pil.transpose(Image.FLIP_LEFT_RIGHT)
            bg_pil = bg_pil.transpose(Image.FLIP_LEFT_RIGHT)

        origin_t = transforms.ToTensor()(origin_pil)
        bg_t = transforms.ToTensor()(bg_pil)
        return origin_t, bg_t, r_i

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, str, List[torch.Tensor]]]:
        if not self.group_by_camera_slot:
            item = self.pairs[idx]
            do_flip = self.is_train and (random.random() > 0.5)
            origin_t, bg_t, r_i = self._load_single(item["origin_path"], item["bg_path"], do_flip=do_flip)
            return {
                "origin": origin_t,
                "bg": bg_t,
                "origin_norm": self.normalize(origin_t),
                "reliability": float(r_i),
                "origin_path": item["origin_path"],
                "route_id": item["route_id"],
            }
        else:
            # Nhóm K frames của cùng (route, slot) từ các NGÀY KHÁC NHAU nếu có thể
            key = self.group_keys[idx]
            date_dict = self.groups_by_date[key]
            available_dates = list(date_dict.keys())

            if len(available_dates) >= self.k_frames:
                chosen_dates = random.sample(available_dates, self.k_frames)
                selected = [random.choice(date_dict[d]) for d in chosen_dates]
            else:
                # Nếu không đủ K ngày khác nhau, lấy mỗi ngày 1 frame rồi lấy thêm từ pool
                selected = [random.choice(date_dict[d]) for d in available_dates]
                pool = self.groups[key]
                remaining = self.k_frames - len(selected)
                if remaining > 0:
                    extra = random.choices(pool, k=remaining) if len(pool) < remaining else random.sample(pool, remaining)
                    selected.extend(extra)

            # Áp dụng 1 quyết định lật ngang đồng nhất cho TOÀN BỘ K frames trong group
            do_flip = self.is_train and (random.random() > 0.5)

            origins = []
            bgs = []
            rel_list = []
            for item in selected:
                o_t, b_t, r_i = self._load_single(item["origin_path"], item["bg_path"], do_flip=do_flip)
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
