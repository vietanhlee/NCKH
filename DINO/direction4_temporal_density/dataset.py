"""
=============================================================================
 Hướng 4: Spatio-Temporal DINO for Continuous Road Space Occupancy 
          and Congestion Level Estimation
 Module: Dataset (Chuỗi thời gian khung hình và mỏ neo chiếm dụng lòng đường)
 Chuẩn Q1: rho_proxy tính strictly trên Road Mask |R|, tích hợp Cổng tin cậy r_i
=============================================================================
"""

import os
import re
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

_dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)

from common.matcher import TrafficPairMatcher
from common.subtraction import BackgroundSubtractor
from common.reliability import estimate_background_reliability


class TemporalTrafficDataset(Dataset):
    """
    Dataset phục vụ bài toán ước lượng Độ chiếm dụng mặt đường (Road Space Occupancy Ratio rho_proxy)
    và Mức độ ùn tắc (Congestion Level) theo chuỗi thời gian:
      - Tỷ lệ chiếm dụng mặt đường tính strictly trên Road Mask |R|:
            rho_proxy(t) = (1 / |R|) * sum_{(u, v) in R} 1(Delta_t(u, v) > tau)
      - Phân lớp mức ùn tắc (Congestion Levels 0..3):
          + Level 0 (Free-Flow, Thông thoáng)
          + Level 1 (Moderate, Trung bình)
          + Level 2 (Slow, Đông đúc)
          + Level 3 (Gridlock, Ùn tắc nghiêm trọng)
      - Tổ chức dữ liệu theo cửa sổ trượt thời gian K khung hình liên tiếp.
    """

    def __init__(
        self,
        bg_dir: str,
        origin_dir: str,
        window_size: int = 4,
        img_size: int = 224,
        match_strategy: str = "route_hourly",
        csv_file: Optional[str] = None,
        road_masks_dict: Optional[Dict[str, np.ndarray]] = None,
        is_train: bool = True,
        max_sequences: Optional[int] = None,
        delta_threshold: float = 0.15,
        thresholds_los: Tuple[float, float, float] = (0.15, 0.35, 0.60),
    ):
        super().__init__()
        self.bg_dir = bg_dir
        self.origin_dir = origin_dir
        self.window_size = window_size
        self.img_size = img_size
        self.is_train = is_train
        self.delta_threshold = delta_threshold
        self.thresholds_los = thresholds_los
        self.road_masks_dict = road_masks_dict or {}

        self.matcher = TrafficPairMatcher(
            bg_dir=bg_dir,
            origin_dir=origin_dir,
            match_strategy=match_strategy,
        )
        self.subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)

        self.transform_rgb = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform_delta = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.labels_dict: Dict[str, float] = {}
        if csv_file and os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                fcol = [c for c in df.columns if "file" in c.lower() or "image" in c.lower()]
                tcol = [c for c in df.columns if "tong" in c.lower() or "total" in c.lower() or "count" in c.lower()]
                if fcol and tcol:
                    for _, r in df.iterrows():
                        self.labels_dict[str(r[fcol[0]]).strip()] = float(r[tcol[0]])
            except Exception:
                pass

        self.windows = self._build_sliding_windows(max_sequences=max_sequences)

    def _build_sliding_windows(self, max_sequences: Optional[int]) -> List[Dict[str, Any]]:
        self.matcher.build_background_index()
        origins = self.matcher.list_origin_images()

        camera_groups: Dict[str, List[Dict[str, Any]]] = {}
        for p in origins:
            fname = os.path.basename(p)
            route_id, ts, hour = self.matcher.parse_origin_filename(fname)
            if route_id is None or ts is None:
                continue

            bg_cand = self.matcher.find_best_background(route_id, hour if hour is not None else 12)
            if not bg_cand:
                continue

            if route_id not in camera_groups:
                camera_groups[route_id] = []

            camera_groups[route_id].append({
                "origin_path": p,
                "origin_name": fname,
                "timestamp": ts,
                "hour": hour,
                "bg_path": bg_cand[0],
            })

        windows = []
        for cam_id, frames in camera_groups.items():
            frames.sort(key=lambda x: x["timestamp"])
            if len(frames) < self.window_size:
                if len(frames) >= 2:
                    pad_frames = frames + [frames[-1]] * (self.window_size - len(frames))
                    windows.append({"cam_id": cam_id, "frames": pad_frames})
                continue

            stride = 1 if self.is_train else max(1, self.window_size // 2)
            for i in range(0, len(frames) - self.window_size + 1, stride):
                win_frames = frames[i : i + self.window_size]
                windows.append({"cam_id": cam_id, "frames": win_frames})

        if max_sequences and len(windows) > max_sequences:
            windows = windows[:max_sequences]
        return windows

    def map_occupancy_to_los(self, occ: float) -> int:
        th1, th2, th3 = self.thresholds_los
        if occ <= th1:
            return 0  # Free-flow
        elif occ <= th2:
            return 1  # Moderate
        elif occ <= th3:
            return 2  # Slow
        else:
            return 3  # Gridlock

    def _get_road_mask(self, cam_id: str, H: int, W: int) -> np.ndarray:
        """Lấy road mask của camera hoặc dùng proxy 65% phía dưới lòng đường."""
        if cam_id in self.road_masks_dict:
            m = self.road_masks_dict[cam_id]
            if m.shape != (H, W):
                m_img = Image.fromarray(m.astype(np.uint8)).resize((W, H), Image.NEAREST)
                return np.array(m_img) > 0
            return m > 0
        # Mặc định: loại bỏ 35% trên đỉnh (bầu trời, nhà)
        mask = np.zeros((H, W), dtype=bool)
        mask[int(H * 0.35):, :] = True
        return mask

    def _load_window_data(
        self,
        frames_info: List[Dict[str, Any]],
        cam_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_list = []
        delta_list = []
        occ_list = []
        los_list = []
        rel_list = []

        road_mask = self._get_road_mask(cam_id, self.img_size, self.img_size)
        road_pixels = max(1, int(np.sum(road_mask)))

        for f_info in frames_info:
            try:
                orig_img = Image.open(f_info["origin_path"]).convert("RGB")
                bg_img = Image.open(f_info["bg_path"]).convert("RGB")
            except Exception:
                orig_img = Image.fromarray(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
                bg_img = orig_img

            # Đo độ tin cậy r_i từ vùng tĩnh
            r_i, _ = estimate_background_reliability(orig_img, bg_img)

            orig_np = np.array(orig_img.resize((self.img_size, self.img_size)))
            bg_np = np.array(bg_img.resize((self.img_size, self.img_size)))

            delta_norm, _ = self.subtractor.compute_delta(orig_np, bg_np)

            # Tính toán rho_proxy CHỈ TRÊN ROAD MASK |R|
            vehicle_mask = (delta_norm > self.delta_threshold) & road_mask
            occupancy = float(np.sum(vehicle_mask) / road_pixels)
            occupancy = max(0.0, min(1.0, occupancy))
            los_class = self.map_occupancy_to_los(occupancy)

            delta_pil = Image.fromarray((np.clip(delta_norm, 0.0, 1.0) * 255.0).astype(np.uint8))

            rgb_tensor = self.transform_rgb(orig_img)
            delta_tensor = self.transform_delta(delta_pil)

            rgb_list.append(rgb_tensor)
            delta_list.append(delta_tensor)
            occ_list.append(occupancy)
            los_list.append(los_class)
            rel_list.append(r_i)

        rgb_seq = torch.stack(rgb_list, dim=0)                             # (K, 3, H, W)
        delta_seq = torch.stack(delta_list, dim=0)                         # (K, 1, H, W)
        occupancy_seq = torch.tensor(occ_list, dtype=torch.float32)       # (K,)
        los_seq = torch.tensor(los_list, dtype=torch.long)                 # (K,)
        rel_seq = torch.tensor(rel_list, dtype=torch.float32)              # (K,)

        return rgb_seq, delta_seq, occupancy_seq, los_seq, rel_seq

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        win = self.windows[idx]
        rgb_seq, delta_seq, occ_seq, los_seq, rel_seq = self._load_window_data(win["frames"], win["cam_id"])

        delta_trend = occ_seq[-1] - occ_seq[0]
        count_label = -1.0
        counts = [self.labels_dict.get(f["origin_name"], -1.0) for f in win["frames"]]
        valid_counts = [c for c in counts if c >= 0]
        if valid_counts:
            count_label = float(valid_counts[-1])

        return {
            "cam_id": win["cam_id"],
            "rgb_seq": rgb_seq,                    # (K, 3, H, W)
            "delta_seq": delta_seq,                # (K, 1, H, W)
            "occupancy_seq": occ_seq,              # (K,)
            "los_seq": los_seq,                    # (K,)
            "reliability_seq": rel_seq,            # (K,)
            "current_occupancy": occ_seq[-1],      # scalar float
            "current_los": los_seq[-1],            # scalar int: 0..3
            "trend": float(delta_trend),
            "vehicle_count": torch.tensor(count_label, dtype=torch.float32),
        }
