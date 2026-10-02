"""
=============================================================================
 Hướng 5: Spatio-Temporal DINO for Continuous Road Space Occupancy 
          and Congestion Level of Service (LoS) Estimation
 Module: Dataset (Chuỗi thời gian khung hình và mỏ neo chiếm dụng mặt đường vật lý)
=============================================================================
"""

import os
import re
import sys
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Nạp module common từ project root
_dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)

from common.matcher import TrafficPairMatcher
from common.subtraction import BackgroundSubtractor


class TemporalTrafficDataset(Dataset):
    """
    Dataset phục vụ bài toán ước lượng Độ chiếm dụng mặt đường (Road Space Occupancy Ratio)
    và Cấp độ dịch vụ giao thông (Level of Service - LoS) theo chuỗi thời gian:
      - Tận dụng trường sai khác quang học Delta = |I_origin - I_bg| để tính toán trực tiếp
        tỷ lệ chiếm dụng mặt đường vật lý rho_phys(t) in [0.0, 1.0] làm mỏ neo tự thân (Self-Supervised Ground Truth).
      - Tự động gán nhãn 4 mức độ ùn tắc theo chuẩn Highway Capacity Manual (HCM):
          + LoS 0 (Free-Flow, Thông thoáng): rho <= 0.15
          + LoS 1 (Moderate, Trung bình): 0.15 < rho <= 0.35
          + LoS 2 (Slow, Đông đúc): 0.35 < rho <= 0.60
          + LoS 3 (Gridlock, Ùn tắc nghiêm trọng): rho > 0.60
      - Tổ chức dữ liệu theo từng cửa sổ trượt thời gian K khung hình liên tiếp.
    """

    def __init__(
        self,
        bg_dir: str,
        origin_dir: str,
        window_size: int = 4,
        img_size: int = 224,
        match_strategy: str = "route_hourly",
        csv_file: Optional[str] = None,
        is_train: bool = True,
        max_sequences: Optional[int] = None,
        delta_threshold: float = 0.15,
    ):
        """
        Khởi tạo TemporalTrafficDataset.

        Args:
            bg_dir: Thư mục chứa ảnh nền tĩnh (traffic_backgrounds).
            origin_dir: Thư mục chứa ảnh origin (output).
            window_size: Số khung hình trong mỗi cửa sổ thời gian K (mặc định 4).
            img_size: Độ phân giải không gian chuẩn hóa (224x224).
            match_strategy: Chiến lược ghép cặp nền ("route_hourly").
            csv_file: Đường dẫn CSV nhãn số lượng xe (nếu có để đối chiếu).
            is_train: Chế độ huấn luyện hay kiểm thử.
            max_sequences: Giới hạn số cửa sổ phục vụ kiểm thử nhanh.
            delta_threshold: Ngưỡng cường độ Delta để coi một pixel là xe cộ.
        """
        super().__init__()
        self.bg_dir = bg_dir
        self.origin_dir = origin_dir
        self.window_size = window_size
        self.img_size = img_size
        self.is_train = is_train
        self.delta_threshold = delta_threshold

        self.matcher = TrafficPairMatcher(
            bg_dir=bg_dir,
            origin_dir=origin_dir,
            match_strategy=match_strategy,
        )
        self.subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)

        # Pipeline chuẩn hóa ảnh cho ViT
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

        # Đọc nhãn phụ trợ nếu có
        self.labels_dict: Dict[str, float] = {}
        if csv_file and os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                fname_col = next((c for c in df.columns if any(k in c.lower() for k in ["file", "image", "name"])), None)
                cnt_col = next((c for c in df.columns if any(k in c.lower() for k in ["tong", "total", "count"])), None)
                if fname_col and cnt_col:
                    for _, row in df.iterrows():
                        self.labels_dict[str(row[fname_col])] = float(row[cnt_col])
            except Exception as e:
                print(f"⚠️ [TemporalDataset Notice] {e}")

        self.camera_sequences = self._group_and_sort_by_camera()
        self.windows = self._build_temporal_windows()

        if max_sequences and len(self.windows) > max_sequences:
            self.windows = self.windows[:max_sequences]

        print(f"✅ [TemporalTrafficDataset] Khởi tạo thành công {len(self.windows)} cửa sổ thời gian (Window Size = {window_size}).")

    def _group_and_sort_by_camera(self) -> Dict[str, List[Dict[str, Any]]]:
        """Gom nhóm ảnh theo camera_id và sắp xếp tăng dần theo timestamp."""
        all_pairs = self.matcher.discover_pairs()
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for p in all_pairs:
            orig_name = p["origin_name"]
            m = re.match(r"^(\d+)_(.+)\.(jpg|jpeg|png)$", orig_name, re.IGNORECASE)
            if m:
                cam_id = m.group(1)
                try:
                    ts = float(m.group(2))
                except ValueError:
                    ts = 0.0
            else:
                cam_id = "unknown"
                ts = 0.0

            item = {
                "origin_name": orig_name,
                "origin_path": p["origin_path"],
                "bg_path": p["bg_path"],
                "cam_id": cam_id,
                "timestamp": ts,
            }
            groups.setdefault(cam_id, []).append(item)

        for cid in groups:
            groups[cid].sort(key=lambda x: x["timestamp"])

        return groups

    def _build_temporal_windows(self) -> List[Dict[str, Any]]:
        """Chia chuỗi ảnh của từng camera thành các cửa sổ trượt kích thước K."""
        windows = []
        for cam_id, items in self.camera_sequences.items():
            if len(items) < self.window_size:
                if len(items) > 0:
                    padded = (items * ((self.window_size // len(items)) + 1))[:self.window_size]
                    windows.append({"cam_id": cam_id, "frames": padded})
                continue

            step = 1 if self.is_train else self.window_size
            for i in range(0, len(items) - self.window_size + 1, step):
                w_current = items[i : i + self.window_size]
                windows.append({"cam_id": cam_id, "frames": w_current})

        return windows

    @staticmethod
    def map_occupancy_to_los(occupancy: float) -> int:
        """
        Quy đổi tỷ lệ chiếm dụng mặt đường sang Cấp độ dịch vụ (LoS) theo chuẩn HCM:
          - 0: Free-flow (rho <= 0.15)
          - 1: Moderate (0.15 < rho <= 0.35)
          - 2: Slow (0.35 < rho <= 0.60)
          - 3: Congested (rho > 0.60)
        """
        if occupancy <= 0.15:
            return 0
        elif occupancy <= 0.35:
            return 1
        elif occupancy <= 0.60:
            return 2
        else:
            return 3

    def _load_window_data(self, frames: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Nạp chuỗi K ảnh origin, K ảnh Delta, tính tỷ lệ chiếm dụng vật lý rho_phys và nhãn LoS.
        Returns:
            rgb_seq: (K, 3, H, W)
            delta_seq: (K, 1, H, W)
            occupancy_seq: (K,)
            los_seq: (K,)
        """
        rgb_list = []
        delta_list = []
        occ_list = []
        los_list = []

        for f_info in frames:
            try:
                orig_img = Image.open(f_info["origin_path"]).convert("RGB")
                bg_img = Image.open(f_info["bg_path"]).convert("RGB")
            except Exception:
                orig_img = Image.fromarray(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
                bg_img = orig_img

            orig_np = np.array(orig_img.resize((self.img_size, self.img_size)))
            bg_np = np.array(bg_img.resize((self.img_size, self.img_size)))

            # Tính toán bản đồ sai khác quang học Delta
            delta_norm, _ = self.subtractor.compute_delta(orig_np, bg_np)

            # Tính tỷ lệ chiếm dụng mặt đường vật lý rho_phys(t) = sum(Delta > threshold) / Total_Pixels
            vehicle_mask = (delta_norm > self.delta_threshold).astype(np.float32)
            occupancy = float(np.mean(vehicle_mask))
            los_class = self.map_occupancy_to_los(occupancy)

            delta_pil = Image.fromarray((np.clip(delta_norm, 0.0, 1.0) * 255.0).astype(np.uint8))

            rgb_tensor = self.transform_rgb(orig_img)
            delta_tensor = self.transform_delta(delta_pil)

            rgb_list.append(rgb_tensor)
            delta_list.append(delta_tensor)
            occ_list.append(occupancy)
            los_list.append(los_class)

        rgb_seq = torch.stack(rgb_list, dim=0)                             # (K, 3, H, W)
        delta_seq = torch.stack(delta_list, dim=0)                         # (K, 1, H, W)
        occupancy_seq = torch.tensor(occ_list, dtype=torch.float32)       # (K,)
        los_seq = torch.tensor(los_list, dtype=torch.long)                 # (K,)

        return rgb_seq, delta_seq, occupancy_seq, los_seq

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        win = self.windows[idx]
        rgb_seq, delta_seq, occ_seq, los_seq = self._load_window_data(win["frames"])

        # Tính xu hướng biến thiên dòng xe qua cửa sổ thời gian d(rho)/dt
        delta_trend = occ_seq[-1] - occ_seq[0]

        # Nhãn phụ trợ đếm xe nếu có trong CSV
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
            "current_occupancy": occ_seq[-1],      # (scalar float)
            "current_los": los_seq[-1],            # (scalar int: 0..3)
            "trend": delta_trend.clone().detach().float() if isinstance(delta_trend, torch.Tensor) else torch.tensor(float(delta_trend), dtype=torch.float32),
            "vehicle_count": torch.tensor(count_label, dtype=torch.float32),
        }
