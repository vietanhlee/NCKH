"""
=============================================================================
 Hướng 5: Temporal Contrastive Learning for Traffic Density Estimation
 Module: Dataset (Chuỗi thời gian khung hình và bản đồ sai khác quang học)
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
    Dataset nạp chuỗi thời gian các khung hình giao thông (Temporal Sequence of Frames).
    Tổ chức dữ liệu theo từng cửa sổ thời gian (Temporal Window) kích thước K khung hình:
      - Khai thác tính liên tục thời gian giữa các khung hình liên tiếp từ cùng camera.
      - Sinh cặp cửa sổ dương (Positive Pair: W_a, W_b+ liền kề nhau).
      - Đồng bộ và tính toán chuỗi bản đồ sai khác quang học Delta tương ứng với ảnh nền tĩnh.
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
    ):
        """
        Khởi tạo TemporalTrafficDataset.

        Args:
            bg_dir: Thư mục chứa ảnh background (vd: traffic_backgrounds).
            origin_dir: Thư mục chứa ảnh origin (vd: output).
            window_size: Số khung hình trong mỗi cửa sổ thời gian K (mặc định 4).
            img_size: Độ phân giải không gian chuẩn hóa (mặc định 224x224).
            match_strategy: Chiến lược ghép cặp nền ("route_hourly").
            csv_file: Đường dẫn CSV nhãn số lượng xe / mật độ (nếu có nhãn downstream).
            is_train: Chế độ huấn luyện (sinh cặp tương phản) hay kiểm thử.
            max_sequences: Giới hạn số chuỗi tối đa phục vụ debug nhanh.
        """
        super().__init__()
        self.bg_dir = bg_dir
        self.origin_dir = origin_dir
        self.window_size = window_size
        self.img_size = img_size
        self.is_train = is_train

        self.matcher = TrafficPairMatcher(
            bg_dir=bg_dir,
            origin_dir=origin_dir,
            match_strategy=match_strategy,
        )
        self.subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)

        # Chuẩn hóa ảnh cho ViT
        self.transform_rgb = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform_delta = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        # Đọc nhãn nếu có
        self.labels_dict: Dict[str, float] = {}
        if csv_file and os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                fname_col = next((c for c in df.columns if "file" in c.lower() or "image" in c.lower() or "name" in c.lower()), None)
                density_col = next((c for c in df.columns if any(k in c.lower() for k in ["tong", "total", "density", "count"])), None)
                if fname_col and density_col:
                    for _, row in df.iterrows():
                        self.labels_dict[str(row[fname_col])] = float(row[density_col])
            except Exception as e:
                print(f"⚠️ [TemporalDataset] Không thể nạp nhãn từ {csv_file}: {e}")

        # Nhóm các ảnh origin theo Camera ID và sắp xếp theo timestamp
        self.camera_sequences = self._group_and_sort_by_camera()
        self.windows = self._build_temporal_windows()

        if max_sequences and len(self.windows) > max_sequences:
            self.windows = self.windows[:max_sequences]

        print(f"✅ [TemporalTrafficDataset] Đã tạo {len(self.windows)} cửa sổ thời gian (Window Size = {window_size}).")

    def _group_and_sort_by_camera(self) -> Dict[str, List[Dict[str, Any]]]:
        """Gom nhóm ảnh theo camera_id và sắp xếp tăng dần theo timestamp."""
        all_pairs = self.matcher.discover_pairs()
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for p in all_pairs:
            orig_name = p["origin_name"]
            # Trích xuất camera ID và timestamp theo định dạng {route}_{timestamp}.jpg
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

        # Sắp xếp từng camera theo thời gian
        for cid in groups:
            groups[cid].sort(key=lambda x: x["timestamp"])

        return groups

    def _build_temporal_windows(self) -> List[Dict[str, Any]]:
        """Chia các chuỗi camera thành các cửa sổ trượt kích thước K."""
        windows = []
        for cam_id, items in self.camera_sequences.items():
            if len(items) < self.window_size:
                # Nếu camera có ít hơn window_size ảnh, thực hiện lặp lại (cycle pad)
                if len(items) > 0:
                    padded = (items * ((self.window_size // len(items)) + 1))[:self.window_size]
                    windows.append({"cam_id": cam_id, "frames": padded, "next_frames": padded})
                continue

            # Sinh các cửa sổ trượt
            step = max(1, self.window_size // 2)
            for i in range(0, len(items) - self.window_size + 1, step):
                w_current = items[i : i + self.window_size]
                # Cửa sổ liền kề kế tiếp làm cặp dương nếu có
                if i + self.window_size + self.window_size <= len(items):
                    w_next = items[i + self.window_size : i + 2 * self.window_size]
                else:
                    w_next = w_current  # Fallback nếu ở cuối chuỗi

                windows.append({
                    "cam_id": cam_id,
                    "frames": w_current,
                    "next_frames": w_next,
                })
        return windows

    def _load_window_tensor(self, frames: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Nạp chuỗi K ảnh origin và K bản đồ Delta.
        Returns:
            rgb_seq: (K, 3, H, W)
            delta_seq: (K, 1, H, W)
        """
        rgb_list = []
        delta_list = []

        for f_info in frames:
            try:
                orig_img = Image.open(f_info["origin_path"]).convert("RGB")
                bg_img = Image.open(f_info["bg_path"]).convert("RGB")
            except Exception:
                orig_img = Image.fromarray(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
                bg_img = orig_img

            # Resize trước khi trừ nền để đồng bộ
            orig_np = np.array(orig_img.resize((self.img_size, self.img_size)))
            bg_np = np.array(bg_img.resize((self.img_size, self.img_size)))

            delta_norm, _ = self.subtractor.compute_delta(orig_np, bg_np)
            delta_pil = Image.fromarray((np.clip(delta_norm, 0.0, 1.0) * 255.0).astype(np.uint8))

            rgb_tensor = self.transform_rgb(orig_img)
            delta_tensor = self.transform_delta(delta_pil)

            rgb_list.append(rgb_tensor)
            delta_list.append(delta_tensor)

        rgb_seq = torch.stack(rgb_list, dim=0)       # (K, 3, H, W)
        delta_seq = torch.stack(delta_list, dim=0)   # (K, 1, H, W)
        return rgb_seq, delta_seq

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        win = self.windows[idx]

        rgb_a, delta_a = self._load_window_tensor(win["frames"])

        # Trích xuất nhãn trung bình nếu có
        density_label = -1.0
        labels = [self.labels_dict.get(f["origin_name"], -1.0) for f in win["frames"]]
        valid_labels = [l for l in labels if l >= 0]
        if valid_labels:
            density_label = float(np.mean(valid_labels))

        item = {
            "cam_id": win["cam_id"],
            "rgb_seq": rgb_a,           # (K, 3, H, W)
            "delta_seq": delta_a,       # (K, 1, H, W)
            "density": torch.tensor(density_label, dtype=torch.float32),
        }

        # Nếu trong chế độ huấn luyện tương phản, nạp thêm cửa sổ dương liền kề
        if self.is_train:
            rgb_b, delta_b = self._load_window_tensor(win["next_frames"])
            item["rgb_seq_pos"] = rgb_b
            item["delta_seq_pos"] = delta_b

        return item
