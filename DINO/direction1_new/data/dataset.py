"""
=============================================================================
 Hướng 1 Mới (Direction 1 New): Các lớp Dataset giao thông
 Gồm TrafficCameraDataset, SyntheticTrafficDataset và SimpleImageDataset
=============================================================================
"""

import os
import random
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from direction1_new.core.tam import refine_traffic_foreground_map


class TrafficCameraDataset(Dataset):
    """
    Dataset nạp ảnh giao thông thực tế từ thư mục (ví dụ: 'output').
    Tự động nhóm các frame theo camera để phục vụ cơ chế ghép vùng tĩnh SRS (Static Region Swap)
    và lấy mẫu hard-negative cùng camera qua CameraGroupedSampler.
    """
    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (256, 448),
        cross_cam_srs: bool = False,
        min_frame_offset: int = 12,
        tam_cache_dir: Optional[str] = None,
        refine_tam: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.H, self.W = img_size
        self.cross_cam_srs = cross_cam_srs
        self.min_frame_offset = min_frame_offset
        self.tam_cache_dir = tam_cache_dir
        self.refine_tam = refine_tam
        if tam_cache_dir and os.path.isdir(tam_cache_dir):
            print(f"⚡ [TAM Cache] Kích hoạt chế độ đọc trực tiếp bản đồ tiền cảnh pi từ: '{tam_cache_dir}'")

        # 1. Tìm tất cả ảnh hợp lệ
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        all_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    all_files.append(os.path.join(root, f))
        all_files.sort()

        if not all_files:
            raise ValueError(f"Không tìm thấy ảnh hợp lệ (.jpg, .png) trong thư mục '{data_dir}'!")

        # 2. Phân nhóm theo camera (dựa trên tên thư mục con hoặc tiền tố tên file)
        self.cam_to_files = defaultdict(list)
        for path in all_files:
            rel = os.path.relpath(path, data_dir)
            parts = rel.split(os.sep)
            if len(parts) > 1:
                cam_key = parts[0]
            else:
                fname = os.path.splitext(parts[0])[0]
                tokens = fname.split("_")
                cam_key = tokens[0] if len(tokens) > 1 else "cam_0"
            self.cam_to_files[cam_key].append(path)

        self.cam_keys = sorted(list(self.cam_to_files.keys()))
        self.cam_to_id = {k: i for i, k in enumerate(self.cam_keys)}
        self.num_cams = len(self.cam_keys)

        # Danh sách phẳng các mẫu và ánh xạ camera_to_indices cho CameraGroupedSampler
        self.samples = []
        self.camera_to_indices = defaultdict(list)
        for cam_key, files in self.cam_to_files.items():
            cid = self.cam_to_id[cam_key]
            for local_pos, f in enumerate(files):
                global_idx = len(self.samples)
                self.samples.append((f, cid, cam_key, local_pos))
                self.camera_to_indices[cid].append(global_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_img(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (self.W, self.H):
                img = img.resize((self.W, self.H), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        path1, cid, cam_key, local_pos = self.samples[idx]
        x1 = self._load_img(path1)

        # Chọn frame x2 phục vụ SRS
        if self.cross_cam_srs and self.num_cams > 1:
            other_keys = [k for k in self.cam_keys if k != cam_key]
            chosen_key2 = random.choice(other_keys)
            cid2 = self.cam_to_id[chosen_key2]
            path2 = random.choice(self.cam_to_files[chosen_key2])
        else:
            cid2 = cid
            cam_files = self.cam_to_files[cam_key]
            n_files = len(cam_files)
            if n_files > 1:
                distant_cands = [
                    cam_files[j] for j in range(n_files)
                    if abs(j - local_pos) >= self.min_frame_offset
                ]
                if distant_cands:
                    path2 = random.choice(distant_cands)
                else:
                    other_cands = [p for p in cam_files if p != path1]
                    path2 = random.choice(other_cands) if other_cands else path1
            else:
                path2 = path1

        x2 = self._load_img(path2)
        res = {"x1": x1, "x2": x2, "cid": cid, "cid2": cid2}

        # Nếu có cache pi, nạp trực tiếp giúp loại bỏ forward FrozenExtractor lúc train
        if self.tam_cache_dir:
            rel1 = os.path.relpath(path1, self.data_dir)
            f_pi1 = os.path.join(self.tam_cache_dir, rel1 + ".npy")
            rel2 = os.path.relpath(path2, self.data_dir)
            f_pi2 = os.path.join(self.tam_cache_dir, rel2 + ".npy")
            if os.path.isfile(f_pi1) and os.path.isfile(f_pi2):
                try:
                    p1 = torch.from_numpy(np.load(f_pi1)).float()
                    p2 = torch.from_numpy(np.load(f_pi2)).float()
                    if self.refine_tam:
                        res["pi1"] = refine_traffic_foreground_map(p1)
                        res["pi2"] = refine_traffic_foreground_map(p2)
                    else:
                        res["pi1"] = p1
                        res["pi2"] = p2
                except Exception:
                    pass

        return res


class SyntheticTrafficDataset(Dataset):
    """Dataset giả lập phục vụ chạy demo hoặc khi chưa liên kết dataset thực tế."""
    def __init__(self, num_samples: int = 200, num_cams: int = 4):
        self.num_samples = num_samples
        self.num_cams = num_cams
        self.camera_to_indices = defaultdict(list)
        for idx in range(num_samples):
            cid = idx % num_cams
            self.camera_to_indices[cid].append(idx)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        cid = idx % self.num_cams
        x1 = torch.rand(3, 256, 448) * 0.6 + 0.2
        x2 = x1 * 0.9 + 0.1 * torch.rand(3, 256, 448)
        return {"x1": x1, "x2": x2, "cid": cid, "cid2": cid}


class SimpleImageDataset(Dataset):
    """
    Dataset tuần tự nạp ảnh phục vụ tính trước bản đồ tiền cảnh TAM (precompute_tam_cache).
    Trả về ảnh tensor, relative path để lưu cache và camera ID.
    """
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (256, 448)):
        super().__init__()
        self.data_dir = data_dir
        self.H, self.W = img_size

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        all_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    all_files.append(os.path.join(root, f))
        all_files.sort()

        if not all_files:
            raise ValueError(f"Không tìm thấy ảnh hợp lệ trong thư mục '{data_dir}'!")

        self.cam_to_files = defaultdict(list)
        for path in all_files:
            rel = os.path.relpath(path, data_dir)
            parts = rel.split(os.sep)
            if len(parts) > 1:
                cam_key = parts[0]
            else:
                fname = os.path.splitext(parts[0])[0]
                tokens = fname.split("_")
                cam_key = tokens[0] if len(tokens) > 1 else "cam_0"
            self.cam_to_files[cam_key].append((path, rel))

        self.cam_keys = sorted(list(self.cam_to_files.keys()))
        self.cam_to_id = {k: i for i, k in enumerate(self.cam_keys)}
        self.num_cams = len(self.cam_keys)

        self.samples = []
        for cam_key, items in self.cam_to_files.items():
            cid = self.cam_to_id[cam_key]
            for path, rel in items:
                self.samples.append((path, rel, cid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, rel, cid = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (self.W, self.H):
                img = img.resize((self.W, self.H), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, rel, cid
