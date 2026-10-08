"""
=============================================================================
 ST-BMB: Sliding Window Traffic Dataset Pipeline
 Quản lý Dữ liệu Chuỗi Thời Gian Khung Hình Liên Tiếp Chuẩn Production
=============================================================================
Cải tiến đột phá chuẩn Fixed CCTV:
  1. Parse Timestamp chuẩn xác bằng regex: re.findall(r"\\d{8,14}", basename) lấy số cuối.
  2. Kiểm tra max_gap: Loại bỏ các cửa sổ có khoảng cách thời gian quá lớn (> max_gap)
     hoặc bị đứt đoạn kết nối camera.
  3. Loại bỏ cửa sổ lỗi: Tuyệt đối KHÔNG nhân bản frame cuối hoặc chèn ảnh xám giả tạo.
  4. Phân chia Train / Val theo Camera ID hoặc thời gian, chống rò rỉ cửa sổ trượt.
  5. Giữ đúng tỷ lệ khung hình (Aspect-Ratio Preserving Letterbox Resize).
  6. Tăng cường dữ liệu đồng bộ (Synchronized Augmentation) và Photometric Jitter
     kèm nhãn (g, b) cho toàn bộ chuỗi W khung hình.
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import glob
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def letterbox_resize(pil_img: Image.Image, target_size: int) -> Image.Image:
    """
    Thay đổi kích thước ảnh bảo toàn tỷ lệ khung hình (Aspect-Ratio Preserving Resize),
    chèn padding màu đen ở các cạnh để đạt kích thước vuông (target_size x target_size).
    """
    w, h = pil_img.size
    if w == target_size and h == target_size:
        return pil_img

    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = pil_img.resize((new_w, new_h), Image.BICUBIC)

    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded.paste(resized, (pad_x, pad_y))
    return padded


class SlidingWindowTrafficDataset(Dataset):
    """
    Dataset phục vụ huấn luyện và đánh giá luồng ST-BMB.
    Tạo các chuỗi W khung hình liên tiếp của cùng camera: shape (W, 3, H, W).
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 5,
        stride: int = 1,
        img_size: int = 256,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        max_gap: Optional[float] = None,
        is_train: Optional[bool] = None,
        max_samples: Optional[int] = None,
        apply_jitter: bool = True,
        apply_crop: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.img_size = img_size
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.max_gap = max_gap
        # Nếu is_train không được truyền, suy ra từ split
        self.is_train = (split == "train") if is_train is None else is_train
        self.apply_jitter = apply_jitter
        self.apply_crop = apply_crop

        # Khám phá và xây dựng danh sách các chuỗi cửa sổ trượt
        self.windows = self._discover_sliding_windows()
        if max_samples is not None and max_samples > 0:
            self.windows = self.windows[:max_samples]

        if len(self.windows) == 0:
            raise RuntimeError(
                f"Không tìm thấy cửa sổ chuỗi thời gian nào thỏa mãn W={window_size}, split='{split}' tại {data_dir}."
            )

        print(
            f"📦 [SlidingWindowTrafficDataset] Đã tạo thành công {len(self.windows)} cửa sổ thời gian "
            f"(Kích thước W={window_size}, Stride={stride}, Split='{split}') từ {data_dir}."
        )

    def _parse_cam_and_time(self, filepath: str) -> Tuple[str, float]:
        """
        Trích xuất (cam_id, timestamp) từ tên tệp:
        1. Hỗ trợ định dạng datetime đầy đủ YYYYMMDD_HHMMSS hoặc YYYYMMDDHHMMSS
           chuyển đổi thành Unix timestamp (giây) theo múi giờ Việt Nam GMT+7.
        2. Hỗ trợ Unix timestamp (10 chữ số giây chuẩn hoặc 13 chữ số mili-giây).
        3. Tự động trích xuất cam_id từ tiền tố trước timestamp.
        """
        basename = os.path.splitext(os.path.basename(filepath))[0]

        # Kiểm tra mẫu YYYYMMDD_HHMMSS (ví dụ 20240510_143000)
        dt_pattern = re.search(r"(\d{8})_(\d{6})", basename)
        if dt_pattern:
            date_str = f"{dt_pattern.group(1)}_{dt_pattern.group(2)}"
            try:
                # Múi giờ Việt Nam GMT+7
                tz_vn = timezone(timedelta(hours=7))
                dt = datetime.strptime(date_str, "%Y%m%d_%H%M%S").replace(tzinfo=tz_vn)
                ts = float(dt.timestamp())
                cam_id = basename[:dt_pattern.start()].rstrip("_-") or "cam_default"
                return cam_id, ts
            except Exception:
                pass

        digits = re.findall(r"\d{8,14}", basename)
        if digits:
            ts_str = digits[-1]
            idx = basename.rfind(ts_str)
            cam_id = basename[:idx].rstrip("_-") or "cam_default"

            # Nếu là chuỗi 14 chữ số YYYYMMDDHHMMSS
            if len(ts_str) == 14:
                try:
                    tz_vn = timezone(timedelta(hours=7))
                    dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S").replace(tzinfo=tz_vn)
                    return cam_id, float(dt.timestamp())
                except Exception:
                    pass
            # Nếu là chuỗi 13 chữ số (mili-giây)
            elif len(ts_str) == 13:
                return cam_id, float(ts_str) / 1000.0
            # Nếu là chuỗi 10 chữ số (Unix epoch giây chuẩn)
            elif len(ts_str) == 10:
                return cam_id, float(ts_str)
            # Nếu là chuỗi 8 chữ số (YYYYMMDD)
            elif len(ts_str) == 8 and (ts_str.startswith("20") or ts_str.startswith("19")):
                try:
                    tz_vn = timezone(timedelta(hours=7))
                    dt = datetime.strptime(ts_str, "%Y%m%d").replace(tzinfo=tz_vn)
                    return cam_id, float(dt.timestamp())
                except Exception:
                    pass

            return cam_id, float(ts_str)

        # Fallback an toàn: thư mục cha làm cam_id, mtime làm timestamp
        parent_dir = os.path.basename(os.path.dirname(filepath))
        mtime = os.path.getmtime(filepath)
        return parent_dir or "cam_default", float(mtime)

    def _discover_sliding_windows(self) -> List[Dict[str, Union[str, List[str], List[float]]]]:
        """
        Quét toàn bộ ảnh và tạo các cửa sổ trượt W khung hình với kiểm tra max_gap
        và phân chia Train/Val theo Camera ID để chống rò rỉ dữ liệu.
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        all_files: List[str] = []

        for root, _, files in os.walk(self.data_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    all_files.append(os.path.join(root, f))

        # Nhóm theo camera
        cam_groups: Dict[str, List[Tuple[float, str]]] = {}
        for path in all_files:
            cam_id, ts = self._parse_cam_and_time(path)
            if cam_id not in cam_groups:
                cam_groups[cam_id] = []
            cam_groups[cam_id].append((ts, path))

        # Phân chia Train/Val theo Camera ID
        all_cams = sorted(list(cam_groups.keys()))
        rng = random.Random(self.seed)

        if len(all_cams) > 1 and self.split in ("train", "val") and self.val_ratio > 0.0:
            shuffled_cams = list(all_cams)
            rng.shuffle(shuffled_cams)
            num_val = max(1, int(round(len(shuffled_cams) * self.val_ratio)))
            val_cams = set(shuffled_cams[:num_val])
            train_cams = set(shuffled_cams[num_val:])

            target_cams = train_cams if self.split == "train" else val_cams
        else:
            target_cams = set(all_cams)

        windows: List[Dict[str, Union[str, List[str], List[float]]]] = []

        for cam_id in sorted(list(target_cams)):
            items = cam_groups[cam_id]
            # Sắp xếp theo thứ tự thời gian tăng dần
            items.sort(key=lambda x: x[0])
            n = len(items)

            # Loại bỏ nếu camera có ít hơn window_size ảnh (KHÔNG nhân bản frame giả tạo)
            if n < self.window_size:
                continue

            # Nếu chỉ có 1 camera và cần chia Train/Val, chia theo thứ tự thời gian
            if len(all_cams) == 1 and self.split in ("train", "val") and self.val_ratio > 0.0:
                split_idx = int(round(n * (1.0 - self.val_ratio)))
                if self.split == "train":
                    items = items[:split_idx]
                else:
                    items = items[split_idx:]
                n = len(items)
                if n < self.window_size:
                    continue

            # Tạo sliding window với bước trượt stride
            for start_idx in range(0, n - self.window_size + 1, self.stride):
                sub_slice = items[start_idx : start_idx + self.window_size]

                # Kiểm tra max_gap: nếu khoảng cách giữa 2 frame liên tiếp > max_gap, loại bỏ cửa sổ
                if self.max_gap is not None and self.max_gap > 0:
                    gap_exceeded = False
                    for i in range(len(sub_slice) - 1):
                        gap = abs(sub_slice[i + 1][0] - sub_slice[i][0])
                        if gap > self.max_gap:
                            gap_exceeded = True
                            break
                    if gap_exceeded:
                        continue

                windows.append({
                    "cam_id": cam_id,
                    "file_paths": [x[1] for x in sub_slice],
                    "timestamps": [x[0] for x in sub_slice],
                })

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, List[float]]]:
        item = self.windows[idx]
        file_paths = item["file_paths"]
        cam_id = item["cam_id"]
        timestamps = item["timestamps"]

        # Tăng cường lật ngang đồng bộ cho toàn bộ W khung hình trong chuỗi
        do_hflip = self.is_train and (random.random() > 0.5)

        # Photometric Jitter ngẫu nhiên có nhãn (gain g, bias b) để kiểm định Illumination Adaptor
        if self.is_train and self.apply_jitter and random.random() < 0.5:
            jitter_g = random.uniform(0.75, 1.30)
            jitter_b = random.uniform(-0.15, 0.15)
        else:
            jitter_g = 1.0
            jitter_b = 0.0

        # Synchronous Random Crop box (tính 1 lần cho cả chuỗi W frames để bảo toàn phối cảnh tĩnh)
        crop_box = None
        if self.is_train and self.apply_crop and random.random() < 0.5:
            try:
                with Image.open(file_paths[0]) as first_img:
                    w_orig, h_orig = first_img.size
                scale = random.uniform(0.75, 1.0)
                crop_w = max(16, int(round(w_orig * scale)))
                crop_h = max(16, int(round(h_orig * scale)))
                crop_x = random.randint(0, max(0, w_orig - crop_w))
                crop_y = random.randint(0, max(0, h_orig - crop_h))
                crop_box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
            except Exception:
                crop_box = None

        frames_list: List[torch.Tensor] = []
        for p_idx, p in enumerate(file_paths):
            try:
                pil_img = Image.open(p).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Lỗi đọc ảnh tại đường dẫn {p}: {e}")

            # Cắt ảnh đồng bộ (Synchronous Crop)
            if crop_box is not None:
                pil_img = pil_img.crop(crop_box)

            # Resize bảo toàn tỷ lệ khung hình (Letterbox)
            pil_img = letterbox_resize(pil_img, self.img_size)

            if do_hflip:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

            t_img = transforms.ToTensor()(pil_img)  # (3, H, W) in [0, 1]

            # Áp dụng jitter cho frame cuối nếu có jitter
            if p_idx == len(file_paths) - 1 and (jitter_g != 1.0 or jitter_b != 0.0):
                t_img = torch.clamp(t_img * jitter_g + jitter_b, 0.0, 1.0)

            frames_list.append(t_img)

        # Xếp chồng thành tensor thời gian 4D: (W, 3, H, W)
        frames_tensor = torch.stack(frames_list, dim=0)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)

        return {
            "frames": frames_tensor,                    # (W, 3, H, W)
            "cam_id": cam_id,
            "timestamps": timestamps_tensor,           # (W,)
            "jitter_gain": float(jitter_g),
            "jitter_bias": float(jitter_b),
        }
