"""
=============================================================================
Module: load_unlabeled_images.py
Mục đích: PyTorch Dataset chuẩn production nạp chuỗi hình ảnh camera giao thông
          thực tế KHÔNG NHÃN (HCMC-TrafficCam7D).
          Tối ưu hóa cho Self-Supervised Learning (SSL), DINO, Masked Autoencoder,
          học biểu diễn không-thời gian và phát hiện bất thường không giám sát.
=============================================================================
"""

import os
import sys

# Ngăn chặn xung đột runtime thư viện OpenMP kép trên môi trường Windows / Anaconda
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime, timezone, timedelta
from typing import Optional, Callable, Tuple, Dict, Any, List
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset

# Múi giờ chuẩn Việt Nam (UTC+7)
VN_TZ = timezone(timedelta(hours=7))


class TrafficCameraUnlabeledDataset(Dataset):
    """
    PyTorch Dataset nạp ảnh camera giao thông không nhãn từ 608 trạm tại TP.HCM.

    Quy ước đặt tên tệp:
        {station_id}_{unix_timestamp}.jpg  (Ví dụ: 1_1755698811.jpg)

    Cung cấp:
        - Image Tensor chuẩn hóa theo torchvision transform.
        - Siêu dữ liệu ngữ cảnh phong phú: station_id, unix_timestamp, hour_of_day,
          day_of_week, local_time_string, file_path.
        - Bản đồ chỉ số camera_to_indices phục vụ gom nhóm theo camera (Grouped Sampler).
    """

    def __init__(
        self,
        image_dir: str,
        resolution: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        allowed_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        preload_index: bool = True,
    ) -> None:
        """
        Khởi tạo Dataset.

        Args:
            image_dir (str): Đường dẫn đến thư mục chứa ảnh (hoặc một ngày shard).
            resolution (Tuple[int, int], optional): Độ phân giải mục tiêu (W, H) để resize ảnh tự động.
            transform (Callable, optional): Pipeline biến đổi ảnh torchvision.
            allowed_extensions (Tuple[str, ...]): Các định dạng ảnh được chấp nhận.
            preload_index (bool): Quét và lập chỉ mục metadata ngay khi khởi tạo.
        """
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh tại: {image_dir}")

        self.image_dir: str = image_dir
        self.resolution: Optional[Tuple[int, int]] = resolution
        self.transform: Optional[Callable] = transform
        self.allowed_extensions: Tuple[str, ...] = tuple(ext.lower() for ext in allowed_extensions)

        self.samples: List[Dict[str, Any]] = []
        self.camera_to_indices: Dict[int, List[int]] = defaultdict(list)

        if preload_index:
            self._scan_and_index()

    def _scan_and_index(self) -> None:
        """Quét đĩa đệ quy và lập chỉ mục toàn bộ ảnh, trích xuất camera_id và timestamp."""
        idx_counter = 0

        for root_dir, _, filenames in os.walk(self.image_dir):
            for fname in sorted(filenames):
                # Bỏ qua tệp ẩn hoặc tệp không đúng định dạng
                if fname.startswith("."):
                    continue
                lower_fname = fname.lower()
                if not any(lower_fname.endswith(ext) for ext in self.allowed_extensions):
                    continue

                full_path = os.path.join(root_dir, fname)

                # Phân tích cú pháp tên tệp: {camera_id}_{unix_timestamp}.jpg
                base_name, _ = os.path.splitext(fname)
                parts = base_name.split("_")

                station_id: int = -1
                unix_ts: int = 0
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    station_id = int(parts[0])
                    unix_ts = int(parts[1])
                elif len(parts) == 1 and parts[0].isdigit():
                    station_id = int(parts[0])

                # Tính toán thời gian cục bộ (Việt Nam UTC+7)
                hour_of_day: int = -1
                day_of_week: int = -1
                dt_str: str = ""
                if unix_ts > 0:
                    try:
                        dt = datetime.fromtimestamp(unix_ts, tz=VN_TZ)
                        hour_of_day = dt.hour
                        day_of_week = dt.weekday()  # 0: Thứ Hai, 6: Chủ Nhật
                        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass

                meta_record: Dict[str, Any] = {
                    "file_path": full_path,
                    "file_name": fname,
                    "station_id": station_id,
                    "unix_timestamp": unix_ts,
                    "hour_of_day": hour_of_day,
                    "day_of_week": day_of_week,
                    "local_time_str": dt_str,
                }

                self.samples.append(meta_record)
                if station_id >= 0:
                    self.camera_to_indices[station_id].append(idx_counter)
                idx_counter += 1

        print(
            f"[TrafficCameraUnlabeledDataset] Loaded index of {len(self.samples)} frames "
            f"across {len(self.camera_to_indices)} active camera stations."
        )

    def __len__(self) -> int:
        """Trả về tổng số khung hình trong dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Nạp một khung hình ảnh và siêu dữ liệu đi kèm.

        Returns:
            Tuple: (image_tensor [C, H, W], metadata_dict)
        """
        sample_meta = self.samples[idx]
        img_path = sample_meta["file_path"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            # Tránh làm sập tiến trình huấn luyện khi gặp 1 khung hình hỏng ngẫu nhiên
            print(f"⚠️ [TrafficCameraUnlabeledDataset] Cảnh báo lỗi đọc ảnh {img_path}: {exc}")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.resolution is not None and image.size != self.resolution:
            image = image.resize(self.resolution, Image.BILINEAR)

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            import torchvision.transforms.functional as TF
            image_tensor = TF.to_tensor(image)

        return image_tensor, sample_meta


# Alias tương thích với tài liệu hướng dẫn nhanh (README.md)
CameraSequenceDataset = TrafficCameraUnlabeledDataset

