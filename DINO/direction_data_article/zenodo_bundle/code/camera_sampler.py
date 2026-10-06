"""
=============================================================================
Module: camera_sampler.py
Mục đích: Cung cấp các bộ lấy mẫu PyTorch Sampler chuyên biệt cho học tự giám sát
          (Self-Supervised Learning - SSL) trên chuỗi hình ảnh camera giao thông:
          1. CameraGroupedSampler: Lấy nhiều ảnh cùng camera ở các thời điểm khác nhau.
          2. TemporalSequenceSampler: Lấy chuỗi khung hình liên tiếp theo thời gian.
=============================================================================
"""

import random
from typing import Dict, Iterator, List, Optional
import torch
from torch.utils.data import Sampler


class CameraGroupedSampler(Sampler[List[int]]):
    """
    Bộ lấy mẫu theo nhóm Camera phục vụ SSL (Contrastive Learning, DINO, Background-Invariant SSL):
    Mỗi batch gồm `batch_cameras` trạm camera khác nhau, mỗi trạm lấy `frames_per_camera` khung hình
    ở các mốc thời gian khác nhau.
    Tổng kích thước batch (Batch Size) = batch_cameras * frames_per_camera.
    """

    def __init__(
        self,
        camera_to_indices: Dict[int, List[int]],
        batch_cameras: int = 4,
        frames_per_camera: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        """
        Khởi tạo Sampler.

        Args:
            camera_to_indices: Dict ánh xạ từ camera_id -> danh sách chỉ số ảnh trong Dataset.
            batch_cameras: Số lượng camera riêng biệt trong một batch.
            frames_per_camera: Số khung hình được trích xuất cho mỗi camera.
            shuffle: Có xáo trộn thứ tự camera sau mỗi epoch hay không.
            drop_last: Bỏ batch cuối nếu số lượng camera không đủ chia hết.
        """
        self.camera_to_indices: Dict[int, List[int]] = {
            cid: list(indices) for cid, indices in camera_to_indices.items() if len(indices) >= 2
        }
        self.camera_ids: List[int] = sorted(list(self.camera_to_indices.keys()))
        self.batch_cameras: int = batch_cameras
        self.frames_per_camera: int = frames_per_camera
        self.shuffle: bool = shuffle
        self.drop_last: bool = drop_last

        if len(self.camera_ids) == 0:
            raise ValueError("Không tìm thấy camera nào có ít nhất 2 khung hình để lấy mẫu!")

    def __iter__(self) -> Iterator[List[int]]:
        camera_pool = list(self.camera_ids)
        if self.shuffle:
            random.shuffle(camera_pool)

        i = 0
        total_cameras = len(camera_pool)

        while i + self.batch_cameras <= total_cameras or (not self.drop_last and i < total_cameras):
            selected_cameras = camera_pool[i : i + self.batch_cameras]

            # Bù thêm camera ngẫu nhiên nếu batch cuối không đủ và drop_last=False
            if len(selected_cameras) < self.batch_cameras:
                extra_needed = self.batch_cameras - len(selected_cameras)
                selected_cameras.extend(random.choices(self.camera_ids, k=extra_needed))

            batch_indices: List[int] = []
            for cid in selected_cameras:
                candidate_indices = self.camera_to_indices[cid]
                if len(candidate_indices) >= self.frames_per_camera:
                    if self.shuffle:
                        chosen = random.sample(candidate_indices, self.frames_per_camera)
                    else:
                        chosen = candidate_indices[: self.frames_per_camera]
                else:
                    chosen = random.choices(candidate_indices, k=self.frames_per_camera)
                batch_indices.extend(chosen)

            yield batch_indices
            i += self.batch_cameras

    def __len__(self) -> int:
        num_cams = len(self.camera_ids)
        if self.drop_last:
            return num_cams // self.batch_cameras
        return (num_cams + self.batch_cameras - 1) // self.batch_cameras


class TemporalSequenceSampler(Sampler[List[int]]):
    """
    Bộ lấy mẫu chuỗi thời gian liên tục (Temporal Sequence Sampler):
    Lấy chuỗi gồm `sequence_length` khung hình kế tiếp nhau theo thứ tự thời gian
    của cùng một trạm camera (phục vụ Video-MAE, Optical Flow, Spatio-Temporal Transformers).
    """

    def __init__(
        self,
        camera_to_indices: Dict[int, List[int]],
        sequence_length: int = 8,
        stride: int = 1,
        shuffle: bool = True,
    ) -> None:
        """
        Khởi tạo Sampler chuỗi thời gian.

        Args:
            camera_to_indices: Dict ánh xạ từ camera_id -> danh sách chỉ số ảnh đã sắp xếp theo timestamp.
            sequence_length: Độ dài chuỗi khung hình liên tiếp.
            stride: Bước nhảy giữa các khung hình liên tiếp.
            shuffle: Xáo trộn các chuỗi giữa các camera.
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.shuffle = shuffle
        self.sequences: List[List[int]] = []

        # Xây dựng danh sách các chuỗi hợp lệ cho từng camera
        for cid, indices in camera_to_indices.items():
            sorted_indices = sorted(indices)
            total_frames = len(sorted_indices)
            span = (sequence_length - 1) * stride + 1
            for start_idx in range(0, total_frames - span + 1, sequence_length):
                seq = [sorted_indices[start_idx + k * stride] for k in range(sequence_length)]
                self.sequences.append(seq)

        if len(self.sequences) == 0:
            raise ValueError(
                f"Không tạo được chuỗi thời gian nào với sequence_length={sequence_length} và stride={stride}!"
            )

    def __iter__(self) -> Iterator[List[int]]:
        seq_pool = list(self.sequences)
        if self.shuffle:
            random.shuffle(seq_pool)
        for seq in seq_pool:
            yield seq

    def __len__(self) -> int:
        return len(self.sequences)
