"""
=============================================================================
 Hướng G: CameraGroupedSampler
 Bộ lấy mẫu dữ liệu theo nhóm Camera và Ngày để phục vụ SRS & RIC
=============================================================================
"""

import random
from typing import Dict, Iterator, List
from torch.utils.data import Sampler


class CameraGroupedSampler(Sampler[List[int]]):
    """
    Bộ lấy mẫu gom nhóm theo Camera. Mỗi batch bao gồm:
        batch_cameras: Số camera trong 1 batch.
        frames_per_camera: Số frame khác ngày của mỗi camera.
    Tổng batch size = batch_cameras * frames_per_camera.
    """
    def __init__(
        self,
        camera_to_indices: Dict[int, List[int]],
        batch_cameras: int = 4,
        frames_per_camera: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.camera_to_indices = {c: list(idx) for c, idx in camera_to_indices.items() if len(idx) >= 2}
        self.camera_ids = list(self.camera_to_indices.keys())
        self.batch_cameras = batch_cameras
        self.frames_per_camera = frames_per_camera
        self.shuffle = shuffle
        self.drop_last = drop_last

        assert len(self.camera_ids) > 0, "Không có camera nào có ít nhất 2 frame hợp lệ!"

    def __iter__(self) -> Iterator[List[int]]:
        cam_pool = list(self.camera_ids)
        if self.shuffle:
            random.shuffle(cam_pool)

        # Lặp qua các camera
        i = 0
        while i + self.batch_cameras <= len(cam_pool) or (not self.drop_last and i < len(cam_pool)):
            selected_cams = cam_pool[i : i + self.batch_cameras]
            # Nếu thiếu camera ở batch cuối và không drop_last, lấy bù thêm
            if len(selected_cams) < self.batch_cameras:
                extra = self.batch_cameras - len(selected_cams)
                selected_cams.extend(random.choices(self.camera_ids, k=extra))

            batch_indices = []
            for cid in selected_cams:
                cand_idx = self.camera_to_indices[cid]
                if len(cand_idx) >= self.frames_per_camera:
                    if self.shuffle:
                        chosen = random.sample(cand_idx, self.frames_per_camera)
                    else:
                        chosen = cand_idx[: self.frames_per_camera]
                else:
                    chosen = random.choices(cand_idx, k=self.frames_per_camera)
                batch_indices.extend(chosen)

            yield batch_indices
            i += self.batch_cameras

    def __len__(self) -> int:
        n_cams = len(self.camera_ids)
        if self.drop_last:
            return n_cams // self.batch_cameras
        return (n_cams + self.batch_cameras - 1) // self.batch_cameras
