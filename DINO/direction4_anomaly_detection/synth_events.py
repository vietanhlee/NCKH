"""
=============================================================================
 Hướng C: Anomaly Detection — Synthetic Anomaly Suite
 Bộ sinh sự cố tổng hợp có kiểm soát cho Benchmark Đánh giá:
   1. Vật cản / Xe chết máy nằm yên trong Road Mask
   2. Vùng ngập nước đô thị (Road Inundation)
   3. Camera xoay lệch góc / rung giật (Camera Fault)
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np


class SyntheticAnomalyGenerator:
    """
    Tạo các sự kiện bất thường giả lập có Ground-truth chính xác 100% để đo:
    - AUROC cấp Patch và Khung hình
    - Độ trễ phát hiện (Detection Delay tính bằng số frames / phút)
    - Tỷ lệ báo động sai (False Alarm Rate)
    """

    @staticmethod
    def inject_static_obstacle(
        frame_seq: torch.Tensor,
        road_mask: torch.Tensor,
        start_t: int = 5,
        duration: int = 10,
        box_size: Tuple[int, int] = (40, 60),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dán một vật cản cố định (xe hỏng / chướng ngại vật) vào lòng đường từ start_t đến start_t + duration.

        Args:
            frame_seq: Chuỗi ảnh (T, 3, H, W)
            road_mask: Mặt nạ đường (H, W)
            start_t: Thời điểm bắt đầu chèn sự cố
            duration: Số frames sự cố kéo dài
            box_size: Kích thước vật cản (h, w)
            
        Returns:
            corrupted_seq: Chuỗi ảnh đã chèn vật cản (T, 3, H, W)
            gt_masks: Ground-truth mask bất thường theo thời gian (T, 1, H, W)
        """
        T, C, H, W = frame_seq.shape
        out_seq = frame_seq.clone()
        gt_masks = torch.zeros((T, 1, H, W), device=frame_seq.device)

        # Tìm tọa độ hợp lệ trên mặt đường
        road_indices = torch.nonzero(road_mask > 0.5, as_tuple=False)
        if len(road_indices) == 0:
            center_y, center_x = H // 2, W // 2
        else:
            mid_idx = len(road_indices) // 2
            center_y = int(road_indices[mid_idx, 0].item())
            center_x = int(road_indices[mid_idx, 1].item())

        bh, bw = box_size
        y1 = max(0, center_y - bh // 2)
        y2 = min(H, y1 + bh)
        x1 = max(0, center_x - bw // 2)
        x2 = min(W, x1 + bw)

        end_t = min(T, start_t + duration)
        for t in range(start_t, end_t):
            # Tạo vật cản màu sắc tương phản cao (giả lập vật cản kim loại hoặc container)
            obstacle_color = torch.tensor([0.8, 0.2, 0.2], device=frame_seq.device).view(3, 1, 1)
            out_seq[t, :, y1:y2, x1:x2] = obstacle_color
            gt_masks[t, 0, y1:y2, x1:x2] = 1.0

        return out_seq, gt_masks

    @staticmethod
    def inject_road_flooding(
        frame_seq: torch.Tensor,
        road_mask: torch.Tensor,
        start_t: int = 4,
        duration: int = 12,
        darken_factor: float = 0.45,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Giả lập ngập lụt mặt đường: Mặt đường bị sẫm màu, bóng nước phản chiếu đục.
        """
        T, C, H, W = frame_seq.shape
        out_seq = frame_seq.clone()
        gt_masks = torch.zeros((T, 1, H, W), device=frame_seq.device)

        end_t = min(T, start_t + duration)
        r_mask_3d = (road_mask > 0.5).unsqueeze(0).expand(3, -1, -1)

        for t in range(start_t, end_t):
            curr_frame = out_seq[t]
            # Giảm sáng mặt đường và tăng tính tương phản phản xạ
            flooded_road = curr_frame * darken_factor
            # Trộn vệt nước đục
            flooded_road = torch.where(r_mask_3d, flooded_road, curr_frame)
            out_seq[t] = flooded_road
            gt_masks[t, 0] = (road_mask > 0.5).float()

        return out_seq, gt_masks

    @staticmethod
    def inject_camera_shift_fault(
        frame_seq: torch.Tensor,
        start_t: int = 6,
        duration: int = 10,
        shift_pixels: Tuple[int, int] = (25, 20),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Giả lập camera bị ngoại lực xô lệch góc quay (Camera Fault).
        """
        T, C, H, W = frame_seq.shape
        out_seq = frame_seq.clone()
        gt_is_fault = torch.zeros(T, device=frame_seq.device)

        dy, dx = shift_pixels
        end_t = min(T, start_t + duration)

        for t in range(start_t, end_t):
            shifted = torch.roll(out_seq[t], shifts=(dy, dx), dims=(-2, -1))
            out_seq[t] = shifted
            gt_is_fault[t] = 1.0

        return out_seq, gt_is_fault
