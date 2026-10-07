"""
=============================================================================
 Hướng C: Anomaly Detection — Temporal Feature Pooling
 Gộp đặc trưng theo thời gian trong không gian Feature (Median Filtering)
 Loại bỏ xe cộ chuyển động thoáng qua, duy trì và khuếch đại sự cố bất thường kéo dài
=============================================================================
"""

from typing import List, Optional, Tuple, Deque
from collections import deque
import torch
import numpy as np


class TemporalFeaturePooler:
    """
    Bộ đệm thời gian gộp đặc trưng Patch qua cửa sổ trượt W frames.
    
    Công thức:
        \\tilde{F}_t(p) = \\operatorname{median}_{w=0}^{W-1} f_{t-w}(p)
        
    Ý nghĩa:
        - Xe cộ đang lưu thông chỉ lướt qua patch p trong 1-2 frames -> Bị median lọc sạch.
        - Sự cố kéo dài (ngập nước ngâm đường, cây ngã, xe tai nạn đứng yên) -> Chiếm ưu thế
          trong cửa sổ W -> Được giữ lại và làm nổi bật trong vector đặc trưng.
    """

    def __init__(self, window_size: int = 5, normalize: bool = True):
        """
        Args:
            window_size: Kích thước cửa sổ thời gian W (khuyến nghị 5 - 10 frames).
            normalize: Có chuẩn hóa L2 vector sau khi gộp không.
        """
        self.W = window_size
        self.normalize = normalize
        self.buffer: Deque[torch.Tensor] = deque(maxlen=window_size)

    def reset(self) -> None:
        """Làm rỗng bộ đệm."""
        self.buffer.clear()

    def update(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """
        Đẩy đặc trưng frame mới vào bộ đệm và tính đặc trưng gộp:
        
        Args:
            patch_feats: Tensor (B, N_patches, D) hoặc (N_patches, D).
            
        Returns:
            pooled_feats: Tensor cùng shape, biểu diễn đặc trưng bền vững sau khi loại xe động.
        """
        # Lưu vào buffer (tách khỏi autograd graph để tiết kiệm VRAM)
        self.buffer.append(patch_feats.detach().clone())

        # Xếp chồng tensor theo chiều thời gian: (W_curr, B, N, D)
        stacked = torch.stack(list(self.buffer), dim=0)

        # Tính median theo chiều thời gian (dim=0)
        # torch.median trả về (values, indices)
        median_feats, _ = torch.median(stacked, dim=0)

        if self.normalize:
            median_feats = torch.nn.functional.normalize(median_feats, p=2, dim=-1)

        return median_feats

    @property
    def is_full(self) -> bool:
        """Kiểm tra bộ đệm đã tích lũy đủ W frames hay chưa."""
        return len(self.buffer) == self.W

    @property
    def current_length(self) -> int:
        return len(self.buffer)
