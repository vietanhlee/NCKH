"""
=============================================================================
 Common Utility: Background & Camera Reliability Estimation
 Module đo lường định lượng độ tin cậy của ảnh nền (Weak Prior) & căn chỉnh camera
 Theo đặc tả nghiên cứu Q1: Coi background là tín hiệu yếu có nhiễu (Weak/Noisy Prior)
=============================================================================
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional, Union, Dict


def estimate_static_mask_from_sequence(
    frames: np.ndarray,
    variance_threshold: float = 12.0
) -> np.ndarray:
    """
    Ước lượng vùng tĩnh (static region - tòa nhà, cột điện, bầu trời) dựa trên
    phương sai cường độ theo thời gian giữa các frame.
    Lưu ý: Không dùng phần bù của road mask thô vì vỉa hè có người và xe đậu.

    Args:
        frames: Mảng numpy (N, H, W, 3) hoặc (N, H, W) trong khoảng [0, 255].
        variance_threshold: Ngưỡng phương sai tối đa để coi là vùng tĩnh thực sự.

    Returns:
        static_mask: Mảng bool (H, W), True tại các pixel tĩnh ổn định theo thời gian.
    """
    if frames.ndim == 4:
        # Chuyển về thang độ xám để tính phương sai
        gray = 0.2989 * frames[..., 0] + 0.5870 * frames[..., 1] + 0.1140 * frames[..., 2]
    else:
        gray = frames.astype(np.float32)

    var_map = np.var(gray, axis=0)
    static_mask = var_map < (variance_threshold ** 2)
    return static_mask


def estimate_background_reliability(
    origin: Union[np.ndarray, Image.Image],
    background: Union[np.ndarray, Image.Image],
    static_mask: Optional[np.ndarray] = None,
    kappa: float = 0.15
) -> Tuple[float, float]:
    """
    Tính toán chỉ số độ tin cậy r_i của ảnh nền so với ảnh gốc trên vùng tĩnh.
    Công thức theo đặc tả I.2:
        r_i = exp( - mean(Delta_static) / kappa )
    Nếu camera bị rung, đổi góc, ánh sáng chênh lệch lớn thì Delta_static tăng -> r_i tiến về 0.

    Args:
        origin: Ảnh gốc (H, W, 3) dạng uint8 [0, 255] hoặc float [0, 1].
        background: Ảnh nền cùng kích thước.
        static_mask: Mặt nạ bool (H, W) chỉ định vùng tĩnh. Nếu None, dùng 20% rìa trên ảnh (bầu trời/tòa nhà).
        kappa: Hệ số tỷ lệ độ nhạy nhiệt độ (mặc định 0.15 cho dải [0, 1]).

    Returns:
        r_i: Độ tin cậy trong khoảng [0, 1].
        delta_static: Giá trị sai khác trung bình trên vùng tĩnh.
    """
    if isinstance(origin, Image.Image):
        origin = np.array(origin.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(origin, np.ndarray) and origin.dtype == np.uint8:
        origin = origin.astype(np.float32) / 255.0

    if isinstance(background, Image.Image):
        background = np.array(background.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(background, np.ndarray) and background.dtype == np.uint8:
        background = background.astype(np.float32) / 255.0

    H, W = origin.shape[:2]

    # Nếu không có static_mask, mặc định lấy 20% hàng phía trên (bầu trời/tòa nhà cao tầng)
    if static_mask is None:
        static_mask = np.zeros((H, W), dtype=bool)
        top_crop = max(1, int(H * 0.20))
        static_mask[:top_crop, :] = True

    # Tính sai khác L1 màu trung bình trên vùng tĩnh
    delta_pixel = np.mean(np.abs(origin - background), axis=-1)  # (H, W)
    if np.sum(static_mask) == 0:
        delta_static = float(np.mean(delta_pixel))
    else:
        delta_static = float(np.mean(delta_pixel[static_mask]))

    r_i = float(np.exp(-delta_static / max(kappa, 1e-6)))
    r_i = max(0.0, min(1.0, r_i))
    return r_i, delta_static


def check_camera_alignment_phase_correlation(
    origin: Union[np.ndarray, Image.Image],
    background: Union[np.ndarray, Image.Image],
    static_mask: Optional[np.ndarray] = None,
    max_allowed_shift: float = 4.0
) -> Tuple[float, float, bool]:
    """
    Phát hiện độ dịch chuyển hình học (camera shift/vibration) giữa ảnh gốc và ảnh nền
    sử dụng Phase Correlation Fourier 2D trên vùng tĩnh (static region).

    Args:
        origin: Ảnh gốc (H, W, 3).
        background: Ảnh nền (H, W, 3).
        static_mask: Mặt nạ vùng tĩnh (H, W).
        max_allowed_shift: Ngưỡng dịch tối đa (pixel) để coi là align_ok (mặc định 4.0 px).

    Returns:
        dx, dy: Độ dịch chuyển ước lượng theo trục x và y (pixel).
        align_ok: True nếu độ dịch <= max_allowed_shift, False nếu camera bị xô lệch.
    """
    if isinstance(origin, Image.Image):
        origin = np.array(origin.convert("L"), dtype=np.float32)
    elif isinstance(origin, np.ndarray) and origin.ndim == 3:
        origin = 0.2989 * origin[..., 0] + 0.5870 * origin[..., 1] + 0.1140 * origin[..., 2]

    if isinstance(background, Image.Image):
        background = np.array(background.convert("L"), dtype=np.float32)
    elif isinstance(background, np.ndarray) and background.ndim == 3:
        background = 0.2989 * background[..., 0] + 0.5870 * background[..., 1] + 0.1140 * background[..., 2]

    H, W = origin.shape
    if static_mask is not None:
        # Áp mặt nạ cửa sổ Hann trên vùng tĩnh
        window = static_mask.astype(np.float32)
    else:
        window = np.ones((H, W), dtype=np.float32)
        top_crop = max(1, int(H * 0.25))
        window[top_crop:, :] = 0.0

    img1 = (origin - np.mean(origin)) * window
    img2 = (background - np.mean(background)) * window

    # Biến đổi Fourier 2D
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)
    cross_power = (F1 * np.conj(F2)) / (np.abs(F1 * np.conj(F2)) + 1e-8)
    corr = np.real(np.fft.ifft2(cross_power))

    # Tìm vị trí đỉnh cực đại
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    if peak_y > H // 2:
        peak_y -= H
    if peak_x > W // 2:
        peak_x -= W

    dx, dy = float(peak_x), float(peak_y)
    shift_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    align_ok = bool(shift_magnitude <= max_allowed_shift)
    return dx, dy, align_ok


class StaticRegionReliabilityEstimator:
    """Wrapper hướng đối tượng cho bài toán ước lượng độ tin cậy r_i."""
    def __init__(self, temporal_variance_threshold: float = 0.01, kappa: float = 0.15):
        self.var_threshold = temporal_variance_threshold
        self.kappa = kappa

    def compute_reliability_score(
        self,
        sequence: Union[torch.Tensor, np.ndarray],
        background: Union[torch.Tensor, np.ndarray],
        static_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> float:
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()
        if isinstance(background, torch.Tensor):
            background = background.cpu().numpy()
        if isinstance(static_mask, torch.Tensor):
            static_mask = static_mask.cpu().numpy() > 0.5

        if sequence.ndim == 4 and sequence.shape[1] == 3:  # (T, 3, H, W)
            sequence = np.transpose(sequence, (0, 2, 3, 1))
        if background.ndim == 3 and background.shape[0] == 3:  # (3, H, W)
            background = np.transpose(background, (1, 2, 0))

        # Lấy frame đầu hoặc trung bình frame
        ref_frame = sequence[0]
        r_i, _ = estimate_background_reliability(ref_frame, background, static_mask, kappa=self.kappa)
        return float(r_i)


class CameraAlignmentChecker:
    """Wrapper hướng đối tượng cho Phase Correlation kiểm tra camera xô lệch."""
    def __init__(self, shift_threshold_px: float = 4.0):
        self.shift_threshold = shift_threshold_px

    def estimate_camera_shift(
        self,
        origin: Union[torch.Tensor, np.ndarray],
        background: Union[torch.Tensor, np.ndarray],
        static_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Tuple[float, float, bool]:
        if isinstance(origin, torch.Tensor):
            origin = origin.cpu().numpy()
        if isinstance(background, torch.Tensor):
            background = background.cpu().numpy()
        if isinstance(static_mask, torch.Tensor):
            static_mask = static_mask.cpu().numpy() > 0.5

        if origin.ndim == 3 and origin.shape[0] == 3:
            origin = np.transpose(origin, (1, 2, 0))
        if background.ndim == 3 and background.shape[0] == 3:
            background = np.transpose(background, (1, 2, 0))

        dx, dy, is_aligned = check_camera_alignment_phase_correlation(
            origin, background, static_mask, max_allowed_shift=self.shift_threshold
        )
        return dy, dx, is_aligned

