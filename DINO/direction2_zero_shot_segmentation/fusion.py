"""
=============================================================================
 Hướng 2: Zero-Shot Vehicle Segmentation — Mask Fusion & Boundary Refinement
 Thuật toán hợp nhất (Fusion) giữa Background Subtraction và DINOv3 PCA
 kết hợp Guided Filter để bám sát viền phương tiện sắc nét
=============================================================================
"""

import warnings
from typing import Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class MaskFusionEngine:
    """
    Thuật toán hợp nhất đa nguồn (Multi-source Mask Fusion):
      - Nguồn 1: $\\Delta$-Mask từ Background Subtraction (chính xác đến từng pixel chuyển động nhưng nhạy cảm với bóng đổ).
      - Nguồn 2: DINOv3 PCA Mask (hiểu ngữ nghĩa hình học vật thể sâu sắc, không bị bóng đổ, nhưng độ phân giải patch thô).
      - Cơ chế Fusion: Adaptive Weighted Intersection + Guided Bilateral Boundary Snapping.
    """

    def __init__(
        self,
        guided_filter_radius: int = 4,
        guided_filter_eps: float = 1e-3,
        morph_close_kernel: int = 7,
    ):
        self.radius = guided_filter_radius
        self.eps = guided_filter_eps
        self.morph_close_kernel = morph_close_kernel

    def refine_guided_filter(
        self,
        guidance: np.ndarray,
        raw_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Áp dụng Guided Filter bảo toàn cạnh viền (Edge-preserving Smoothing).
        """
        ximgproc = getattr(cv2, "ximgproc", None) if HAS_CV2 else None
        if ximgproc is not None and hasattr(ximgproc, "guidedFilter"):
            try:
                guide_gray = cv2.cvtColor(guidance, cv2.COLOR_RGB2GRAY) if guidance.ndim == 3 else guidance
                refined = ximgproc.guidedFilter(
                    guide=guide_gray,
                    src=raw_mask.astype(np.float32),
                    radius=self.radius,
                    eps=self.eps,
                )
                return np.clip(refined, 0.0, 1.0)
            except Exception:
                pass

        # Fallback: Gaussian / Bilateral filtering
        if HAS_CV2:
            return cv2.GaussianBlur(raw_mask.astype(np.float32), (5, 5), 0)
        return raw_mask.astype(np.float32)

    def fuse(
        self,
        delta_mask: np.ndarray,
        pca_mask: np.ndarray,
        origin_img: Union[np.ndarray, Image.Image],
    ) -> np.ndarray:
        """
        Hợp nhất 2 nguồn mặt nạ tạo thành Pseudo-Mask chất lượng cao.

        Args:
            delta_mask: Mặt nạ từ trừ nền (H, W) uint8 [0, 255].
            pca_mask: Mặt nạ từ DINOv3 PC1 (H_patches, W_patches) float32 [0.0, 1.0].
            origin_img: Ảnh gốc RGB để hướng dẫn viền cạnh.

        Returns:
            refined_mask: Mặt nạ nhị phân hoàn thiện (H, W) uint8 [0, 255].
        """
        if isinstance(origin_img, Image.Image):
            guide_np = np.array(origin_img.convert("RGB"))
        else:
            guide_np = origin_img.copy()

        H, W = guide_np.shape[:2]

        # 1. Phóng đại (Upsample) PCA mask về kích thước ảnh gốc
        if HAS_CV2:
            pca_full = cv2.resize(pca_mask, (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            pca_pil = Image.fromarray((pca_mask * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
            pca_full = np.array(pca_pil).astype(np.float32) / 255.0

        pca_full = np.clip(pca_full, 0.0, 1.0)
        delta_bin = (delta_mask > 127).astype(np.float32)

        # 2. Adaptive Fusion:
        # Vùng lõi phương tiện: Cả delta và PCA đều đồng thuận cao
        # Vùng bóng đổ mặt đường: delta cao nhưng PCA thấp -> Bị triệt tiêu
        fused_raw = (0.6 * delta_bin + 0.4 * pca_full) * (pca_full > 0.35).astype(np.float32)

        # 3. Làm mịn và bám viền vật thể bằng Guided Filter
        fused_smooth = self.refine_guided_filter(guide_np, fused_raw)

        # 4. Ngưỡng hóa và lọc hình thái học
        final_bin = (fused_smooth > 0.45).astype(np.uint8) * 255

        if HAS_CV2 and self.morph_close_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_close_kernel, self.morph_close_kernel))
            final_bin = cv2.morphologyEx(final_bin, cv2.MORPH_CLOSE, k)

        return final_bin
