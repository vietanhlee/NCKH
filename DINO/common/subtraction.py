"""
=============================================================================
 Common Utility: BackgroundSubtractor
 Các thuật toán trừ nền vật lý, trích xuất pseudo-mask và tính toán
 phân bố trọng số foreground cho Vision Transformer & DINO
=============================================================================
"""

import warnings
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn(
        "OpenCV (cv2) chưa được cài đặt. Hệ thống sẽ tự động sử dụng bộ xử lý thay thế bằng NumPy/PIL. "
        "Khuyến nghị cài đặt: pip install opencv-python để tối ưu hóa hiệu năng tính toán morphology."
    )


class BackgroundSubtractor:
    """
    Bộ công cụ trừ nền tĩnh và tinh chỉnh hình thái học (morphological filtering)
    nhằm khử bóng đổ (shadow suppression), triệt tiêu nhiễu rung lắc camera
    và cô lập phương tiện giao thông (Foreground) phục vụ SSL.
    """

    def __init__(
        self,
        color_space: str = "lab",
        blur_kernel: int = 5,
        morph_open_kernel: int = 5,
        morph_close_kernel: int = 11,
        min_area_ratio: float = 0.001,
    ):
        """
        Khởi tạo BackgroundSubtractor.

        Args:
            color_space: Không gian màu ('lab', 'hsv', 'rgb', 'gray'). Không gian 'lab'
                         phân tách Lightness và Chrominance giúp khử bóng đổ xe tốt nhất.
            blur_kernel: Kích thước kernel Gaussian lọc nhiễu cảm biến (phải là số lẻ).
            morph_open_kernel: Kernel phép mở (Opening) để loại bỏ nhiễu hột tiêu.
            morph_close_kernel: Kernel phép đóng (Closing) để lấp đầy phần thân xe bị rỗng.
            min_area_ratio: Ngưỡng diện tích tối thiểu loại bỏ connected component rác.
        """
        self.color_space = color_space.lower()
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.morph_open_kernel = morph_open_kernel
        self.morph_close_kernel = morph_close_kernel
        self.min_area_ratio = min_area_ratio

    def compute_delta(
        self,
        origin_img: Union[np.ndarray, Image.Image],
        bg_img: Union[np.ndarray, Image.Image],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính bản đồ sai khác liên tục $\\Delta = |I_{\\text{origin}} - I_{\\text{bg}}|$.

        Args:
            origin_img: Ảnh gốc có phương tiện (RGB, uint8 hoặc PIL.Image).
            bg_img: Ảnh nền tĩnh tương ứng (RGB, uint8 hoặc PIL.Image).

        Returns:
            delta_norm: Bản đồ sai khác chuẩn hóa 1 kênh (H, W) trong đoạn [0.0, 1.0].
            delta_color: Bản đồ sai khác 3 kênh (H, W, 3) phục vụ hiển thị trực quan.
        """
        # Chuyển đổi định dạng sang ndarray nếu là PIL
        if isinstance(origin_img, Image.Image):
            origin_np = np.array(origin_img.convert("RGB"))
        else:
            origin_np = origin_img.copy()

        if isinstance(bg_img, Image.Image):
            bg_np = np.array(bg_img.convert("RGB"))
        else:
            bg_np = bg_img.copy()

        # Đảm bảo 2 ảnh cùng kích thước tuyệt đối
        if origin_np.shape[:2] != bg_np.shape[:2]:
            h, w = origin_np.shape[:2]
            if HAS_CV2:
                bg_np = cv2.resize(bg_np, (w, h), interpolation=cv2.INTER_CUBIC)
            else:
                bg_pil = Image.fromarray(bg_np).resize((w, h), Image.BICUBIC)
                bg_np = np.array(bg_pil)

        # Tính toán sai khác màu 3 kênh
        delta_color = np.abs(origin_np.astype(np.float32) - bg_np.astype(np.float32)) / 255.0

        if HAS_CV2:
            # 1. Khử nhiễu cục bộ bằng Gaussian Blur
            if self.blur_kernel > 1:
                origin_blur = cv2.GaussianBlur(origin_np, (self.blur_kernel, self.blur_kernel), 0)
                bg_blur = cv2.GaussianBlur(bg_np, (self.blur_kernel, self.blur_kernel), 0)
            else:
                origin_blur, bg_blur = origin_np, bg_np

            # 2. Xử lý theo không gian màu được lựa chọn
            if self.color_space == "lab":
                origin_lab = cv2.cvtColor(origin_blur, cv2.COLOR_RGB2LAB).astype(np.float32)
                bg_lab = cv2.cvtColor(bg_blur, cv2.COLOR_RGB2LAB).astype(np.float32)
                # Kênh L (Lightness: 0..255)
                delta_l = np.abs(origin_lab[:, :, 0] - bg_lab[:, :, 0]) / 255.0
                # Kênh a và b (Màu sắc)
                delta_a = np.abs(origin_lab[:, :, 1] - bg_lab[:, :, 1]) / 255.0
                delta_b = np.abs(origin_lab[:, :, 2] - bg_lab[:, :, 2]) / 255.0
                # Công thức kết hợp khử bóng: Giảm trọng số của L (nơi bóng đổ biến đổi)
                # và tăng trọng số của chromatic channels a, b
                delta_norm = 0.4 * delta_l + 0.3 * delta_a + 0.3 * delta_b

            elif self.color_space == "hsv":
                origin_hsv = cv2.cvtColor(origin_blur, cv2.COLOR_RGB2HSV).astype(np.float32)
                bg_hsv = cv2.cvtColor(bg_blur, cv2.COLOR_RGB2HSV).astype(np.float32)
                delta_v = np.abs(origin_hsv[:, :, 2] - bg_hsv[:, :, 2]) / 255.0
                delta_s = np.abs(origin_hsv[:, :, 1] - bg_hsv[:, :, 1]) / 255.0
                delta_norm = 0.6 * delta_v + 0.4 * delta_s

            elif self.color_space == "gray":
                origin_gray = cv2.cvtColor(origin_blur, cv2.COLOR_RGB2GRAY).astype(np.float32)
                bg_gray = cv2.cvtColor(bg_blur, cv2.COLOR_RGB2GRAY).astype(np.float32)
                delta_norm = np.abs(origin_gray - bg_gray) / 255.0

            else:  # RGB
                diff = np.abs(origin_blur.astype(np.float32) - bg_blur.astype(np.float32))
                delta_norm = np.mean(diff, axis=2) / 255.0

        else:
            # Fallback pure NumPy
            diff = np.abs(origin_np.astype(np.float32) - bg_np.astype(np.float32))
            delta_norm = np.mean(diff, axis=2) / 255.0

        delta_norm = np.clip(delta_norm, 0.0, 1.0).astype(np.float32)
        return delta_norm, delta_color.astype(np.float32)

    def extract_binary_mask(
        self,
        delta_norm: np.ndarray,
        method: str = "otsu",
        fixed_thresh: float = 0.15,
    ) -> np.ndarray:
        """
        Chuyển đổi bản đồ liên tục $\\Delta$ thành nhị phân (Binary Mask) thông qua Otsu
        và các bộ lọc hình thái học (Morphology Operations).

        Args:
            delta_norm: Bản đồ sai khác (H, W) trong [0.0, 1.0].
            method: 'otsu' (tự động thích ứng) hoặc 'fixed' (theo fixed_thresh).
            fixed_thresh: Ngưỡng cố định nếu method='fixed'.

        Returns:
            binary_mask: Mặt nạ nhị phân (H, W) kiểu uint8 với giá trị {0, 255}.
        """
        delta_uint8 = (np.clip(delta_norm, 0.0, 1.0) * 255).astype(np.uint8)

        if HAS_CV2:
            if method == "otsu":
                _, binary = cv2.threshold(delta_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                thr_val = int(fixed_thresh * 255)
                _, binary = cv2.threshold(delta_uint8, thr_val, 255, cv2.THRESH_BINARY)

            # 1. Phép mở (Opening: Erode rồi Dilate) - Quét sạch hạt bụi, nhiễu lá cây
            if self.morph_open_kernel > 1:
                k_open = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.morph_open_kernel, self.morph_open_kernel)
                )
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

            # 2. Phép đóng (Closing: Dilate rồi Erode) - Nối liền thân xe, kính xe
            if self.morph_close_kernel > 1:
                k_close = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.morph_close_kernel, self.morph_close_kernel)
                )
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

            # 3. Lọc bỏ các cụm điểm kết nối (Connected Components) có diện tích quá bé
            if self.min_area_ratio > 0:
                h, w = binary.shape
                min_pixels = int(h * w * self.min_area_ratio)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                for label_id in range(1, num_labels):
                    area = stats[label_id, cv2.CC_STAT_AREA]
                    if area < min_pixels:
                        binary[labels == label_id] = 0

            return binary
        else:
            # Fallback thuần NumPy
            thr = 0.15 if method == "otsu" else fixed_thresh
            binary = (delta_norm > thr).astype(np.uint8) * 255
            return binary

    def compute_patch_mask_weights(
        self,
        delta_norm: np.ndarray,
        img_size: int = 224,
        patch_size: int = 16,
        alpha: float = 0.75,
    ) -> np.ndarray:
        """
        Tính toán phân bố xác suất che mờ theo từng Patch (Foreground-Aware Masking Probabilities)
        cho DINOv3 và Vision Transformer.

        Công thức:
          w_p = Mean(\\Delta[patch_p])
          P_mask(p) = \\alpha * (w_p / max(w)) + (1 - \\alpha) * (1 / N_patches)

        Args:
            delta_norm: Bản đồ $\\Delta$ (H, W).
            img_size: Kích thước đầu vào chuẩn của ViT (mặc định 224).
            patch_size: Kích thước mỗi patch (16 cho DINOv3, 14 cho DINOv2).
            alpha: Trọng số tập trung foreground (alpha=1.0 là 100% che xe, alpha=0 là hoàn toàn ngẫu nhiên).

        Returns:
            prob_matrix: Ma trận xác suất 2D (num_patches_y, num_patches_x) có tổng bằng 1.0.
        """
        h_patches = img_size // patch_size
        w_patches = img_size // patch_size

        # Resize delta về img_size x img_size
        if HAS_CV2:
            delta_resized = cv2.resize(delta_norm, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        else:
            delta_pil = Image.fromarray((delta_norm * 255).astype(np.uint8)).resize(
                (img_size, img_size), Image.BILINEAR
            )
            delta_resized = np.array(delta_pil).astype(np.float32) / 255.0

        # Tính mean delta cho từng patch ô vuông
        patch_weights = np.zeros((h_patches, w_patches), dtype=np.float32)
        for i in range(h_patches):
            for j in range(w_patches):
                region = delta_resized[
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patch_weights[i, j] = np.mean(region)

        # Chuẩn hóa trọng số
        max_w = np.max(patch_weights) + 1e-8
        norm_weights = patch_weights / max_w

        # Kết hợp thành phần uniform và foreground
        n_total = h_patches * w_patches
        uniform_weight = 1.0 / n_total

        probs = alpha * norm_weights + (1.0 - alpha) * uniform_weight
        # Chuẩn hóa về hàm mật độ xác suất (tổng = 1)
        probs = probs / np.sum(probs)

        return probs.astype(np.float32)
