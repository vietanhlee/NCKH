"""
=============================================================================
 Background-Guided Vehicle Foreground Extraction & DINOv3 PCA Segmentation
 Proof-of-Concept Pipeline for SSL Traffic Research

 Implements Phase 1 of the research roadmap:
   1. Background Subtraction → Pseudo Foreground Mask (Δ map)
   2. DINOv3 PCA Patch Feature Extraction → Semantic FG/BG Separation
   3. Combined Refinement → Publication-quality Pseudo Segmentation Map
   4. Quality Metrics & Visualization Artifacts

 Input Requirements:
   - Background images: clean road images (no vehicles) from fixed CCTV cameras
   - Origin images: same camera viewpoint WITH vehicles
   - Both images must share the same viewpoint (static camera assumption)

 Usage:
   python DINO/bg_subtract_poc.py \
       --bg_dir path/to/backgrounds \
       --origin_dir path/to/origins \
       --output_dir checkpoints/bg_subtract_poc \
       --backbone dinov3_vits16 \
       --device cuda

 Output:
   - Δ maps (foreground difference maps)
   - Pseudo binary masks (morphologically refined)
   - DINOv3 PCA semantic maps (3-component RGB visualization)
   - Combined refined masks (Δ ∩ PCA)
   - Quality analysis report (JSON + Markdown)
   - Publication-ready figure grids (PDF + PNG)

 References:
   - Caron et al. "Emerging Properties in Self-Supervised Vision Transformers" (ICCV 2021)
   - Oquab et al. "DINOv2: Learning Robust Visual Features" (TMLR 2024)
   - Meta AI: "DINOv3: Self-Supervised Vision Transformers with RoPE" (2024)
=============================================================================
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend cho server/Kaggle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV (cv2) not installed. Falling back to PIL-based processing. "
                   "Install with: pip install opencv-python")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm


# =====================================================================
# 1. ĐỐI SÁNH BACKGROUND-ORIGIN: TÌM CẶP ẢNH TƯƠNG ỨNG
# =====================================================================

_dino_dir = os.path.dirname(os.path.abspath(__file__))
if _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)
from common.matcher import TrafficPairMatcher

def discover_image_pairs(
    bg_dir: str,
    origin_dir: str,
    match_strategy: str = "route_hourly",
    camera_id_regex: str = r"^(\d+)_",
) -> List[Dict[str, str]]:
    """
    Tự động đối sánh (match) ảnh background với ảnh origin dựa trên TrafficPairMatcher.
    Đồng bộ chuẩn 100% với múi giờ Việt Nam UTC+7 và khoảng cách chu kỳ ngày-đêm 24 giờ.
    """
    matcher = TrafficPairMatcher(
        bg_dir=bg_dir,
        origin_dir=origin_dir,
        match_strategy=match_strategy,
        timezone_offset_hours=7,
        camera_regex=camera_id_regex,
    )
    pairs = matcher.discover_pairs()

    res = []
    for p in pairs:
        res.append({
            "bg_path": p["bg_path"],
            "origin_path": p["origin_path"],
            "camera_id": str(p["route_id"]),
            "origin_name": p["origin_name"],
            "bg_hour": p.get("bg_hour", -1),
            "origin_hour": p.get("origin_hour", -1),
        })

    print(f"📊 [Pair Discovery] Tìm thấy {len(res)} cặp ảnh hợp lệ (strategy='{match_strategy}')")
    return res


# =====================================================================
# 2. BACKGROUND SUBTRACTION & MORPHOLOGICAL REFINEMENT
# =====================================================================

def compute_delta_map(
    bg_img: np.ndarray,
    origin_img: np.ndarray,
    color_space: str = "lab",
    blur_kernel: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính bản đồ khác biệt Δ = |origin - background| với các kỹ thuật
    triệt tiêu nhiễu ánh sáng và bóng đổ.

    Args:
        bg_img: Ảnh background (H, W, 3) uint8 RGB.
        origin_img: Ảnh origin (H, W, 3) uint8 RGB.
        color_space: Không gian màu cho phép trừ:
            - 'rgb': Trừ trực tiếp trên RGB (nhạy với ánh sáng).
            - 'lab': Chuyển sang LAB, trừ trên kênh L (ít nhạy bóng đổ).
            - 'hsv': Chuyển sang HSV, trừ trên kênh V (Value).
            - 'gray': Chuyển sang grayscale rồi trừ.
        blur_kernel: Kích thước kernel Gaussian blur trước khi trừ (giảm nhiễu).

    Returns:
        delta_raw: Bản đồ Δ thô (H, W) float32 trong [0, 1].
        delta_color: Bản đồ Δ 3 kênh (H, W, 3) float32 (cho visualization).
    """
    assert bg_img.shape == origin_img.shape, \
        f"Kích thước ảnh không khớp: bg={bg_img.shape} vs origin={origin_img.shape}"

    if HAS_CV2:
        # Gaussian blur để giảm nhiễu cảm biến trước khi trừ
        if blur_kernel > 0:
            bg_blur = cv2.GaussianBlur(bg_img, (blur_kernel, blur_kernel), 0)
            origin_blur = cv2.GaussianBlur(origin_img, (blur_kernel, blur_kernel), 0)
        else:
            bg_blur, origin_blur = bg_img, origin_img

        if color_space == "lab":
            bg_lab = cv2.cvtColor(bg_blur, cv2.COLOR_RGB2LAB).astype(np.float32)
            origin_lab = cv2.cvtColor(origin_blur, cv2.COLOR_RGB2LAB).astype(np.float32)
            # Kênh L (Lightness) ít nhạy bóng đổ hơn RGB
            delta_l = np.abs(origin_lab[:, :, 0] - bg_lab[:, :, 0]) / 255.0
            # Kênh A, B (chrominance) bắt sự thay đổi màu sắc
            delta_a = np.abs(origin_lab[:, :, 1] - bg_lab[:, :, 1]) / 255.0
            delta_b = np.abs(origin_lab[:, :, 2] - bg_lab[:, :, 2]) / 255.0
            # Kết hợp: ưu tiên lightness nhưng bao gồm cả color change
            delta_raw = np.clip(0.5 * delta_l + 0.25 * delta_a + 0.25 * delta_b, 0, 1)

        elif color_space == "hsv":
            bg_hsv = cv2.cvtColor(bg_blur, cv2.COLOR_RGB2HSV).astype(np.float32)
            origin_hsv = cv2.cvtColor(origin_blur, cv2.COLOR_RGB2HSV).astype(np.float32)
            delta_raw = np.abs(origin_hsv[:, :, 2] - bg_hsv[:, :, 2]) / 255.0

        elif color_space == "gray":
            bg_gray = cv2.cvtColor(bg_blur, cv2.COLOR_RGB2GRAY).astype(np.float32)
            origin_gray = cv2.cvtColor(origin_blur, cv2.COLOR_RGB2GRAY).astype(np.float32)
            delta_raw = np.abs(origin_gray - bg_gray) / 255.0

        else:  # rgb
            diff = np.abs(origin_blur.astype(np.float32) - bg_blur.astype(np.float32))
            delta_raw = np.mean(diff, axis=2) / 255.0

    else:
        # Fallback: PIL-based simple RGB difference
        diff = np.abs(origin_img.astype(np.float32) - bg_img.astype(np.float32))
        delta_raw = np.mean(diff, axis=2) / 255.0

    # Color difference map (cho visualization)
    delta_color = np.abs(origin_img.astype(np.float32) - bg_img.astype(np.float32)) / 255.0

    return delta_raw.astype(np.float32), delta_color.astype(np.float32)


def refine_to_binary_mask(
    delta_map: np.ndarray,
    threshold: float = 0.0,
    otsu: bool = True,
    morph_open_kernel: int = 5,
    morph_close_kernel: int = 11,
    min_area_ratio: float = 0.001,
) -> np.ndarray:
    """
    Chuyển Δ map liên tục sang binary mask nhị phân qua thresholding + morphology.

    Args:
        delta_map: Bản đồ Δ (H, W) float32 trong [0, 1].
        threshold: Ngưỡng cố định (0 = tự động Otsu).
        otsu: Sử dụng Otsu's thresholding tự động.
        morph_open_kernel: Kernel opening (loại bỏ nhiễu nhỏ).
        morph_close_kernel: Kernel closing (lấp đầy lỗ hổng trong xe).
        min_area_ratio: Loại bỏ connected component có diện tích < ratio * tổng diện tích ảnh.

    Returns:
        binary_mask: Mask nhị phân (H, W) uint8 với 0=background, 255=foreground.
    """
    # Chuyển sang uint8 cho OpenCV
    delta_uint8 = (np.clip(delta_map, 0, 1) * 255).astype(np.uint8)

    if HAS_CV2:
        if otsu and threshold <= 0:
            _, binary = cv2.threshold(delta_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thr = int(threshold * 255) if threshold <= 1.0 else int(threshold)
            _, binary = cv2.threshold(delta_uint8, thr, 255, cv2.THRESH_BINARY)

        # Morphological operations: Open (loại nhiễu) → Close (lấp lỗ)
        if morph_open_kernel > 0:
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_kernel, morph_open_kernel))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

        if morph_close_kernel > 0:
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_kernel, morph_close_kernel))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

        # Loại bỏ connected components quá nhỏ (noise)
        if min_area_ratio > 0:
            total_area = binary.shape[0] * binary.shape[1]
            min_area = int(total_area * min_area_ratio)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for label_id in range(1, n_labels):
                if stats[label_id, cv2.CC_STAT_AREA] < min_area:
                    binary[labels == label_id] = 0

    else:
        # Fallback: simple thresholding
        thr = threshold if threshold > 0 else 0.15
        binary = (delta_map > thr).astype(np.uint8) * 255

    return binary


# =====================================================================
# 3. DINOv3 PCA FEATURE EXTRACTION & SEMANTIC SEGMENTATION
# =====================================================================

@torch.no_grad()
def extract_dino_pca_features(
    backbone: nn.Module,
    image: Image.Image,
    device: torch.device,
    img_size: int = 224,
    patch_size: int = 16,
    n_components: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trích xuất DINOv3 patch-level features, áp dụng PCA để tách foreground/background.

    DINOv3/DINOv2 có tính chất nổi trội (emergent property): thành phần PCA thứ nhất
    tự nhiên phân biệt foreground (phương tiện) và background (đường, vỉa hè).

    Args:
        backbone: DINOv3/v2 backbone đã load.
        image: Ảnh PIL RGB.
        device: Thiết bị tính toán.
        img_size: Kích thước resize (phải chia hết cho patch_size).
        patch_size: Kích thước patch (16 cho DINOv3, 14 cho DINOv2).
        n_components: Số thành phần PCA (3 cho visualization RGB).

    Returns:
        pca_map: PCA feature map (H_patch, W_patch, n_components) float32.
        fg_mask_pca: Binary mask từ PC1 (H_patch, W_patch) float32.
        patch_tokens: Raw patch tokens (N_patches, embed_dim) float32.
    """
    backbone.eval()

    # Chuẩn hóa ảnh
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    x = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Trích xuất patch tokens (không lấy CLS token)
    # DINOv3/v2 backbone trả về CLS token mặc định, cần forward intermediate
    try:
        # Thử trích xuất qua forward_features hoặc get_intermediate_layers (DINOv2/v3 API)
        if hasattr(backbone, "get_intermediate_layers"):
            outputs = backbone.get_intermediate_layers(x, n=1)
            patch_tokens = outputs[0]  # (B, N_patches, embed_dim)
            # Loại bỏ CLS token nếu có (thường ở vị trí 0)
            if patch_tokens.shape[1] == (img_size // patch_size) ** 2 + 1:
                patch_tokens = patch_tokens[:, 1:, :]  # Bỏ CLS
        elif hasattr(backbone, "forward_features"):
            feat = backbone.forward_features(x)
            if isinstance(feat, dict):
                patch_tokens = feat.get("x_norm_patchtokens", feat.get("x_prenorm", None))
                if patch_tokens is None:
                    # Lấy toàn bộ rồi bỏ CLS
                    full = feat.get("x_norm_clstoken", list(feat.values())[0])
                    patch_tokens = full
            else:
                patch_tokens = feat
                if patch_tokens.shape[1] == (img_size // patch_size) ** 2 + 1:
                    patch_tokens = patch_tokens[:, 1:, :]
        else:
            # Fallback: forward thường, chỉ lấy CLS → không có patch-level info
            cls_token = backbone(x)
            print("⚠️  Backbone không hỗ trợ patch token extraction. "
                  "Chỉ có CLS token → PCA segmentation bị giới hạn.")
            # Tạo dummy patch map từ CLS
            n_patches_hw = img_size // patch_size
            dummy = cls_token.unsqueeze(1).expand(-1, n_patches_hw ** 2, -1)
            patch_tokens = dummy

    except Exception as e:
        print(f"⚠️  Lỗi trích xuất patch tokens: {e}. Sử dụng CLS fallback.")
        cls_token = backbone(x)
        n_patches_hw = img_size // patch_size
        patch_tokens = cls_token.unsqueeze(1).expand(-1, n_patches_hw ** 2, -1)

    tokens = patch_tokens[0].cpu().numpy()  # (N_patches, embed_dim)
    n_patches = tokens.shape[0]
    n_patches_hw = int(np.sqrt(n_patches))

    # PCA trên patch tokens
    from sklearn.decomposition import PCA
    tokens_centered = tokens - tokens.mean(axis=0, keepdims=True)
    pca = PCA(n_components=min(n_components, tokens.shape[1]))
    pca_result = pca.fit_transform(tokens_centered)  # (N_patches, n_components)

    # Reshape thành spatial map
    pca_map = pca_result.reshape(n_patches_hw, n_patches_hw, -1)

    # PC1 (thành phần chính thứ nhất) thường tách FG/BG
    pc1 = pca_map[:, :, 0]
    # Xác định hướng: foreground thường có giá trị cao hơn (hoặc thấp hơn)
    # Heuristic: foreground ở trung tâm → check giá trị center vs border
    center_val = pc1[n_patches_hw // 4: 3 * n_patches_hw // 4,
                     n_patches_hw // 4: 3 * n_patches_hw // 4].mean()
    border_val = np.concatenate([pc1[0, :], pc1[-1, :], pc1[:, 0], pc1[:, -1]]).mean()

    if center_val < border_val:
        pc1 = -pc1  # Đảo ngược nếu FG có giá trị thấp hơn

    # Chuẩn hóa và threshold
    pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    fg_mask_pca = (pc1_norm > 0.5).astype(np.float32)

    # Chuẩn hóa PCA map cho visualization (mỗi component → 0..1)
    for c in range(pca_map.shape[2]):
        comp = pca_map[:, :, c]
        pca_map[:, :, c] = (comp - comp.min()) / (comp.max() - comp.min() + 1e-8)

    return pca_map.astype(np.float32), fg_mask_pca, tokens


# =====================================================================
# 4. COMBINED REFINEMENT: Δ ∩ PCA
# =====================================================================

def combine_masks(
    delta_mask: np.ndarray,
    pca_mask: np.ndarray,
    origin_shape: Tuple[int, int],
    strategy: str = "intersection",
) -> np.ndarray:
    """
    Kết hợp pseudo mask từ background subtraction và DINOv3 PCA.

    Args:
        delta_mask: Binary mask từ Δ map (H, W) uint8.
        pca_mask: Binary mask từ PCA PC1 (H_patch, W_patch) float32.
        origin_shape: Kích thước ảnh gốc (H, W).
        strategy: 'intersection' (AND), 'union' (OR), 'pca_guided' (Δ refined bởi PCA).

    Returns:
        combined_mask: Binary mask kết hợp (H, W) uint8.
    """
    H, W = origin_shape[:2]

    # Upsample PCA mask về kích thước ảnh gốc
    pca_upsampled = np.array(
        Image.fromarray((pca_mask * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    )

    # Đảm bảo delta_mask cùng kích thước
    if delta_mask.shape[:2] != (H, W):
        delta_mask = np.array(
            Image.fromarray(delta_mask).resize((W, H), Image.NEAREST)
        )

    delta_bin = (delta_mask > 127).astype(np.uint8)
    pca_bin = (pca_upsampled > 127).astype(np.uint8)

    if strategy == "intersection":
        combined = (delta_bin & pca_bin) * 255
    elif strategy == "union":
        combined = (delta_bin | pca_bin) * 255
    elif strategy == "pca_guided":
        # Δ mask giữ lại chỉ những vùng PCA cũng đồng ý là foreground
        # Nhưng cho phép Δ mở rộng nhẹ (dilate PCA trước khi AND)
        if HAS_CV2:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            pca_dilated = cv2.dilate(pca_bin * 255, k, iterations=1)
            combined = (delta_bin & (pca_dilated > 127).astype(np.uint8)) * 255
        else:
            combined = (delta_bin & pca_bin) * 255
    else:
        raise ValueError(f"strategy không hợp lệ: {strategy}")

    return combined.astype(np.uint8)


# =====================================================================
# 5. PATCH-LEVEL Δ MAP CHO FOREGROUND-AWARE MASKING (HƯỚNG 1)
# =====================================================================

def compute_patch_level_delta(
    delta_map: np.ndarray,
    patch_size: int = 16,
    img_size: int = 224,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Tính xác suất masking theo patch cho Foreground-Aware Masking (FAM).

    Công thức:
        w_p = mean(Δ[patch_p])
        P_mask(p) = α · (w_p / max(w)) + (1 - α) · (1 / N_patches)

    Args:
        delta_map: Bản đồ Δ (H, W) float32 trong [0, 1].
        patch_size: Kích thước patch (16 cho DINOv3).
        img_size: Kích thước ảnh resize.
        alpha: Tỷ lệ foreground-bias (0.7 = 70% foreground-biased, 30% random).

    Returns:
        mask_probs: Xác suất mask cho mỗi patch (n_patches_hw, n_patches_hw) float32.
    """
    # Resize delta map về img_size
    if delta_map.shape[0] != img_size or delta_map.shape[1] != img_size:
        delta_resized = np.array(
            Image.fromarray((delta_map * 255).astype(np.uint8)).resize(
                (img_size, img_size), Image.BILINEAR
            )
        ).astype(np.float32) / 255.0
    else:
        delta_resized = delta_map

    n_patches_hw = img_size // patch_size
    patch_weights = np.zeros((n_patches_hw, n_patches_hw), dtype=np.float32)

    for i in range(n_patches_hw):
        for j in range(n_patches_hw):
            patch_region = delta_resized[
                i * patch_size: (i + 1) * patch_size,
                j * patch_size: (j + 1) * patch_size
            ]
            patch_weights[i, j] = patch_region.mean()

    # Chuẩn hóa và tính xác suất
    max_w = patch_weights.max() + 1e-8
    n_total = n_patches_hw * n_patches_hw
    uniform_prob = 1.0 / n_total

    mask_probs = alpha * (patch_weights / max_w) + (1 - alpha) * uniform_prob

    # Normalize tổng xác suất = 1 (dùng cho np.random.choice)
    mask_probs = mask_probs / (mask_probs.sum() + 1e-8)

    return mask_probs


# =====================================================================
# 6. METRICS & QUALITY ANALYSIS
# =====================================================================

def compute_mask_quality_metrics(
    delta_map: np.ndarray,
    binary_mask: np.ndarray,
    pca_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Tính các chỉ số chất lượng của pseudo foreground mask.

    Returns:
        Dict chứa: foreground_ratio, mean_delta_fg, mean_delta_bg, fg_bg_contrast,
                    mask_compactness, pca_agreement (nếu có PCA mask).
    """
    mask_bin = (binary_mask > 127).astype(bool) if binary_mask.max() > 1 else binary_mask.astype(bool)
    total_pixels = mask_bin.size
    fg_pixels = mask_bin.sum()
    bg_pixels = total_pixels - fg_pixels

    metrics = {
        "foreground_ratio": float(fg_pixels / total_pixels),
        "foreground_pixels": int(fg_pixels),
        "background_pixels": int(bg_pixels),
    }

    # Resize delta_map nếu cần
    if delta_map.shape != mask_bin.shape:
        delta_resized = np.array(
            Image.fromarray((delta_map * 255).astype(np.uint8)).resize(
                (mask_bin.shape[1], mask_bin.shape[0]), Image.BILINEAR
            )
        ).astype(np.float32) / 255.0
    else:
        delta_resized = delta_map

    if fg_pixels > 0:
        metrics["mean_delta_foreground"] = float(delta_resized[mask_bin].mean())
    else:
        metrics["mean_delta_foreground"] = 0.0

    if bg_pixels > 0:
        metrics["mean_delta_background"] = float(delta_resized[~mask_bin].mean())
    else:
        metrics["mean_delta_background"] = 0.0

    # Foreground-Background Contrast (càng cao càng tốt)
    metrics["fg_bg_contrast"] = abs(
        metrics["mean_delta_foreground"] - metrics["mean_delta_background"]
    )

    # PCA agreement (nếu có)
    if pca_mask is not None:
        pca_bin = (pca_mask > 0.5).astype(bool)
        # Resize nếu cần
        if pca_bin.shape != mask_bin.shape:
            pca_resized = np.array(
                Image.fromarray((pca_mask * 255).astype(np.uint8)).resize(
                    (mask_bin.shape[1], mask_bin.shape[0]), Image.NEAREST
                )
            ) > 127
        else:
            pca_resized = pca_bin

        intersection = (mask_bin & pca_resized).sum()
        union = (mask_bin | pca_resized).sum()
        metrics["pca_iou"] = float(intersection / (union + 1e-8))

    return metrics


# =====================================================================
# 7. PUBLICATION VISUALIZATION
# =====================================================================

def plot_analysis_grid(
    origin_img: np.ndarray,
    bg_img: np.ndarray,
    delta_map: np.ndarray,
    delta_color: np.ndarray,
    binary_mask: np.ndarray,
    pca_map: Optional[np.ndarray],
    pca_fg_mask: Optional[np.ndarray],
    combined_mask: Optional[np.ndarray],
    patch_probs: Optional[np.ndarray],
    metrics: Dict[str, float],
    camera_id: str,
    save_path: str,
):
    """
    Tạo figure grid 3×3 hoặc 2×4 chất lượng cao cho publication.
    """
    has_pca = pca_map is not None and pca_fg_mask is not None

    if has_pca:
        fig = plt.figure(figsize=(20, 15), dpi=150)
        gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.25)
    else:
        fig = plt.figure(figsize=(16, 8), dpi=150)
        gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.25)

    # --- Row 1: Input Images ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(bg_img)
    ax1.set_title(f"Background (Camera {camera_id})", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(origin_img)
    ax2.set_title("Origin (with Vehicles)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(delta_color)
    ax3.set_title("Δ Color Difference", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # --- Row 2: Processing Results ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(delta_map, cmap="hot", vmin=0, vmax=1)
    ax4.set_title(f"Δ Grayscale (FG ratio: {metrics.get('foreground_ratio', 0):.1%})",
                  fontsize=11, fontweight="bold")
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(binary_mask, cmap="gray")
    ax5.set_title(f"Binary Mask (Otsu + Morph)\nContrast: {metrics.get('fg_bg_contrast', 0):.3f}",
                  fontsize=11, fontweight="bold")
    ax5.axis("off")

    if has_pca:
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(pca_map[:, :, :3])
        ax6.set_title("DINOv3 PCA (PC1=R, PC2=G, PC3=B)", fontsize=11, fontweight="bold")
        ax6.axis("off")

        # --- Row 3: PCA Mask, Combined, Patch Probs ---
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.imshow(pca_fg_mask, cmap="gray")
        iou_str = f" | IoU: {metrics.get('pca_iou', 0):.3f}" if "pca_iou" in metrics else ""
        ax7.set_title(f"PCA FG Mask (PC1 Threshold){iou_str}", fontsize=11, fontweight="bold")
        ax7.axis("off")

        ax8 = fig.add_subplot(gs[2, 1])
        if combined_mask is not None:
            ax8.imshow(combined_mask, cmap="gray")
            ax8.set_title("Combined (Δ ∩ PCA)", fontsize=11, fontweight="bold")
        else:
            # Overlay mask trên ảnh gốc
            overlay = origin_img.copy()
            mask_resized = np.array(
                Image.fromarray(binary_mask).resize(
                    (origin_img.shape[1], origin_img.shape[0]), Image.NEAREST)
            )
            overlay[mask_resized > 127] = (
                overlay[mask_resized > 127] * 0.5 +
                np.array([255, 0, 0], dtype=np.float32) * 0.5
            ).astype(np.uint8)
            ax8.imshow(overlay)
            ax8.set_title("Overlay (Red = Foreground)", fontsize=11, fontweight="bold")
        ax8.axis("off")

        ax9 = fig.add_subplot(gs[2, 2])
        if patch_probs is not None:
            im = ax9.imshow(patch_probs, cmap="YlOrRd", vmin=0)
            ax9.set_title("Patch Mask Probability\n(for BG-Guided DINO FAM)",
                          fontsize=11, fontweight="bold")
            plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)
        else:
            ax9.axis("off")
        ax9.axis("off") if patch_probs is None else None

    else:
        ax6 = fig.add_subplot(gs[1, 2])
        # Overlay mask trên ảnh gốc
        overlay = origin_img.copy()
        mask_resized = np.array(
            Image.fromarray(binary_mask).resize(
                (origin_img.shape[1], origin_img.shape[0]), Image.NEAREST)
        )
        overlay[mask_resized > 127] = (
            overlay[mask_resized > 127] * 0.5 +
            np.array([255, 0, 0], dtype=np.float32) * 0.5
        ).astype(np.uint8)
        ax6.imshow(overlay)
        ax6.set_title("Overlay (Red = Foreground)", fontsize=11, fontweight="bold")
        ax6.axis("off")

    plt.suptitle(
        f"Background-Guided Vehicle Foreground Extraction — Camera {camera_id}",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Lưu
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


# =====================================================================
# 8. MAIN PIPELINE
# =====================================================================

def run_poc_pipeline(args):
    """Main Proof-of-Concept pipeline."""
    print("\n" + "=" * 78)
    print(" 🔬 BACKGROUND-GUIDED VEHICLE FOREGROUND EXTRACTION — PoC Pipeline")
    print("=" * 78)
    print(f" Background Dir     : {args.bg_dir}")
    print(f" Origin Dir         : {args.origin_dir}")
    print(f" Output Dir         : {args.output_dir}")
    print(f" Match Strategy     : {args.match_strategy}")
    print(f" Color Space        : {args.color_space}")
    print(f" DINOv3 Backbone    : {args.backbone}")
    print(f" Use PCA Refinement : {args.use_pca}")
    print(f" Max Samples        : {args.max_samples}")
    print("=" * 78)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Tìm cặp ảnh
    pairs = discover_image_pairs(
        bg_dir=args.bg_dir,
        origin_dir=args.origin_dir,
        match_strategy=args.match_strategy,
        camera_id_regex=args.camera_regex,
    )

    if not pairs:
        print("❌ Không tìm được cặp ảnh nào. Thoát.")
        return

    # Giới hạn số lượng mẫu
    if args.max_samples > 0 and len(pairs) > args.max_samples:
        import random
        random.seed(42)
        pairs = random.sample(pairs, args.max_samples)
        print(f"   📋 Giới hạn: xử lý {len(pairs)} cặp mẫu.")

    # 2. Load DINOv3 backbone (nếu dùng PCA refinement)
    backbone = None
    device = torch.device("cpu")
    if args.use_pca:
        device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from train_ssl_dinov3 import build_backbone
            backbone, embed_dim = build_backbone(
                model_name=args.backbone,
                pretrained=True,
                weights_path=args.weights,
            )
            backbone = backbone.to(device).eval()
            print(f"✅ [DINOv3] Loaded backbone: {args.backbone} (embed_dim={embed_dim}) on {device}")

            # Detect patch size
            is_patch16 = "16" in args.backbone or "dinov3" in args.backbone.lower()
            patch_size = 16 if is_patch16 else 14
        except Exception as e:
            print(f"⚠️  [DINOv3] Không load được backbone: {e}. Tiếp tục không có PCA refinement.")
            backbone = None
            args.use_pca = False
            patch_size = 16
    else:
        patch_size = 16

    # 3. Xử lý từng cặp
    all_metrics = []
    vis_dir = os.path.join(args.output_dir, "visualizations")
    mask_dir = os.path.join(args.output_dir, "masks")
    delta_dir = os.path.join(args.output_dir, "delta_maps")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(delta_dir, exist_ok=True)

    for pair in tqdm(pairs, desc="Processing BG-Origin Pairs"):
        cam_id = pair["camera_id"]
        origin_name = pair["origin_name"]
        stem = os.path.splitext(origin_name)[0]

        try:
            # Load ảnh
            bg_pil = Image.open(pair["bg_path"]).convert("RGB")
            origin_pil = Image.open(pair["origin_path"]).convert("RGB")

            # Đảm bảo cùng kích thước
            if bg_pil.size != origin_pil.size:
                origin_pil = origin_pil.resize(bg_pil.size, Image.BICUBIC)

            bg_np = np.array(bg_pil)
            origin_np = np.array(origin_pil)

            # 3a. Background Subtraction
            delta_map, delta_color = compute_delta_map(
                bg_np, origin_np, color_space=args.color_space
            )

            # 3b. Binary Mask (Otsu + Morphology)
            binary_mask = refine_to_binary_mask(delta_map, otsu=True)

            # 3c. DINOv3 PCA (nếu bật)
            pca_map = None
            pca_fg_mask = None
            combined_mask = None
            patch_probs = None

            if backbone is not None:
                pca_map, pca_fg_mask, _ = extract_dino_pca_features(
                    backbone, origin_pil, device,
                    img_size=224, patch_size=patch_size
                )

                # Combined mask: Δ ∩ PCA
                combined_mask = combine_masks(
                    binary_mask, pca_fg_mask, origin_np.shape[:2],
                    strategy="pca_guided"
                )

            # 3d. Patch-level masking probabilities (cho BG-Guided DINO)
            patch_probs = compute_patch_level_delta(
                delta_map, patch_size=patch_size, img_size=224, alpha=args.alpha
            )

            # 3e. Quality metrics
            metrics = compute_mask_quality_metrics(
                delta_map, binary_mask, pca_fg_mask
            )
            metrics["camera_id"] = cam_id
            metrics["origin_name"] = origin_name
            all_metrics.append(metrics)

            # 3f. Lưu outputs
            # Delta map (grayscale)
            delta_save = (np.clip(delta_map, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(delta_save).save(os.path.join(delta_dir, f"{stem}_delta.png"))

            # Binary mask
            Image.fromarray(binary_mask).save(os.path.join(mask_dir, f"{stem}_mask.png"))

            # Combined mask (nếu có)
            if combined_mask is not None:
                Image.fromarray(combined_mask).save(
                    os.path.join(mask_dir, f"{stem}_mask_combined.png")
                )

            # Visualization grid
            if len(all_metrics) <= args.max_vis:
                plot_analysis_grid(
                    origin_img=origin_np,
                    bg_img=bg_np,
                    delta_map=delta_map,
                    delta_color=delta_color,
                    binary_mask=binary_mask,
                    pca_map=pca_map,
                    pca_fg_mask=pca_fg_mask,
                    combined_mask=combined_mask,
                    patch_probs=patch_probs,
                    metrics=metrics,
                    camera_id=cam_id,
                    save_path=os.path.join(vis_dir, f"{stem}_analysis.png"),
                )

        except Exception as e:
            print(f"⚠️  Lỗi xử lý {origin_name}: {e}")
            continue

    # 4. Tổng hợp báo cáo
    if all_metrics:
        # JSON report
        report_path = os.path.join(args.output_dir, "bg_subtract_poc_results.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": vars(args),
                "total_pairs_processed": len(all_metrics),
                "aggregate_metrics": {
                    "mean_foreground_ratio": float(np.mean([m["foreground_ratio"] for m in all_metrics])),
                    "std_foreground_ratio": float(np.std([m["foreground_ratio"] for m in all_metrics])),
                    "mean_fg_bg_contrast": float(np.mean([m["fg_bg_contrast"] for m in all_metrics])),
                    "mean_pca_iou": float(np.mean([m.get("pca_iou", 0) for m in all_metrics])) if args.use_pca else None,
                },
                "per_sample_metrics": all_metrics,
            }, f, indent=2, ensure_ascii=False)

        # Summary
        mean_fg = np.mean([m["foreground_ratio"] for m in all_metrics])
        mean_contrast = np.mean([m["fg_bg_contrast"] for m in all_metrics])
        print(f"\n{'=' * 78}")
        print(f" ✅ PoC PIPELINE HOÀN TẤT")
        print(f"{'=' * 78}")
        print(f" Tổng cặp xử lý    : {len(all_metrics)}")
        print(f" FG Ratio trung bình: {mean_fg:.1%}")
        print(f" FG/BG Contrast TB  : {mean_contrast:.4f}")
        if args.use_pca:
            mean_iou = np.mean([m.get("pca_iou", 0) for m in all_metrics])
            print(f" PCA IoU trung bình : {mean_iou:.4f}")
        print(f" Output directory   : {args.output_dir}")
        print(f" Report saved       : {report_path}")
        print(f"{'=' * 78}")


# =====================================================================
# 9. CLI ARGUMENT PARSER
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Background-Guided Vehicle Foreground Extraction PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Chỉ background subtraction (không cần GPU)
  python DINO/bg_subtract_poc.py \\
      --bg_dir backgrounds/ --origin_dir images/ \\
      --output_dir checkpoints/bg_poc --no_pca

  # Đầy đủ với DINOv3 PCA refinement
  python DINO/bg_subtract_poc.py \\
      --bg_dir backgrounds/ --origin_dir images/ \\
      --backbone dinov3_vits16 --device cuda

  # Dùng match strategy route_hourly (IC4SD-Traffic-HCM)
  python DINO/bg_subtract_poc.py \\
      --bg_dir traffic_backgrounds/ --origin_dir output/ \\
      --match_strategy route_hourly --device cuda

  # Dùng match strategy theo subfolder
  python DINO/bg_subtract_poc.py \\
      --bg_dir backgrounds/ --origin_dir images/ \\
      --match_strategy subfolder
        """,
    )

    # Data paths
    parser.add_argument("--bg_dir", type=str, required=True,
                        help="Thư mục chứa ảnh background (nền tĩnh, không phương tiện)")
    parser.add_argument("--origin_dir", type=str, required=True,
                        help="Thư mục chứa ảnh origin (có phương tiện)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/bg_subtract_poc",
                        help="Thư mục lưu kết quả")

    # Matching
    parser.add_argument("--match_strategy", type=str, default="route_hourly",
                        choices=["camera_id", "same_name", "subfolder", "route_hourly"],
                        help="Chiến lược ghép cặp BG-Origin (mặc định: route_hourly cho IC4SD)")
    parser.add_argument("--camera_regex", type=str, default=r"^(\d+)_",
                        help="Regex trích camera ID từ tên file (nhóm capture 1)")

    # Processing
    parser.add_argument("--color_space", type=str, default="lab",
                        choices=["rgb", "lab", "hsv", "gray"],
                        help="Không gian màu cho background subtraction")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Tỷ lệ foreground-bias cho patch masking (0=random, 1=full FG)")

    # DINOv3
    parser.add_argument("--backbone", type=str, default="dinov3_vits16",
                        help="DINOv3/v2 backbone model name")
    parser.add_argument("--weights", type=str, default=None,
                        help="Custom backbone weights path (None = Meta official)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device (cuda/cpu)")
    parser.add_argument("--use_pca", action="store_true", default=True,
                        help="Sử dụng DINOv3 PCA refinement (mặc định: bật)")
    parser.add_argument("--no_pca", action="store_true", default=False,
                        help="Tắt DINOv3 PCA (chỉ dùng background subtraction)")

    # Limits
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Số cặp tối đa xử lý (0 = tất cả)")
    parser.add_argument("--max_vis", type=int, default=20,
                        help="Số figure visualization tối đa sinh ra")

    args = parser.parse_args()

    if args.no_pca:
        args.use_pca = False

    return args


if __name__ == "__main__":
    args = parse_args()
    run_poc_pipeline(args)
