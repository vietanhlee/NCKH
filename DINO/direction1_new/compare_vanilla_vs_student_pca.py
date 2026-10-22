"""
=============================================================================
 Bộ Công Cụ Đối Chứng Per-Image PCA: DINOv3 Gốc vs Student (Hướng G)
 Phương pháp: Two-Stage PCA chuẩn Meta DINOv2 (ICCV 2023)

 Quy trình chuẩn khoa học:
   Stage 1: PCA(n=1) trên toàn bộ patch tokens → tách Foreground / Background
   Stage 2: PCA(n=3) CHỈ trên foreground tokens → mã hóa semantic parts thành RGB

 Lưu ý quan trọng:
   - KHÔNG L2-normalize tokens trước PCA (phá hủy phương sai tự nhiên)
   - KHÔNG dùng Batch SVD gộp nhiều camera khác nhau (inter-image variance áp đảo)
   - Foreground mask trích xuất từ PC1 (self-contained), không dùng TAM bên ngoài
=============================================================================
"""

import os
# Ngăn ngừa lỗi OpenMP xung đột thư viện trên Windows (bắt buộc đặt trước torch, numpy, matplotlib)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import time
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")  # Backend không cần display, tránh lỗi trên server headless
import matplotlib.pyplot as plt
from PIL import Image

# Đảm bảo đường dẫn import tương thích mọi môi trường
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dino_root = os.path.dirname(_current_dir)
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from common.backbone_loader import get_dino_backbone, extract_tokens, imagenet_normalize
from common.gpu_utils import smart_load_state_dict


# =============================================================================
# 1. TWO-STAGE PCA CHUẨN META DINOV2 (ICCV 2023)
# =============================================================================
def compute_two_stage_pca(
    feat: np.ndarray,
    grid_h: int = 16,
    grid_w: int = 28,
    fg_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-Stage PCA chuẩn Meta DINOv2 (Oquab et al., ICCV 2023).

    Bước 1: PCA(n_components=1) trên toàn bộ patch tokens.
            PC1 tự nhiên phân tách foreground (xe, người) khỏi background (đường, trời).
            Dùng PC1 để tạo binary foreground mask.

    Bước 2: PCA(n_components=3) CHỈ trên foreground tokens.
            3 hướng phương sai lớn nhất mã hóa semantic parts (đầu xe, thân xe, bánh xe)
            thành 3 kênh RGB.

    Args:
        feat: (N_patches, D) — Raw patch tokens từ ViT, KHÔNG L2-normalize.
              (DINOv3 ViT-S/16: N=448, D=384; tokens có norm tự nhiên ~40-60)
        grid_h, grid_w: Kích thước lưới patch (16 x 28 = 448 cho ảnh 256x448).
        fg_threshold: Ngưỡng trên PC1 đã chuẩn hóa [0, 1] để tách foreground.
                      Mặc định 0.5 (tương đương median split). Giảm → nhiều FG hơn.

    Returns:
        fg_pca_rgb: (grid_h, grid_w, 3) — Foreground PCA RGB (BG = đen). Chuẩn DINOv2 paper.
        full_pca_rgb: (grid_h, grid_w, 3) — Full-image PCA 3-component (tham chiếu).
        fg_mask_2d: (grid_h, grid_w) — Binary foreground mask từ PC1 (bool).
        explained_var: (3,) — Explained variance ratio của top-3 components (tính trên FG).
    """
    N, D = feat.shape
    assert N == grid_h * grid_w, f"Mismatch: feat.shape[0]={N} != grid_h*grid_w={grid_h*grid_w}"

    # --- Chỉ số các patch ở 4 góc ảnh (dùng để căn dấu eigenvector) ---
    # Quy ước: Góc ảnh camera giao thông thường là nền (trời, tường, cây)
    corners = []
    for r in [0, 1, grid_h - 2, grid_h - 1]:
        for c in [0, 1, grid_w - 2, grid_w - 1]:
            idx = r * grid_w + c
            if 0 <= idx < N:
                corners.append(idx)
    corners = list(set(corners))

    # ===========================================================
    # STAGE 1: PCA(n=1) → Tách Foreground / Background
    # ===========================================================
    f_mean = feat.mean(axis=0, keepdims=True)
    f_centered = feat - f_mean
    _, s_vals, Vt = np.linalg.svd(f_centered, full_matrices=False)

    # Chiếu tất cả tokens lên PC1
    pc1 = f_centered @ Vt[0]  # (N,)

    # Căn dấu: Quy ước nền (corners) có pc1 thấp, foreground (xe) có pc1 cao
    if np.mean(pc1[corners]) > np.mean(pc1):
        pc1 = -pc1

    # MinMax scale PC1 về [0, 1]
    pc1_min, pc1_max = pc1.min(), pc1.max()
    if pc1_max - pc1_min < 1e-8:
        # Trường hợp suy biến: ảnh hoàn toàn đồng nhất
        pc1_norm = np.full_like(pc1, 0.5)
    else:
        pc1_norm = (pc1 - pc1_min) / (pc1_max - pc1_min)

    # Tạo foreground mask: patches có PC1 cao = foreground
    fg_mask = pc1_norm > fg_threshold

    # Đảm bảo có đủ patch foreground cho SVD (cần >= 4 tokens cho PCA 3-component)
    if fg_mask.sum() < 4:
        # Fallback: lấy top 30% patches có PC1 cao nhất
        top_k = max(4, int(0.3 * N))
        threshold_val = np.sort(pc1_norm)[-top_k]
        fg_mask = pc1_norm >= threshold_val

    # ===========================================================
    # STAGE 2: PCA(n=3) CHỈ trên Foreground Tokens → RGB
    # ===========================================================
    fg_tokens = feat[fg_mask]   # (N_fg, D) — dùng raw feat, KHÔNG centered toàn ảnh
    fg_mean = fg_tokens.mean(axis=0, keepdims=True)
    fg_centered = fg_tokens - fg_mean
    _, s_fg, Vt_fg = np.linalg.svd(fg_centered, full_matrices=False)

    # Tính Explained Variance Ratio trên foreground
    total_fg_var = np.sum(s_fg ** 2)
    n_components = min(3, len(s_fg))
    explained_var = np.zeros(3)
    explained_var[:n_components] = (s_fg[:n_components] ** 2) / max(total_fg_var, 1e-12)

    # Chiếu foreground tokens lên 3 PC
    fg_proj = fg_centered @ Vt_fg[:3].T  # (N_fg, 3)

    # Căn dấu cho từng PC: Quy ước PC1 có median dương (foreground phía dưới ảnh sáng hơn)
    # Với PC2, PC3: Quy ước tâm ảnh sáng hơn rìa
    for c in range(min(3, fg_proj.shape[1])):
        if np.median(fg_proj[:, c]) < 0:
            fg_proj[:, c] = -fg_proj[:, c]

    # MinMax scale per-channel về [0, 1] (dùng percentile 1-99 để chống outlier)
    for c in range(min(3, fg_proj.shape[1])):
        p_lo = np.percentile(fg_proj[:, c], 1)
        p_hi = np.percentile(fg_proj[:, c], 99)
        if p_hi - p_lo < 1e-8:
            fg_proj[:, c] = 0.5
        else:
            fg_proj[:, c] = np.clip((fg_proj[:, c] - p_lo) / (p_hi - p_lo), 0.0, 1.0)

    # Gán RGB vào lưới: FG = PCA RGB, BG = đen
    fg_pca_flat = np.zeros((N, 3), dtype=np.float32)
    fg_pca_flat[fg_mask] = fg_proj.astype(np.float32)

    # ===========================================================
    # FULL-IMAGE PCA (Tham chiếu) — 3 components trên toàn bộ tokens
    # ===========================================================
    full_proj = f_centered @ Vt[:3].T  # (N, 3)

    # Căn dấu: Corners có giá trị thấp (quy ước)
    for c in range(3):
        if np.mean(full_proj[corners, c]) > np.mean(full_proj[:, c]):
            full_proj[:, c] = -full_proj[:, c]

    # MinMax scale per-channel percentile 1-99
    for c in range(3):
        p_lo = np.percentile(full_proj[:, c], 1)
        p_hi = np.percentile(full_proj[:, c], 99)
        if p_hi - p_lo < 1e-8:
            full_proj[:, c] = 0.5
        else:
            full_proj[:, c] = np.clip((full_proj[:, c] - p_lo) / (p_hi - p_lo), 0.0, 1.0)

    # Reshape về lưới 2D
    fg_pca_rgb = fg_pca_flat.reshape(grid_h, grid_w, 3)
    full_pca_rgb = full_proj.astype(np.float32).reshape(grid_h, grid_w, 3)
    fg_mask_2d = fg_mask.reshape(grid_h, grid_w)

    return fg_pca_rgb, full_pca_rgb, fg_mask_2d, explained_var


# =============================================================================
# 2. DATASET NẠP ẢNH TÙY CHỌN (ĐƠN LẺ HOẶC THƯ MỤC)
# =============================================================================
class SimpleTrafficDataset(Dataset):
    """Dataset đọc ảnh linh hoạt từ đường dẫn file lẻ hoặc danh sách file."""
    def __init__(self, file_paths: List[str], target_size: Tuple[int, int] = (256, 448)):
        self.file_paths = file_paths
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        p = self.file_paths[idx]
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize(
                    (self.target_size[1], self.target_size[0]),
                    Image.Resampling.BILINEAR,
                )
                tensor_img = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
        except Exception:
            # Fallback tạo ảnh placeholder nếu file lỗi
            tensor_img = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)

        return {
            "img": tensor_img,
            "path": p,
            "filename": os.path.basename(p),
        }


def collect_image_files(input_path: str, max_samples: int = 16) -> List[str]:
    """Thu thập danh sách file ảnh hợp lệ, phân bổ đều qua nhiều camera khác nhau (Cross-Camera Diversity)."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        from collections import defaultdict
        import re
        cam_to_files = defaultdict(list)
        all_files = []

        for root, _, filenames in os.walk(input_path):
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() in valid_exts:
                    full_p = os.path.join(root, fn)
                    all_files.append(full_p)
                    # Trích xuất camera prefix (ví dụ: '101' từ '101_1790937453.jpg')
                    match = re.match(r"^([A-Za-z0-9_-]+?)_\d{8,14}", fn)
                    cam_id = match.group(1) if match else os.path.basename(root)
                    cam_to_files[cam_id].append(full_p)

        if len(cam_to_files) > 1:
            selected = []
            cam_list = sorted(list(cam_to_files.keys()))
            idx = 0
            while len(selected) < max_samples and idx < 100:
                for c in cam_list:
                    if len(cam_to_files[c]) > idx:
                        selected.append(cam_to_files[c][idx])
                        if len(selected) >= max_samples:
                            break
                idx += 1
            return selected
        return all_files[:max_samples]
    return []


# =============================================================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG TỪ MÔ HÌNH DINO (KHÔNG L2-NORMALIZE)
# =============================================================================
@torch.no_grad()
def extract_raw_patch_tokens(
    model: nn.Module,
    imgs: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Trích xuất Raw Patch Tokens từ backbone ViT DINO.
    Tokens trả về giữ nguyên magnitude tự nhiên (norm ~ 40-60 cho ViT-S/16).

    QUAN TRỌNG: KHÔNG L2-normalize. Phương pháp PCA chuẩn DINOv2 yêu cầu
    raw tokens vì:
      - L2-norm ép mọi token thành unit vector → xóa sạch magnitude variance
      - PCA trên unit vectors chỉ bắt được angular noise, mất cấu trúc ngữ nghĩa
      - DINOv2 paper KHÔNG normalize trước PCA

    Args:
        model: Backbone ViT (DINOv3/DINOv2), đã ở chế độ eval.
        imgs: Tensor (B, 3, H, W) ở dải [0, 1] (chưa chuẩn hóa ImageNet).
        patch_size: Kích thước patch (16 cho DINOv3, 14 cho DINOv2).

    Returns:
        patch_tokens: (B, N_patches, D) — Raw features, magnitude tự nhiên.
    """
    model.eval()
    x_norm = imagenet_normalize(imgs)
    _, patch_spatial = extract_tokens(model, x_norm, patch_size=patch_size)
    # patch_spatial: (B, Hp, Wp, D)
    B, Hp, Wp, D = patch_spatial.shape
    return patch_spatial.reshape(B, Hp * Wp, D)


# =============================================================================
# 4. BILINEAR UPSAMPLE HỖ TRỢ
# =============================================================================
def upsample_pca_map(pca_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Nội suy bilinear bản đồ PCA từ lưới patch lên kích thước pixel gốc.

    Args:
        pca_map: (grid_h, grid_w, C) với C = 1 hoặc 3.
        target_h, target_w: Kích thước đích (pixel).

    Returns:
        upsampled: (target_h, target_w, C) — Mượt mà, không bị pixelated.
    """
    if pca_map.ndim == 2:
        pca_map = pca_map[..., np.newaxis]
    C = pca_map.shape[2]
    # (grid_h, grid_w, C) → (1, C, grid_h, grid_w)
    t = torch.from_numpy(pca_map).permute(2, 0, 1).unsqueeze(0).float()
    t_up = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    # (1, C, H, W) → (H, W, C)
    result = t_up.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
    if C == 1:
        return result.squeeze(-1)
    return result


# =============================================================================
# 5. HÀM VẼ SO SÁNH VÀ KẾT XUẤT HÌNH ẢNH
# =============================================================================
def render_comparison_figure(
    samples_data: List[dict],
    out_path: str,
    has_student: bool = True,
):
    """
    Kết xuất biểu đồ đối chiếu PCA giữa DINOv3 gốc và Student.

    Layout khi có Student (5 cột):
      Cột 1: Ảnh gốc RGB
      Cột 2: DINOv3 Gốc — Foreground PCA (Two-Stage, BG = đen)
      Cột 3: Student — Foreground PCA (Two-Stage, BG = đen)
      Cột 4: Student — Full PCA (toàn ảnh, tham chiếu)
      Cột 5: Bản đồ sai khác đặc trưng |F_student - F_vanilla| (Heatmap)

    Layout khi chỉ có DINO gốc (4 cột):
      Cột 1: Ảnh gốc RGB
      Cột 2: Foreground PCA (Two-Stage chuẩn DINOv2)
      Cột 3: Full PCA (toàn ảnh)
      Cột 4: FG Mask từ PC1 (minh họa Stage 1)
    """
    n_rows = len(samples_data)
    n_cols = 5 if has_student else 4
    fig_w = 5.0 * n_cols
    fig_h = 4.0 * n_rows + 0.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=180)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, item in enumerate(samples_data):
        img_np = item["img"]  # (H, W, 3) — RGB [0, 1]
        H, W = img_np.shape[:2]

        # ---- Cột 1: Ảnh gốc RGB ----
        axes[i, 0].imshow(img_np)
        fn_short = item["filename"]
        if len(fn_short) > 25:
            fn_short = fn_short[:12] + "..." + fn_short[-10:]
        axes[i, 0].set_title(f"Mẫu #{i+1}\n{fn_short}", fontsize=9, fontweight="bold")
        axes[i, 0].axis("off")

        # ---- Cột 2: DINOv3 Gốc — Foreground PCA (Two-Stage) ----
        v_fg_up = upsample_pca_map(item["vanilla_fg_pca"], H, W)
        axes[i, 1].imshow(v_fg_up)
        var_v = item["vanilla_var"]
        n_fg_v = item["vanilla_fg_ratio"]
        axes[i, 1].set_title(
            f"DINOv3 Gốc — FG PCA\n"
            f"Top3 Var: {sum(var_v)*100:.1f}% | FG: {n_fg_v*100:.0f}%",
            fontsize=8.5, fontweight="bold", color="#1f77b4",
        )
        axes[i, 1].axis("off")

        if has_student:
            # ---- Cột 3: Student — Foreground PCA (Two-Stage) ----
            s_fg_up = upsample_pca_map(item["student_fg_pca"], H, W)
            axes[i, 2].imshow(s_fg_up)
            var_s = item["student_var"]
            n_fg_s = item["student_fg_ratio"]
            axes[i, 2].set_title(
                f"Student — FG PCA\n"
                f"Top3 Var: {sum(var_s)*100:.1f}% | FG: {n_fg_s*100:.0f}%",
                fontsize=8.5, fontweight="bold", color="#2ca02c",
            )
            axes[i, 2].axis("off")

            # ---- Cột 4: Student — Full PCA (toàn ảnh, tham chiếu) ----
            s_full_up = upsample_pca_map(item["student_full_pca"], H, W)
            axes[i, 3].imshow(s_full_up)
            axes[i, 3].set_title(
                "Student — Full PCA\n(Toàn ảnh, tham chiếu)",
                fontsize=8.5, fontweight="bold", color="#008080",
            )
            axes[i, 3].axis("off")

            # ---- Cột 5: Heatmap sai khác đặc trưng ----
            diff_map = item["diff_map"]  # (grid_h, grid_w)
            diff_up = upsample_pca_map(diff_map, H, W)
            vmax_diff = max(float(np.percentile(diff_map, 98)), 0.1)
            im_diff = axes[i, 4].imshow(diff_up, cmap="hot", vmin=0.0, vmax=vmax_diff)
            mean_diff = float(diff_map.mean())
            axes[i, 4].set_title(
                f"Sai Khác |Δ Features|\nTB = {mean_diff:.3f}",
                fontsize=8.5, fontweight="bold", color="#d62728",
            )
            axes[i, 4].axis("off")
            plt.colorbar(im_diff, ax=axes[i, 4], fraction=0.046, pad=0.04, format="%.2f")

        else:
            # ---- Cột 3: Full PCA (toàn ảnh) ----
            v_full_up = upsample_pca_map(item["vanilla_full_pca"], H, W)
            axes[i, 2].imshow(v_full_up)
            axes[i, 2].set_title(
                "Full PCA\n(Toàn ảnh, tham chiếu)",
                fontsize=8.5, fontweight="bold", color="#008080",
            )
            axes[i, 2].axis("off")

            # ---- Cột 4: FG Mask từ PC1 (minh họa Stage 1) ----
            fg_mask = item["vanilla_fg_mask"]  # (grid_h, grid_w) bool
            fg_mask_up = upsample_pca_map(fg_mask.astype(np.float32), H, W)
            axes[i, 3].imshow(fg_mask_up, cmap="gray", vmin=0, vmax=1)
            axes[i, 3].set_title(
                f"FG Mask (PC1 Stage 1)\nFG: {fg_mask.mean()*100:.0f}%",
                fontsize=8.5, fontweight="bold", color="#9467bd",
            )
            axes[i, 3].axis("off")

    main_title = (
        "ĐỐI CHIẾU TWO-STAGE PCA: DINOv3 GỐC vs STUDENT (HƯỚNG G — DIRECTION 1 NEW)"
        if has_student else
        "PHÂN TÍCH TWO-STAGE PCA TRÊN DINOv3 GỐC (VANILLA PRETRAINED)"
    )
    plt.suptitle(main_title, fontsize=12, fontweight="bold", y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Đảm bảo thư mục đích tồn tại
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"🎉 [KẾT QUẢ] Đã xuất biểu đồ PCA Two-Stage tại:\n   -> {os.path.abspath(out_path)}")


# =============================================================================
# 6. TÌM KIẾM CHECKPOINT TỰ ĐỘNG (ƯU TIÊN DIRECTION1_NEW)
# =============================================================================
def resolve_student_checkpoint(user_path: Optional[str] = None) -> Optional[str]:
    """
    Tự động tìm kiếm checkpoint Student theo thứ tự ưu tiên.
    Ưu tiên tuyệt đối direction1_new, loại bỏ hoàn toàn directionG.
    """
    if user_path and os.path.isfile(user_path):
        return user_path

    candidates = [
        # Tương đối khi đứng ở DINO/
        "checkpoints/direction1_new/last_checkpoint.pth",
        "checkpoints/direction1_new/best_checkpoint.pth",
        # Tương đối khi đứng ở root repo
        "DINO/checkpoints/direction1_new/last_checkpoint.pth",
        "DINO/checkpoints/direction1_new/best_checkpoint.pth",
        # Server Linux (Vast.ai)
        "/workspace/nckh/DINO/checkpoints/direction1_new/last_checkpoint.pth",
        "/workspace/nckh/DINO/checkpoints/direction1_new/best_checkpoint.pth",
        "/workspace/nckh/checkpoints/direction1_new/last_checkpoint.pth",
        # Windows local
        "G:/checkpoints/direction1_new/last_checkpoint.pth",
        "G:/nckh/DINO/checkpoints/direction1_new/last_checkpoint.pth",
        # _dino_root based
        os.path.join(_dino_root, "checkpoints/direction1_new/last_checkpoint.pth"),
        os.path.join(_dino_root, "checkpoints/direction1_new/best_checkpoint.pth"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            try:
                if os.path.getsize(c) > 10 * 1024 * 1024:  # > 10MB = file thực
                    return c
            except OSError:
                pass
    return None


# =============================================================================
# 7. HÀM CHÍNH (MAIN CLI ENTRYPOINT)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Đối chứng Two-Stage PCA chuẩn DINOv2: DINOv3 Gốc vs Student (Hướng G).",
    )
    parser.add_argument("--img", "--image_path", type=str, default=None,
                        help="Đường dẫn đến 1 ảnh duy nhất để chạy thử nghiệm.")
    parser.add_argument("--data_dir", "--img_dir", type=str, default=None,
                        help="Thư mục chứa ảnh camera giao thông.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Đường dẫn file checkpoint Student (.pth).")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16",
                        help="Tên backbone DINOv3 (mặc định: dinov3_vits16).")
    parser.add_argument("--weights", type=str, default=None,
                        help="Đường dẫn trọng số baseline gốc nếu có sẵn offline.")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face Token để tải trọng số nếu cần.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Thiết bị tính toán: cuda hoặc cpu.")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Số lượng ảnh mẫu (mặc định: 4).")
    parser.add_argument("--output", type=str, default="pca_comparison_twostage.png",
                        help="Tên file ảnh kết quả.")
    parser.add_argument("--img_h", type=int, default=256, help="Chiều cao ảnh chuẩn hóa.")
    parser.add_argument("--img_w", type=int, default=448, help="Chiều rộng ảnh chuẩn hóa.")
    parser.add_argument("--fg_threshold", type=float, default=0.5,
                        help="Ngưỡng PC1 để tách foreground (mặc định: 0.5).")
    args = parser.parse_args()

    device = torch.device(args.device)
    grid_h = args.img_h // 16
    grid_w = args.img_w // 16

    print("=" * 80)
    print(" 🚗 [PCA BENCHMARK] TWO-STAGE PCA CHUẨN META DINOV2")
    print("=" * 80)
    print(f" • Thiết bị         : {device}")
    print(f" • Backbone         : {args.model_name}")
    print(f" • Kích thước ảnh   : {args.img_w}×{args.img_h} → lưới {grid_w}×{grid_h} = {grid_w*grid_h} patches")
    print(f" • FG threshold     : {args.fg_threshold}")
    print(f" • File kết quả     : {args.output}")

    # ---- 1. Tìm kiếm ảnh đầu vào ----
    candidate_files = []
    if args.img and os.path.isfile(args.img):
        candidate_files = [args.img]
    elif args.data_dir and os.path.isdir(args.data_dir):
        candidate_files = collect_image_files(args.data_dir, max_samples=args.num_samples)
    else:
        for dir_candidate in [
            "sample_test_images/audit_set",
            "/workspace/output",
            os.path.join(_dino_root, "..", "sample_test_images", "audit_set"),
        ]:
            if os.path.isdir(dir_candidate):
                candidate_files = collect_image_files(dir_candidate, max_samples=args.num_samples)
                if candidate_files:
                    print(f"   💡 Tự động tìm thấy {len(candidate_files)} ảnh từ '{dir_candidate}'.")
                    break

    if not candidate_files:
        print("⚠️ Không tìm thấy ảnh thực tế! Tạo ảnh synthetic để kiểm thử...")
        os.makedirs("sample_test_images", exist_ok=True)
        synthetic_path = "sample_test_images/synthetic_traffic_sample.png"
        dummy = np.zeros((256, 448, 3), dtype=np.uint8)
        dummy[100:250, :] = [60, 60, 65]       # Mặt đường xám
        dummy[140:190, 80:160] = [200, 30, 30]  # Xe đỏ
        dummy[160:210, 260:360] = [30, 100, 220]  # Xe xanh
        Image.fromarray(dummy).save(synthetic_path)
        candidate_files = [synthetic_path]

    eval_files = candidate_files[: min(args.num_samples, len(candidate_files))]
    print(f"\n✅ Xử lý {len(eval_files)} ảnh:")
    for idx, f in enumerate(eval_files):
        print(f"   [{idx+1}] {f}")

    # ---- 2. Tải DINOv3 Vanilla Pretrained ----
    print("\n📦 [1/2] Nạp DINOv3 Gốc (Vanilla Pretrained)...")
    backbone_vanilla, embed_dim, patch_size = get_dino_backbone(
        model_name=args.model_name,
        pretrained=True,
        weights_path=args.weights,
        device=device,
        hf_token=args.hf_token,
    )
    backbone_vanilla.eval()
    for p in backbone_vanilla.parameters():
        p.requires_grad = False
    print(f"   ✅ Thành công! embed_dim={embed_dim}, patch_size={patch_size}")

    # ---- 3. Tải Student Checkpoint ----
    student_ckpt_path = resolve_student_checkpoint(args.checkpoint)
    backbone_student = None
    has_student = False

    if student_ckpt_path:
        print(f"\n🔄 [2/2] Nạp Student Checkpoint: {student_ckpt_path}")
        try:
            backbone_student, _, _ = get_dino_backbone(
                model_name=args.model_name, pretrained=False, device=device,
            )
            try:
                ckpt = torch.load(student_ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(student_ckpt_path, map_location="cpu")

            bb_sd = ckpt.get("student_backbone",
                             ckpt.get("student",
                                      ckpt.get("model_state",
                                               ckpt.get("state_dict", ckpt))))
            smart_load_state_dict(backbone_student, bb_sd, strict=False, verbose=False)
            backbone_student.eval()
            for p in backbone_student.parameters():
                p.requires_grad = False
            has_student = True
            print("   ✅ Thành công!")
        except Exception as e:
            print(f"   ⚠️ Không thể nạp checkpoint: {e}. Chạy chế độ chỉ DINO gốc.")
            backbone_student = None
            has_student = False
    else:
        print("\n💡 Không tìm thấy Student Checkpoint. Chạy chế độ phân tích DINO gốc.")

    # ---- 4. Forward & PCA ----
    dataset = SimpleTrafficDataset(eval_files, target_size=(args.img_h, args.img_w))
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False)

    print("\n⚡ Đang trích xuất Raw Patch Tokens và tính Two-Stage PCA...")
    samples_data = []
    t0 = time.time()

    for batch in loader:
        imgs = batch["img"].to(device)  # (B, 3, H, W) — [0, 1]
        filenames = batch["filename"]
        B = imgs.shape[0]

        # Raw patch tokens từ Vanilla (KHÔNG L2-normalize)
        tokens_vanilla = extract_raw_patch_tokens(backbone_vanilla, imgs, patch_size=patch_size)
        tokens_vanilla_np = tokens_vanilla.float().cpu().numpy()

        # Raw patch tokens từ Student (KHÔNG L2-normalize)
        tokens_student_np = None
        if has_student and backbone_student is not None:
            tokens_student = extract_raw_patch_tokens(backbone_student, imgs, patch_size=patch_size)
            tokens_student_np = tokens_student.float().cpu().numpy()

        for b in range(B):
            img_b = imgs[b].permute(1, 2, 0).cpu().numpy().clip(0, 1)
            f_v = tokens_vanilla_np[b]  # (448, D) — Raw, KHÔNG L2-norm

            # Two-Stage PCA cho Vanilla
            v_fg_pca, v_full_pca, v_fg_mask, v_var = compute_two_stage_pca(
                f_v, grid_h=grid_h, grid_w=grid_w, fg_threshold=args.fg_threshold,
            )
            v_fg_ratio = float(v_fg_mask.mean())

            item = {
                "img": img_b,
                "filename": filenames[b],
                "vanilla_fg_pca": v_fg_pca,
                "vanilla_full_pca": v_full_pca,
                "vanilla_fg_mask": v_fg_mask,
                "vanilla_var": v_var,
                "vanilla_fg_ratio": v_fg_ratio,
            }

            if has_student and tokens_student_np is not None:
                f_s = tokens_student_np[b]  # (448, D)

                # Two-Stage PCA cho Student
                s_fg_pca, s_full_pca, s_fg_mask, s_var = compute_two_stage_pca(
                    f_s, grid_h=grid_h, grid_w=grid_w, fg_threshold=args.fg_threshold,
                )
                s_fg_ratio = float(s_fg_mask.mean())

                item["student_fg_pca"] = s_fg_pca
                item["student_full_pca"] = s_full_pca
                item["student_fg_mask"] = s_fg_mask
                item["student_var"] = s_var
                item["student_fg_ratio"] = s_fg_ratio

                # Sai khác đặc trưng L2 (per-patch) — KHÔNG normalize trước khi tính diff
                diff = np.linalg.norm(f_s - f_v, axis=-1)  # (N_patches,)
                diff_2d = diff.reshape(grid_h, grid_w)
                item["diff_map"] = diff_2d

                # Cosine similarity trung bình (dùng normalize cho cosine, chỉ để in metric)
                f_v_n = f_v / (np.linalg.norm(f_v, axis=-1, keepdims=True) + 1e-8)
                f_s_n = f_s / (np.linalg.norm(f_s, axis=-1, keepdims=True) + 1e-8)
                cos_sim = float(np.sum(f_v_n * f_s_n, axis=-1).mean())

                print(
                    f"   • [{filenames[b]}] "
                    f"Cos Sim: {cos_sim:.4f} | "
                    f"L2 Diff TB: {diff.mean():.2f} | "
                    f"Vanilla FG: {v_fg_ratio*100:.0f}% | "
                    f"Student FG: {s_fg_ratio*100:.0f}%"
                )
            else:
                print(
                    f"   • [{filenames[b]}] "
                    f"Var: PC1={v_var[0]*100:.1f}% PC2={v_var[1]*100:.1f}% PC3={v_var[2]*100:.1f}% | "
                    f"FG: {v_fg_ratio*100:.0f}%"
                )

            samples_data.append(item)

    elapsed = time.time() - t0
    fps = len(samples_data) / max(elapsed, 1e-4)
    print(f"\n⏱️ Hoàn tất trong {elapsed:.2f}s ({fps:.1f} FPS).")

    # ---- 5. Xuất hình ảnh ----
    render_comparison_figure(samples_data, args.output, has_student=has_student)
    print("=" * 80)


if __name__ == "__main__":
    main()
