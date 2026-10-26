"""
=============================================================================
 Hướng 1 Mới (Direction 1 New): Trực quan hóa tiến trình huấn luyện 8 cột & Đồ thị Loss
 - Cột 5 (DINO Gốc PCA) & Cột 7 (Student PCA Nội Tại): Full-Image 3-Component PCA
   hiển thị đầy đủ 100% khung hình (loại bỏ mảng cắt đen nền).
 - Cột 6 (Student PCA Đồng Bộ Cam): Batch-Consistent SVD đồng bộ màu sắc xuyên camera.
 - Cột 8 (Student Foreground PCA): Two-Stage Object-Centric PCA định hướng bởi TAM pi
   trên nền tối (0.05) để bóc tách chi tiết phương tiện giao thông.
=============================================================================
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


_CORNERS_IDX = [0, 1, 26, 27, 448 - 28, 448 - 27, 448 - 2, 448 - 1]


def compute_full_image_pca(feat: np.ndarray) -> np.ndarray:
    """
    Tính Full-Image 3-Component PCA trên toàn bộ 448 patch tokens của 1 ảnh (không cắt đen nền).
    Căn chỉnh dấu của 3 vector riêng theo 4 góc ảnh để đảm bảo màu sắc nền/đường ổn định.

    Args:
        feat: (448, D) numpy array của patch tokens.
    Returns:
        pca_rgb: (16, 28, 3) trong dải [0, 1].
    """
    f_mean = feat.mean(axis=0, keepdims=True)
    f_cent = feat - f_mean
    _, _, vt = np.linalg.svd(f_cent, full_matrices=False)
    pca3 = f_cent @ vt[:3].T  # (448, 3)

    for c in range(3):
        if pca3[_CORNERS_IDX, c].mean() > pca3[:, c].mean():
            pca3[:, c] = -pca3[:, c]

    p_lo = np.percentile(pca3, 1, axis=0, keepdims=True)
    p_hi = np.percentile(pca3, 99, axis=0, keepdims=True)
    res_rgb = np.clip((pca3 - p_lo) / np.maximum(p_hi - p_lo, 1e-6), 0.0, 1.0)
    return res_rgb.reshape(16, 28, 3)


def compute_foreground_pca(feat: np.ndarray, fg_mask_in: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Two-Stage Object-Centric PCA chuẩn Meta DINOv2 dành riêng cho Cột 8 (Student Foreground PCA):
    Lọc các patch phương tiện (qua fg_mask_in từ TAM pi hoặc PC1) và chạy SVD thứ hai
    chỉ trên các patch phương tiện để phân biệt chi tiết đầu xe, thùng xe, bánh xe trên nền tối (0.05).

    Args:
        feat: (448, D) numpy array của patch tokens.
        fg_mask_in: (448,) bool mask của vùng phương tiện.
    Returns:
        fg_rgb: (16, 28, 3) trong dải [0, 1].
    """
    N, _ = feat.shape
    f_mean = feat.mean(axis=0, keepdims=True)
    f_cent = feat - f_mean
    _, _, vt = np.linalg.svd(f_cent, full_matrices=False)
    pc1 = f_cent @ vt[0]
    if pc1[_CORNERS_IDX].mean() > pc1.mean():
        pc1 = -pc1
    pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)

    if fg_mask_in is not None:
        fg_mask = fg_mask_in
    else:
        fg_mask = (pc1_norm > 0.50)
        if fg_mask.sum() < 8:
            fg_mask = (pc1_norm >= np.percentile(pc1_norm, 75))

    res_rgb = np.full((N, 3), 0.05, dtype=np.float32)
    if fg_mask.sum() >= 4:
        f_fg = feat[fg_mask]
        f_fg_mean = f_fg.mean(axis=0, keepdims=True)
        f_fg_cent = f_fg - f_fg_mean
        _, _, vt_fg = np.linalg.svd(f_fg_cent, full_matrices=False)
        fg_proj = f_fg_cent @ vt_fg[:3].T
        p_lo = np.percentile(fg_proj, 2, axis=0, keepdims=True)
        p_hi = np.percentile(fg_proj, 98, axis=0, keepdims=True)
        fg_norm = np.clip((fg_proj - p_lo) / np.maximum(p_hi - p_lo, 1e-6), 0.0, 1.0)
        res_rgb[fg_mask] = fg_norm
    else:
        pca3 = f_cent @ vt[:3].T
        p_lo = np.percentile(pca3, 2, axis=0, keepdims=True)
        p_hi = np.percentile(pca3, 98, axis=0, keepdims=True)
        res_rgb = np.clip((pca3 - p_lo) / np.maximum(p_hi - p_lo, 1e-6), 0.0, 1.0)
    return res_rgb.reshape(16, 28, 3)


def save_direction_g_visuals(
    x_orig: torch.Tensor,
    pi: torch.Tensor,
    mask: torch.Tensor,
    x_srs: torch.Tensor,
    patch_features: Optional[torch.Tensor],
    save_path: str,
    epoch: int = 1,
    vanilla_patch_features: Optional[torch.Tensor] = None,
):
    """
    Xuất biểu đồ 8 cột trực quan hóa quá trình học Tự Giám Sát Hướng G đối chiếu DINO gốc:
      Cột 1: Ảnh gốc frame x1
      Cột 2: Bản đồ tiền cảnh bất thường TAM pi (Heatmap)
      Cột 3: Mặt nạ che phân tầng AGM (Mask)
      Cột 4: Ảnh sau khi hoán đổi vùng tĩnh SRS (kèm mức sai khác nền Δ)
      Cột 5: DINOv3 Gốc PCA (Full-Image Pretrained Baseline - đầy đủ 100% khung hình)
      Cột 6: Student PCA Đồng Bộ (Batch-Consistent: màu nền & xe nhất quán qua các camera)
      Cột 7: Student PCA Nội Tại (Full-Image Per-Image: đối chiếu toàn cảnh 1-1 với Cột 5)
      Cột 8: Student Foreground PCA (Object-Centric: bóc tách chi tiết thân xe trên nền tối)
    """
    B = min(4, x_orig.shape[0])
    fig, axes = plt.subplots(B, 8, figsize=(28, 3.2 * B), dpi=150)
    if B == 1:
        axes = np.expand_dims(axes, axis=0)

    # 1. Tiền xử lý Full-Image PCA cho DINO gốc (Cột 5 - không cắt đen nền)
    vanilla_pca_grid_batch = []
    if vanilla_patch_features is not None:
        v_feats_np = vanilla_patch_features[:B].detach().cpu().numpy()
        for b in range(B):
            v_pca_map = compute_full_image_pca(v_feats_np[b])
            vanilla_pca_grid_batch.append(v_pca_map)
        vanilla_pca_grid_batch = np.stack(vanilla_pca_grid_batch, axis=0)
    else:
        vanilla_pca_grid_batch = None

    # 2. Tiền xử lý PCA cho Student (Cột 6, 7, 8)
    batch_pca_grid = []
    per_image_pca_grid = []
    fg_grid_batch = []

    if patch_features is not None:
        feats_np = patch_features[:B].detach().cpu().numpy()  # (B, 448, D)

        # A. Cột 6: Batch-Consistent PCA (Chiếu toàn bộ B ảnh lên 1 hệ trục SVD chung)
        feats_all = feats_np.reshape(B * 448, -1)
        all_mean = feats_all.mean(axis=0, keepdims=True)
        feats_all_cent = feats_all - all_mean
        _, _, vt_batch = np.linalg.svd(feats_all_cent, full_matrices=False)
        pca3_batch_all = feats_all_cent @ vt_batch[:3].T  # (B * 448, 3)

        corners_batch_idx = [b_idx * 448 + c_idx for b_idx in range(B) for c_idx in _CORNERS_IDX]
        for c in range(3):
            if pca3_batch_all[corners_batch_idx, c].mean() > pca3_batch_all[:, c].mean():
                pca3_batch_all[:, c] = -pca3_batch_all[:, c]

        bp_min = np.percentile(pca3_batch_all, 1, axis=0, keepdims=True)
        bp_max = np.percentile(pca3_batch_all, 99, axis=0, keepdims=True)
        pca3_batch_norm = np.clip((pca3_batch_all - bp_min) / np.maximum(bp_max - bp_min, 1e-6), 0.0, 1.0)
        batch_pca_grid = pca3_batch_norm.reshape(B, 16, 28, 3)

        # B. Cột 7 (Full-Image Per-Image PCA) và Cột 8 (Foreground Object-Centric PCA)
        for b in range(B):
            f_b = feats_np[b]  # (448, D)
            pi_b = pi[b].detach().cpu().numpy().reshape(-1)
            fg_mask_b = (pi_b > 0.40)
            if fg_mask_b.sum() < 4:
                fg_mask_b = (pi_b >= np.percentile(pi_b, 80))

            # Cột 7: Full-Image PCA nội tại (100% khung hình, không cắt đen nền)
            per_image_pca_grid.append(compute_full_image_pca(f_b))
            # Cột 8: Student Foreground PCA có định hướng bởi TAM trên nền tối
            fg_grid_batch.append(compute_foreground_pca(f_b, fg_mask_in=fg_mask_b))

        per_image_pca_grid = np.stack(per_image_pca_grid, axis=0)
        fg_grid_batch = np.stack(fg_grid_batch, axis=0)
    else:
        batch_pca_grid = None
        per_image_pca_grid = None
        fg_grid_batch = None

    for b in range(B):
        # 1. Ảnh gốc
        img_np = x_orig[b].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        H_orig, W_orig = img_np.shape[0], img_np.shape[1]

        axes[b, 0].imshow(img_np)
        axes[b, 0].set_title(f"Mẫu #{b+1} — Ảnh Gốc $x_1$", fontsize=10, fontweight="bold")
        axes[b, 0].axis("off")

        # 2. TAM Heatmap (16, 28) nội suy lên kích thước ảnh
        pi_np = pi[b].reshape(16, 28).detach().cpu().numpy()
        axes[b, 1].imshow(img_np)
        axes[b, 1].imshow(pi_np, cmap="jet", alpha=0.6, extent=[0, W_orig, H_orig, 0])
        axes[b, 1].set_title("Bản Đồ Tiền Cảnh TAM $\\pi$", fontsize=10, fontweight="bold")
        axes[b, 1].axis("off")

        # 3. AGM Mask (448 patches -> 16x28)
        mask_np = mask[b].reshape(16, 28).detach().cpu().numpy().astype(float)
        axes[b, 2].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[b, 2].set_title("Mặt Nạ Che Phân Tầng AGM", fontsize=10, fontweight="bold")
        axes[b, 2].axis("off")

        # 4. SRS Swapped Image & Sai khác nền
        srs_np = x_srs[b].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        diff_val = float(np.abs(srs_np - img_np).mean() * 100.0)
        axes[b, 3].imshow(srs_np)
        axes[b, 3].set_title(f"Ảnh Ghép Nền SRS (Δ={diff_val:.1f}%)", fontsize=10, fontweight="bold")
        axes[b, 3].axis("off")

        # 5. DINO Gốc PCA (Full-Image Pretrained Baseline)
        if vanilla_pca_grid_batch is not None:
            v_tensor = torch.from_numpy(vanilla_pca_grid_batch[b]).permute(2, 0, 1).unsqueeze(0).float()
            v_smooth = F.interpolate(v_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
            v_smooth_np = v_smooth.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
            axes[b, 4].imshow(v_smooth_np)
            axes[b, 4].set_title("DINO Gốc PCA (Pretrained)", fontsize=10, fontweight="bold", color="#1f77b4")
        axes[b, 4].axis("off")

        # 6. Student PCA Đồng Bộ (Batch-Consistent)
        if batch_pca_grid is not None:
            bp_tensor = torch.from_numpy(batch_pca_grid[b]).permute(2, 0, 1).unsqueeze(0).float()
            bp_smooth = F.interpolate(bp_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
            bp_smooth_np = bp_smooth.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
            axes[b, 5].imshow(bp_smooth_np)
            axes[b, 5].set_title("Student PCA Đồng Bộ Cam", fontsize=10, fontweight="bold", color="#008080")
        axes[b, 5].axis("off")

        # 7. Student PCA Nội Tại (Full-Image Per-Image)
        if per_image_pca_grid is not None:
            pca_tensor = torch.from_numpy(per_image_pca_grid[b]).permute(2, 0, 1).unsqueeze(0).float()
            pca_smooth = F.interpolate(pca_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
            pca_smooth_np = pca_smooth.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
            axes[b, 6].imshow(pca_smooth_np)
            axes[b, 6].set_title("Student PCA Nội Tại", fontsize=10, fontweight="bold", color="#2ca02c")
        axes[b, 6].axis("off")

        # 8. Student Foreground PCA Feature Map (Object-Centric bóc tách chi tiết phương tiện)
        if fg_grid_batch is not None:
            fg_tensor = torch.from_numpy(fg_grid_batch[b]).permute(2, 0, 1).unsqueeze(0).float()
            fg_smooth = F.interpolate(fg_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
            fg_smooth_np = fg_smooth.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
            axes[b, 7].imshow(fg_smooth_np)
            axes[b, 7].set_title("Student Foreground PCA", fontsize=10, fontweight="bold", color="#d62728")
        axes[b, 7].axis("off")

    plt.suptitle(f"Tiến Trình Huấn Luyện SSL Hướng G — Epoch {epoch} (Đối Chiếu DINO Gốc vs Student)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"🖼️ [Trực Quan Hóa] Đã xuất biểu đồ học biểu diễn (8 cột đối chiếu DINO Gốc vs Student) tại: {save_path}")


def save_loss_curves(history: Dict[str, List[float]], save_path: str):
    """Vẽ và lưu biểu đồ đường cong hàm mất mát chi tiết (2 panels: Core Objectives & Regularization)."""
    if not history or not history.get("total"):
        return
    epochs = list(range(1, len(history["total"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=180)

    # Panel 1: Core SSL Representation Objectives
    ax1.plot(epochs, history["total"], "b-o", linewidth=2.2, markersize=5, label="Total Objective")
    if "dino" in history and history["dino"]:
        ax1.plot(epochs, history["dino"], "g--s", linewidth=1.8, markersize=4, label="DINO [CLS] Distillation")
    if "ibot" in history and history["ibot"]:
        ax1.plot(epochs, history["ibot"], "m-.^", linewidth=1.8, markersize=4, label="iBOT Patch Masking")

    ax1.set_title("Core SSL Objectives (Total, DINO, iBOT)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="semibold")
    ax1.set_ylabel("Loss Magnitude", fontsize=10, fontweight="semibold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Panel 2: Geometric Invariance & Shortcut Elimination
    has_reg = False
    if "koleo" in history and history["koleo"]:
        ax2.plot(epochs, history["koleo"], color="#e67e22", linestyle="-", marker="x", linewidth=1.8, markersize=5, label="KoLeo Entropy (Uniformity)")
        has_reg = True
    if "ric" in history and history["ric"]:
        ax2.plot(epochs, history["ric"], color="#9b59b6", linestyle="--", marker="d", linewidth=1.8, markersize=4, label="RIC Invariance (Geometry & Illum)")
        has_reg = True
    if "cam_adv" in history and history["cam_adv"]:
        ax2.plot(epochs, history["cam_adv"], color="#e74c3c", linestyle="-.", marker="v", linewidth=1.8, markersize=4, label="CamAdv Loss (Shortcut Suppression)")
        has_reg = True
    if "patch_var" in history and history["patch_var"]:
        ax2.plot(epochs, history["patch_var"], color="#27ae60", linestyle=":", marker="o", linewidth=1.8, markersize=4, label="Patch Variance (Anti-Collapse)")
        has_reg = True
    if "anchor" in history and history["anchor"]:
        ax2.plot(epochs, history["anchor"], color="#2980b9", linestyle="--", marker="p", linewidth=1.8, markersize=4, label="DINOv3 Anchor (Anti-Drift)")
        has_reg = True

    ax2.set_title("Regularization & Shortcut Elimination (RIC, KoLeo, CamAdv)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="semibold")
    ax2.set_ylabel("Loss Magnitude", fontsize=10, fontweight="semibold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    if has_reg:
        ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.suptitle("Tiến Trình Hội Tụ Hàm Mất Mát — Hướng G (DINOv3 Vehicle-Centric SSL)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"📊 [Đồ Thị] Đã cập nhật đường cong tổn thất tại: {save_path}")
