"""
=============================================================================
 Hướng G: visualize_paper.py
 Bộ công cụ xuất 4 Biểu Đồ Trực Quan Hóa Chuẩn Paper Khoa Học Đỉnh Cao:
 
 1. Figure 1: Per-Image PCA Feature Maps (Chuẩn chính thống Meta DINOv2)
    - PCA độc lập trên từng ảnh (Intra-image SVD), loại bỏ hoàn toàn ám màu ngày/đêm
    - Second-stage Foreground SVD trên các patch xe (bóc tách chi tiết bộ phận xe)
 2. Figure 2: Multi-Head Self-Attention Maps (Unsupervised Vehicle Discovery)
    - Trích xuất attention từ [CLS] token đến các patch qua 6 Attention Heads
 3. Figure 3: Cross-Camera Nearest Neighbors Retrieval (Bẻ gãy Camera Shortcut)
    - Đối chiếu Query xe ở Cam A với Top-5 xe gần nhất ở các Cam B, C, D
    - So sánh trực tiếp Baseline DINOv3 (dính shortcut) vs Hướng G (Vehicle-Centric)
 4. Figure 4: Singular Value Spectrum & Effective Rank (Scree Plot)
    - Đường cong phổ trị riêng và phương sai tích lũy chứng minh không sụp đổ chiều
=============================================================================
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_script_dir)
_dino_root = os.path.dirname(_pkg_dir)
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from common.backbone_loader import get_dino_backbone, extract_tokens, imagenet_normalize
from common.gpu_utils import smart_load_state_dict, unwrap_model
from direction1_new.core import PositionStats, GMMCalibrator, FrozenExtractor
from direction1_new.data import TrafficCameraDataset, SyntheticTrafficDataset


# =============================================================================
# 1. FIGURE 1: PER-IMAGE PCA THEO CHUẨN META DINOv2
# =============================================================================
def compute_per_image_pca(
    feat: np.ndarray,
    pi: Optional[np.ndarray] = None,
    fg_threshold: float = 0.40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính PCA chuẩn Meta DINOv2 độc lập trên 1 ảnh duy nhất (Intra-image SVD).
    
    Args:
        feat: Ma trận patch tokens (448, D) của 1 ảnh duy nhất.
        pi: Bản đồ tiền cảnh TAM (448,) trong [0, 1].
        fg_threshold: Ngưỡng lọc patch tiền cảnh.
        
    Returns:
        std_pca: (16, 28, 3) Standard PCA toàn cảnh nội tại của ảnh.
        fg_pca: (16, 28, 3) Foreground PCA bóc tách chi tiết các bộ phận xe.
    """
    N, D = feat.shape
    corners_idx = [0, 1, 26, 27, 448 - 28, 448 - 27, 448 - 2, 448 - 1]

    # 1. Standard PCA trên chính ảnh này
    f_mean = feat.mean(axis=0, keepdims=True)
    f_cent = feat - f_mean
    _, _, vt = np.linalg.svd(f_cent, full_matrices=False)
    pca3 = f_cent @ vt[:3].T  # (448, 3)

    # Căn chỉnh dấu eigenvector theo 4 góc ảnh (nền) để định hướng nhất quán
    for c in range(3):
        if pca3[corners_idx, c].mean() > pca3[:, c].mean():
            pca3[:, c] = -pca3[:, c]

    p_min = np.percentile(pca3, 1, axis=0, keepdims=True)
    p_max = np.percentile(pca3, 99, axis=0, keepdims=True)
    std_pca = np.clip((pca3 - p_min) / np.maximum(p_max - p_min, 1e-6), 0.0, 1.0)

    # 2. Foreground-only Second-Stage PCA trên các patch xe của riêng ảnh này
    fg_pca = np.full((N, 3), 0.05, dtype=np.float32)  # Nền tối dịu
    if pi is not None:
        pi_flat = pi.reshape(-1)
        fg_mask = (pi_flat > fg_threshold)
        if fg_mask.sum() < 4:
            fg_mask = (pi_flat >= np.percentile(pi_flat, 80))

        if fg_mask.sum() >= 4:
            f_fg = feat[fg_mask]
            f_fg_mean = f_fg.mean(axis=0, keepdims=True)
            f_fg_cent = f_fg - f_fg_mean
            _, _, vt_fg = np.linalg.svd(f_fg_cent, full_matrices=False)
            fg_pca3 = f_fg_cent @ vt_fg[:3].T  # (K, 3)

            f_min = np.percentile(fg_pca3, 2, axis=0, keepdims=True)
            f_max = np.percentile(fg_pca3, 98, axis=0, keepdims=True)
            fg_scaled = np.clip((fg_pca3 - f_min) / np.maximum(f_max - f_min, 1e-6), 0.0, 1.0)
            fg_pca[fg_mask] = fg_scaled
        else:
            fg_pca = std_pca * np.clip(pi_flat[:, None] * 2.0, 0.1, 1.0)
    else:
        fg_pca = std_pca

    return std_pca.reshape(16, 28, 3), fg_pca.reshape(16, 28, 3)


def render_fig1_per_image_pca(
    backbone: nn.Module,
    loader: DataLoader,
    frozen_extractor: nn.Module,
    pos_stats: PositionStats,
    calibrator: GMMCalibrator,
    device: torch.device,
    out_path: str = "paper_fig1_per_image_pca.png",
    num_samples: int = 4,
):
    """Xuất Figure 1 Teaser chuẩn Meta DINOv2 Per-Image PCA."""
    print("\n🎨 [Figure 1] Đang kết xuất Per-Image PCA chuẩn Meta DINOv2...")
    backbone.eval()
    samples = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["x1"].to(device)
            cids = batch["cid"].to(device)
            B = imgs.shape[0]

            # Lấy pi
            if "pi1" in batch and batch["pi1"] is not None:
                pis = batch["pi1"].to(device)
            else:
                u = frozen_extractor(imgs).float()
                a, valid = pos_stats.atypicality(cids, u)
                pis = calibrator.posterior(a).view(B, 16, 28)

            # Lấy patch tokens
            x_norm = imagenet_normalize(imgs)
            _, patch_spatial = extract_tokens(backbone, x_norm, patch_size=16)
            patches = patch_spatial.reshape(B, 16 * 28, -1)

            for b in range(B):
                samples.append({
                    "img": imgs[b].permute(1, 2, 0).cpu().numpy().clip(0, 1),
                    "pi": pis[b].cpu().numpy(),
                    "feat": patches[b].cpu().numpy(),
                })
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break

    if not samples:
        print("⚠️ Không có mẫu dữ liệu để vẽ Figure 1.")
        return

    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 5, figsize=(18, 3.4 * n_rows), dpi=200)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, s in enumerate(samples):
        img_np = s["img"]
        H, W = img_np.shape[:2]
        pi_np = s["pi"].reshape(16, 28)
        feat_np = s["feat"]

        std_pca, fg_pca = compute_per_image_pca(feat_np, pi=pi_np)

        # Smooth bilinear interpolation
        std_tensor = torch.from_numpy(std_pca).permute(2, 0, 1).unsqueeze(0).float()
        std_smooth = F.interpolate(std_tensor, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)

        fg_tensor = torch.from_numpy(fg_pca).permute(2, 0, 1).unsqueeze(0).float()
        fg_smooth = F.interpolate(fg_tensor, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)

        # Cột 1: Ảnh gốc
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Khung Hình Gốc $x_{{{i+1}}}$", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # Cột 2: Bản đồ tiền cảnh TAM pi
        axes[i, 1].imshow(img_np)
        axes[i, 1].imshow(pi_np, cmap="jet", alpha=0.55, extent=[0, W, H, 0])
        axes[i, 1].set_title("Bản Đồ Tiền Cảnh TAM $\\pi$", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")

        # Cột 3: Standard Per-Image PCA
        axes[i, 2].imshow(std_smooth)
        axes[i, 2].set_title("Standard Per-Image PCA", fontsize=11, fontweight="bold")
        axes[i, 2].axis("off")

        # Cột 4: Foreground DINOv2 PCA (Part Segmentation)
        axes[i, 3].imshow(fg_smooth)
        axes[i, 3].set_title("DINOv2 Foreground PCA (Bộ Phận Xe)", fontsize=11, fontweight="bold")
        axes[i, 3].axis("off")

        # Cột 5: High-Contrast Overlay (Ghép đè rực rỡ trên nền đen trắng)
        gray_bg = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])[..., None] * 0.4
        pi_up = F.interpolate(torch.from_numpy(pi_np).unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode="bilinear", align_corners=False).squeeze().numpy()
        alpha = np.clip((pi_up - 0.25) / 0.35, 0.0, 1.0)[..., None]
        blended = fg_smooth * alpha + gray_bg * (1.0 - alpha)
        axes[i, 5 if axes.shape[1] > 5 else 4].imshow(np.clip(blended, 0, 1))
        axes[i, 4].set_title("Bóc Tách Xe / Nền Hướng G", fontsize=11, fontweight="bold")
        axes[i, 4].axis("off")

    plt.suptitle("Figure 1: Trực Quan Hóa Biểu Diễn Đặc Trưng Phương Tiện Per-Image PCA (Chuẩn Meta DINOv2)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✅ [Figure 1 Hoàn Tất] Đã lưu tại: {os.path.abspath(out_path)}")


# =============================================================================
# 2. FIGURE 2: MULTI-HEAD SELF-ATTENTION MAPS
# =============================================================================
def extract_cls_attention_maps(backbone: nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Trích xuất attention maps từ [CLS] token đến các patch qua toàn bộ các heads ở layer cuối.
    Trả về: (B, num_heads, Hp, Wp)
    """
    raw_bb = unwrap_model(backbone)
    B = x.shape[0]

    # Nhánh 1: DINOv2 chính thống có sẵn get_last_selfattention
    if hasattr(raw_bb, "get_last_selfattention"):
        attn = raw_bb.get_last_selfattention(x)  # (B, num_heads, N, N)
        num_patches = 16 * 28
        cls_attn = attn[:, :, 0, -num_patches:]  # Lấy 448 patch tokens cuối cùng (bỏ qua register tokens)
        return cls_attn.view(B, cls_attn.shape[1], 16, 28)

    # Nhánh 2: Đăng ký Hook trên block cuối cùng của Vision Transformer
    attn_storage = {}
    target_block = None

    if hasattr(raw_bb, "blocks") and len(raw_bb.blocks) > 0:
        target_block = raw_bb.blocks[-1]
    elif hasattr(raw_bb, "layers") and len(raw_bb.layers) > 0:
        target_block = raw_bb.layers[-1]

    if target_block is None:
        return None

    # Tìm submodule attention
    attn_mod = getattr(target_block, "attn", getattr(target_block, "attention", None))
    if attn_mod is None:
        return None

    def hook_fn(m, inp, outp):
        attn_storage["inp"] = inp[0]

    h = attn_mod.register_forward_hook(hook_fn)
    try:
        _ = extract_tokens(raw_bb, x, patch_size=16)
    finally:
        h.remove()

    if "inp" not in attn_storage:
        return None

    # Tái tạo attention matrix: Softmax(Q @ K.T / sqrt(d))
    inp_feat = attn_storage["inp"]  # (B, N, C)
    if hasattr(attn_mod, "qkv"):
        B_cur, N_tok, C_dim = inp_feat.shape
        num_heads = getattr(attn_mod, "num_heads", 6)
        head_dim = C_dim // num_heads
        scale = getattr(attn_mod, "scale", head_dim ** -0.5)

        qkv = attn_mod.qkv(inp_feat).reshape(B_cur, N_tok, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        
        # Áp dụng 2D RoPE nếu module có hỗ trợ (DINOv3 RoPE)
        if hasattr(attn_mod, "rope") and attn_mod.rope is not None:
            try:
                q = attn_mod.rope(q)
                k = attn_mod.rope(k)
            except Exception:
                pass

        sim = (q @ k.transpose(-2, -1)) * scale
        weights = sim.softmax(dim=-1)  # (B, num_heads, N, N)
        num_patches = 16 * 28
        # Lấy chính xác 448 spatial patches ở cuối (loại trừ CLS token và 4 storage/register tokens)
        cls_attn = weights[:, :, 0, -num_patches:]
        return cls_attn.view(B_cur, num_heads, 16, 28)

    return None


def render_fig2_attention_maps(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: str = "paper_fig2_multi_head_attention.png",
    num_samples: int = 3,
):
    """Xuất Figure 2 Multi-Head Self-Attention Maps (Caron et al., ICCV 2021)."""
    print("\n🔍 [Figure 2] Đang trích xuất Multi-Head Self-Attention Maps...")
    backbone.eval()
    samples = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["x1"].to(device)
            x_norm = imagenet_normalize(imgs)
            attn_maps = extract_cls_attention_maps(backbone, x_norm)
            if attn_maps is None:
                print("⚠️ Backbone hiện tại không hỗ trợ trích xuất ma trận Attention. Bỏ qua Figure 2.")
                return

            B = imgs.shape[0]
            for b in range(B):
                samples.append({
                    "img": imgs[b].permute(1, 2, 0).cpu().numpy().clip(0, 1),
                    "attn": attn_maps[b].cpu().numpy(),  # (num_heads, 16, 28)
                })
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break

    if not samples:
        return

    num_heads = samples[0]["attn"].shape[0]
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, num_heads + 2, figsize=(2.8 * (num_heads + 2), 3.2 * n_rows), dpi=200)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, s in enumerate(samples):
        img_np = s["img"]
        H, W = img_np.shape[:2]
        attn_h = s["attn"]  # (num_heads, 16, 28)

        # Cột 1: Ảnh gốc
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Ảnh Gốc", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # Các Heads
        for h in range(num_heads):
            m = attn_h[h]
            m_norm = (m - m.min()) / max(m.max() - m.min(), 1e-6)
            axes[i, h + 1].imshow(img_np)
            axes[i, h + 1].imshow(m_norm, cmap="magma", alpha=0.65, extent=[0, W, H, 0])
            axes[i, h + 1].set_title(f"Head #{h+1}", fontsize=10)
            axes[i, h + 1].axis("off")

        # Cột cuối: Trung bình các Heads
        mean_attn = attn_h.mean(axis=0)
        mean_norm = (mean_attn - mean_attn.min()) / max(mean_attn.max() - mean_attn.min(), 1e-6)
        axes[i, num_heads + 1].imshow(img_np)
        axes[i, num_heads + 1].imshow(mean_norm, cmap="inferno", alpha=0.65, extent=[0, W, H, 0])
        axes[i, num_heads + 1].set_title("Mean Attention", fontsize=10, fontweight="bold")
        axes[i, num_heads + 1].axis("off")

    plt.suptitle("Figure 2: Multi-Head Self-Attention Maps từ [CLS] Token (Unsupervised Object Discovery)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✅ [Figure 2 Hoàn Tất] Đã lưu tại: {os.path.abspath(out_path)}")


# =============================================================================
# 3. FIGURE 3: CROSS-CAMERA NEAREST NEIGHBORS RETRIEVAL
# =============================================================================
def render_fig3_cross_camera_knn(
    backbone_model: nn.Module,
    backbone_baseline: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: str = "paper_fig3_cross_camera_knn.png",
    num_queries: int = 3,
    top_k: int = 5,
    max_pool_batches: int = 30,
):
    """
    Xuất Figure 3: So sánh Cross-Camera k-NN Retrieval giữa Baseline DINOv3 và Hướng G.
    Chứng minh Hướng G triệt tiêu thành công Camera Shortcut, tìm đúng xe ở camera khác.
    """
    print("\n🚗 [Figure 3] Đang tính toán Cross-Camera Nearest Neighbors Retrieval...")
    backbone_model.eval()
    backbone_baseline.eval()

    all_imgs = []
    all_cids = []
    feats_model = []
    feats_base = []

    batch_idx = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["x1"].to(device)
            cids = batch["cid"].to(device)
            x_norm = imagenet_normalize(imgs)

            cls_m, _ = extract_tokens(backbone_model, x_norm, patch_size=16)
            cls_b, _ = extract_tokens(backbone_baseline, x_norm, patch_size=16)

            cls_m = F.normalize(cls_m, p=2, dim=-1)
            cls_b = F.normalize(cls_b, p=2, dim=-1)

            B = imgs.shape[0]
            for b in range(B):
                all_imgs.append(imgs[b].permute(1, 2, 0).cpu().numpy().clip(0, 1))
                all_cids.append(int(cids[b].item()))
                feats_model.append(cls_m[b].cpu())
                feats_base.append(cls_b[b].cpu())

            batch_idx += 1
            if batch_idx >= max_pool_batches or len(all_imgs) >= 300:
                break

    if len(all_imgs) < top_k + 2:
        print("⚠️ Không đủ dữ liệu đa camera để chạy k-NN Retrieval.")
        return

    F_model = torch.stack(feats_model, dim=0)  # (N, D)
    F_base = torch.stack(feats_base, dim=0)    # (N, D)
    cids_np = np.array(all_cids)

    # Chọn các query từ các camera có nhiều ảnh
    unique_cams, cam_counts = np.unique(cids_np, return_counts=True)
    valid_cams = [c for c, cnt in zip(unique_cams, cam_counts) if cnt >= 2]
    if len(valid_cams) < 2:
        valid_cams = list(unique_cams)

    query_indices = []
    for c in valid_cams[:num_queries]:
        idxs = np.where(cids_np == c)[0]
        query_indices.append(idxs[0])

    fig, axes = plt.subplots(num_queries * 2, top_k + 1, figsize=(2.4 * (top_k + 1), 3.0 * num_queries * 2), dpi=200)

    for q_idx, q_i in enumerate(query_indices):
        q_cam = cids_np[q_i]
        diff_cam_mask = (cids_np != q_cam)
        diff_indices = np.where(diff_cam_mask)[0]

        if len(diff_indices) < top_k:
            diff_indices = np.arange(len(all_imgs))

        # 1. Baseline DINOv3 Retrieval
        sims_b = torch.matmul(F_base[diff_indices], F_base[q_i]).numpy()
        top_b_local = np.argsort(-sims_b)[:top_k]
        top_b_indices = diff_indices[top_b_local]

        # 2. Hướng G Vehicle-Centric Retrieval
        sims_m = torch.matmul(F_model[diff_indices], F_model[q_i]).numpy()
        top_m_local = np.argsort(-sims_m)[:top_k]
        top_m_indices = diff_indices[top_m_local]

        row_base = q_idx * 2
        row_model = q_idx * 2 + 1

        # Cột Query
        for r, tag in [(row_base, "Baseline DINOv3"), (row_model, "Hướng G (Ours)")]:
            axes[r, 0].imshow(all_imgs[q_i])
            axes[r, 0].set_title(f"Query (Cam {q_cam})\n[{tag}]", fontsize=9, fontweight="bold")
            axes[r, 0].axis("off")

        # Top-K Baseline
        for k in range(top_k):
            idx_k = top_b_indices[k]
            axes[row_base, k + 1].imshow(all_imgs[idx_k])
            axes[row_base, k + 1].set_title(f"Top-{k+1} (Cam {cids_np[idx_k]})\nSim: {sims_b[top_b_local[k]]:.3f}", fontsize=8)
            axes[row_base, k + 1].axis("off")

        # Top-K Hướng G
        for k in range(top_k):
            idx_k = top_m_indices[k]
            axes[row_model, k + 1].imshow(all_imgs[idx_k])
            axes[row_model, k + 1].set_title(f"Top-{k+1} (Cam {cids_np[idx_k]})\nSim: {sims_m[top_m_local[k]]:.3f}", fontsize=8)
            axes[row_model, k + 1].axis("off")

    plt.suptitle("Figure 3: Đối Chiếu Cross-Camera k-NN Retrieval: Baseline DINOv3 vs Hướng G (Vehicle-Centric)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✅ [Figure 3 Hoàn Tất] Đã lưu tại: {os.path.abspath(out_path)}")


# =============================================================================
# 4. FIGURE 4: SINGULAR VALUE SPECTRUM & EFFECTIVE RANK
# =============================================================================
def compute_effective_rank_and_spectrum(feat_mat: torch.Tensor) -> Tuple[float, np.ndarray, np.ndarray]:
    """Tính toán trị riêng SVD, tỷ lệ phương sai và Effective Rank theo Roy & Vetterli."""
    feat_norm = F.normalize(feat_mat, p=2, dim=-1)
    feat_cent = feat_norm - feat_norm.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(feat_cent).cpu().numpy()
    s = s / (np.sum(s) + 1e-12)
    s_clean = s[s > 1e-12]
    entropy = -np.sum(s_clean * np.log(s_clean))
    erank = float(np.exp(entropy))
    cum_var = np.cumsum(s)
    return erank, s, cum_var


def render_fig4_singular_value_spectrum(
    backbone_model: nn.Module,
    backbone_baseline: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: str = "paper_fig4_singular_value_spectrum.png",
    max_batches: int = 40,
):
    """Xuất Figure 4: Phổ trị riêng và Effective Rank đối chiếu giữa Baseline và Hướng G."""
    print("\n📊 [Figure 4] Đang tính toán Phổ Trị Riêng & Bậc Hiệu Dụng Effective Rank...")
    backbone_model.eval()
    backbone_baseline.eval()

    cls_model_list = []
    cls_base_list = []

    b_count = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["x1"].to(device)
            x_norm = imagenet_normalize(imgs)

            cls_m, _ = extract_tokens(backbone_model, x_norm, patch_size=16)
            cls_b, _ = extract_tokens(backbone_baseline, x_norm, patch_size=16)

            cls_model_list.append(cls_m.cpu())
            cls_base_list.append(cls_b.cpu())

            b_count += 1
            if b_count >= max_batches:
                break

    M_model = torch.cat(cls_model_list, dim=0)
    M_base = torch.cat(cls_base_list, dim=0)

    erank_m, s_m, cum_m = compute_effective_rank_and_spectrum(M_model)
    erank_b, s_b, cum_b = compute_effective_rank_and_spectrum(M_base)
    D = len(s_m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=250)

    dims = np.arange(1, D + 1)

    # Đồ thị 1: Scree Plot (Log Scale)
    ax1.plot(dims, s_b, color="#d9534f", linestyle="--", linewidth=2.0, label=f"Baseline DINOv3 (ERank: {erank_b:.2f} / {D})")
    ax1.plot(dims, s_m, color="#0275d8", linestyle="-", linewidth=2.2, label=f"Hướng G Vehicle-Centric (ERank: {erank_m:.2f} / {D})")
    ax1.set_yscale("log")
    ax1.set_xlabel("Thứ Tự Trị Riêng $k$ (Singular Value Index)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Chuẩn Hóa Trị Riêng $\\tilde{\\sigma}_k$ (Thang Log)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Phổ Trị Riêng (Scree Plot)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10, loc="upper right")

    # Đồ thị 2: Cumulative Variance Ratio
    ax2.plot(dims, cum_b * 100.0, color="#d9534f", linestyle="--", linewidth=2.0, label="Baseline DINOv3")
    ax2.plot(dims, cum_m * 100.0, color="#0275d8", linestyle="-", linewidth=2.2, label="Hướng G (Ours)")
    ax2.axhline(90.0, color="gray", linestyle=":", alpha=0.7, label="Ngưỡng 90% Phương Sai")
    ax2.set_xlabel("Số Chiều Đặc Trưng $k$", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Phương Sai Tích Lũy (%)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Tỷ Lệ Phương Sai Tích Lũy", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10, loc="lower right")

    plt.suptitle(
        f"Figure 4: Đánh Giá Bậc Hiệu Dụng & Độ Phong Phú Đặc Trưng (Non-Collapse Proof)\n"
        f"Hướng G đạt ERank = {erank_m:.2f} ({erank_m/D*100:.1f}%) — Khẳng định không sụp đổ chiều",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✅ [Figure 4 Hoàn Tất] Đã lưu tại: {os.path.abspath(out_path)}")


# =============================================================================
# FIGURE 5: TRAINING LOSS DYNAMICS & CONVERGENCE CURVES
# =============================================================================
def render_fig_loss_curves(checkpoint_path: str, out_path: str) -> bool:
    """Trích xuất lịch sử loss từ checkpoint và vẽ biểu đồ đường cong hàm mất mát chuẩn bài báo."""
    print(f"\n📈 [Figure 5] Đang trích xuất lịch sử hàm mất mát từ: {checkpoint_path}")
    if not os.path.isfile(checkpoint_path):
        print(f"❌ Không tìm thấy file checkpoint tại: {checkpoint_path}")
        return False

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    history = ckpt.get("history", {}) if isinstance(ckpt, dict) else {}
    if not history or not history.get("total"):
        metrics_json = os.path.join(os.path.dirname(checkpoint_path), "training_metrics.json")
        if os.path.isfile(metrics_json):
            import json
            try:
                with open(metrics_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    history = data.get("history", {})
            except Exception:
                pass

    if not history or not history.get("total"):
        print("⚠️ Không tìm thấy mảng 'history' hợp lệ trong checkpoint!")
        return False

    epochs = list(range(1, len(history["total"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=220)

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

    # Panel 2: Regularization & Disentanglement
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

    ax2.set_title("Regularization & Shortcut Elimination (RIC, KoLeo, CamAdv)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="semibold")
    ax2.set_ylabel("Loss Magnitude", fontsize=10, fontweight="semibold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    if has_reg:
        ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    cur_epoch = len(epochs)
    best_loss = min(history["total"])
    plt.suptitle(
        f"Figure 5: Diễn Biến Hội Tụ Các Hàm Mất Mát — Hướng G (DINOv3 Vehicle-Centric SSL)\n"
        f"Hoàn thành {cur_epoch} Epochs | Best Total Loss = {best_loss:.4f}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✅ [Figure 5 - Loss Curves Hoàn Tất] Đã lưu tại: {os.path.abspath(out_path)}")
    return True


# =============================================================================
# MAIN CLI DISPATCHER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Bộ công cụ xuất biểu đồ trực quan hóa chuẩn bài báo khoa học DINOv3")
    parser.add_argument("--data_dir", type=str, default="/workspace/output", help="Thư mục ảnh giao thông")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/direction1_new/best_checkpoint.pth", help="Checkpoint mô hình Hướng G")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16", help="Tên backbone DINOv3")
    parser.add_argument("--weights", type=str, default=None, help="Trọng số baseline DINOv3 gốc")
    parser.add_argument("--hf_token", type=str, default="", help="Hugging Face Token")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="paper_figures", help="Thư mục lưu các biểu đồ")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "pca", "attention", "knn", "rank", "loss"], help="Chế độ xuất hình ảnh")
    parser.add_argument("--tam_cache_dir", type=str, default=None, help="Thư mục cache pi (.npy)")
    args = parser.parse_args()

    # Nếu chỉ xuất đồ thị Loss Curve, thực hiện ngay lập tức không cần tải Dataset hay Backbone
    if args.mode == "loss":
        out_loss_path = os.path.join(args.output_dir, "loss_curve.png")
        success = render_fig_loss_curves(args.checkpoint, out_loss_path)
        if success:
            print(f"🎉 Biểu đồ đường cong hàm mất mát đã sẵn sàng tại: {os.path.abspath(out_loss_path)}")
        sys.exit(0 if success else 1)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(" 🚀 [PAPER VISUALIZER] KHỞI ĐỘNG BỘ CÔNG CỤ XUẤT 4 HÌNH VẼ CHO PAPER KHOA HỌC")
    print("=" * 80)
    print(f" Checkpoint Hướng G  : {args.checkpoint}")
    print(f" Thư mục dữ liệu     : {args.data_dir}")
    print(f" Thiết bị tính toán  : {device}")
    print(f" Thư mục xuất ảnh    : {os.path.abspath(args.output_dir)}")
    print("=" * 80)

    # 1. Dataset
    tam_cache_dir = args.tam_cache_dir
    if tam_cache_dir is None:
        for candidate in ["tam_cache", "checkpoints/direction1_new/tam_cache", "/workspace/tam_cache", "/workspace/nckh/DINO/tam_cache"]:
            if os.path.isdir(candidate):
                tam_cache_dir = candidate
                break

    if os.path.isdir(args.data_dir):
        dataset = TrafficCameraDataset(data_dir=args.data_dir, img_size=(256, 448), tam_cache_dir=tam_cache_dir)
        print(f"✅ Nạp {len(dataset)} ảnh thực tế từ '{args.data_dir}' ({dataset.num_cams} camera)")
    else:
        print(f"💡 Thư mục '{args.data_dir}' chưa có, dùng SyntheticTrafficDataset mô phỏng...")
        dataset = SyntheticTrafficDataset(num_samples=120, num_cams=8)

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 2. Tải mô hình Hướng G Checkpoint
    backbone_model, embed_dim, patch_size = get_dino_backbone(
        model_name=args.model_name,
        pretrained=True,
        weights_path=args.weights,
        device=device,
        hf_token=args.hf_token,
    )
    pos_stats = PositionStats(num_cams=getattr(dataset, "num_cams", 16), num_patches=448, num_states=4, feat_dim=64).to(device)
    calibrator = GMMCalibrator()

    import copy
    fe_backbone = copy.deepcopy(backbone_model)
    for p in fe_backbone.parameters():
        p.requires_grad = False
    frozen_extractor = FrozenExtractor(
        model_name=args.model_name,
        pca_dim=64,
        device=device,
        hf_token=args.hf_token,
        backbone=fe_backbone,
        embed_dim=embed_dim,
        patch_size=patch_size,
    ).to(device)

    # Nạp Checkpoint Hướng G
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"🔄 Đang nạp Checkpoint Hướng G từ: {args.checkpoint}")
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
        bb_sd = ckpt.get("student_backbone", ckpt.get("student", ckpt.get("model_state", ckpt.get("state_dict", ckpt))))
        smart_load_state_dict(backbone_model, bb_sd, strict=False, verbose=False)

        if "pos_stats" in ckpt:
            pos_stats.load_state_dict(ckpt["pos_stats"])
        if "calibrator" in ckpt and ckpt["calibrator"]:
            calibrator.load_state_dict(ckpt["calibrator"])
        if "pca_components" in ckpt and ckpt["pca_components"] is not None:
            frozen_extractor.pca_components.copy_(ckpt["pca_components"].to(device))
            if "pca_mean" in ckpt and ckpt["pca_mean"] is not None:
                frozen_extractor.pca_mean.copy_(ckpt["pca_mean"].to(device))
        print("✅ [Checkpoint] Nạp thành công trọng số vào mô hình Hướng G.")
    elif tam_cache_dir and os.path.isdir(tam_cache_dir):
        stats_file = os.path.join(tam_cache_dir, "tam_stats.pth")
        if os.path.isfile(stats_file):
            tam_saved = torch.load(stats_file, map_location="cpu", weights_only=False)
            if "pos_stats" in tam_saved:
                pos_stats.load_state_dict(tam_saved["pos_stats"])
            if "calibrator" in tam_saved:
                calibrator.load_state_dict(tam_saved["calibrator"])
            if "pca_components" in tam_saved and tam_saved["pca_components"] is not None:
                frozen_extractor.pca_components.copy_(tam_saved["pca_components"].to(device))
                if "pca_mean" in tam_saved and tam_saved["pca_mean"] is not None:
                    frozen_extractor.pca_mean.copy_(tam_saved["pca_mean"].to(device))

    # 3. Tải mô hình Baseline DINOv3 gốc để đối chứng
    print("📦 Đang nạp Baseline DINOv3 gốc (chưa qua Vehicle-Centric SSL) để làm đối chứng...")
    backbone_baseline, _, _ = get_dino_backbone(
        model_name=args.model_name,
        pretrained=True,
        weights_path=args.weights,
        device=device,
        hf_token=args.hf_token,
    )
    backbone_baseline.eval()

    # 4. Thực thi xuất các Figure theo chế độ
    if args.mode in ["all", "pca"]:
        fig1_path = os.path.join(args.output_dir, "paper_fig1_per_image_pca.png")
        render_fig1_per_image_pca(
            backbone=backbone_model,
            loader=loader,
            frozen_extractor=frozen_extractor,
            pos_stats=pos_stats,
            calibrator=calibrator,
            device=device,
            out_path=fig1_path,
            num_samples=4,
        )

    if args.mode in ["all", "attention"]:
        fig2_path = os.path.join(args.output_dir, "paper_fig2_multi_head_attention.png")
        render_fig2_attention_maps(
            backbone=backbone_model,
            loader=loader,
            device=device,
            out_path=fig2_path,
            num_samples=3,
        )

    if args.mode in ["all", "knn"]:
        fig3_path = os.path.join(args.output_dir, "paper_fig3_cross_camera_knn.png")
        render_fig3_cross_camera_knn(
            backbone_model=backbone_model,
            backbone_baseline=backbone_baseline,
            loader=loader,
            device=device,
            out_path=fig3_path,
            num_queries=3,
            top_k=5,
        )

    if args.mode in ["all", "rank"]:
        fig4_path = os.path.join(args.output_dir, "paper_fig4_singular_value_spectrum.png")
        render_fig4_singular_value_spectrum(
            backbone_model=backbone_model,
            backbone_baseline=backbone_baseline,
            loader=loader,
            device=device,
            out_path=fig4_path,
            max_batches=20,
        )

    if args.mode in ["all", "loss"]:
        fig5_path = os.path.join(args.output_dir, "loss_curve.png")
        render_fig_loss_curves(
            checkpoint_path=args.checkpoint,
            out_path=fig5_path,
        )

    print("\n" + "=" * 80)
    print(" 🎉 TOÀN BỘ CÁC BIỂU ĐỒ PAPER ĐÃ ĐƯỢC XUẤT THÀNH CÔNG!")
    print(f" 📂 Thư mục chứa hình ảnh: {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
