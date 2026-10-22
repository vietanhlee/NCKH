"""
=============================================================================
 Hướng G: Training Pipeline (directionG_camera_ssl/train.py)
 Pipeline huấn luyện tự giám sát DINOv3 kết hợp TAM, AGM và SRS
 Hỗ trợ DDP / Single-GPU / CPU, EMA Teacher và lưu checkpoint toàn diện
=============================================================================
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.backbone_loader import (
    get_dino_backbone,
    extract_tokens,
    extract_tokens_with_grad,
    imagenet_normalize,
)
from common.gpu_utils import (
    get_available_devices,
    setup_multi_gpu,
    unwrap_model,
    clean_state_dict,
    smart_load_state_dict,
)
from common.logger import setup_training_logger
from direction1_new.features import FrozenExtractor
from direction1_new.tam import PositionStats, GMMCalibrator, refine_traffic_foreground_map
from direction1_new.masking import agm_sample
from direction1_new.srs import static_region_swap
from direction1_new.losses import (
    DINOLoss,
    iBOTPatchLoss,
    KoLeoLoss,
    RICLoss,
    pool_vehicle_embedding,
    CameraAdversarialHead,
    compute_dann_alpha,
)
from direction1_new.sampler import CameraGroupedSampler


class DINOHead(nn.Module):
    """Đầu MLP chiếu đa lớp + lớp Prototype chuẩn hóa cho DINO / iBOT."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 16384,
        use_bn: bool = False,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.last_layer.weight, std=0.02)
        self.norm_last_layer = norm_last_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if len(shape) == 3:  # (B, N, D)
            B, N, D = shape
            x = x.reshape(B * N, D)
            feat = self.mlp(x)
            feat = F.normalize(feat, dim=-1, p=2)
            if self.norm_last_layer:
                w = F.normalize(self.last_layer.weight, dim=-1, p=2)
                out = F.linear(feat, w)
            else:
                out = self.last_layer(feat)
            return out.reshape(B, N, -1)
        else:  # (B, D)
            feat = self.mlp(x)
            feat = F.normalize(feat, dim=-1, p=2)
            if self.norm_last_layer:
                w = F.normalize(self.last_layer.weight, dim=-1, p=2)
                return F.linear(feat, w)
            else:
                return self.last_layer(feat)


class DirectionGStudentModel(nn.Module):
    """
    Wrapper gom toàn bộ luồng tính toán của Student (Backbone + DINO Head + iBOT Head)
    để phân bổ song song hoàn hảo trên toàn bộ các GPU qua nn.DataParallel.

    Lưu ý nghiệp vụ:
      - Đầu vào ở dải [0, 1]; chuẩn hóa ImageNet được thực hiện bên trong wrapper.
      - `masks` (B, N) bool từ AGM: patch bị che được thay bằng mask token của DINOv3,
        Student phải dự đoán phân phối prototype của Teacher tại các vị trí đó (iBOT MIM).
      - BẮT BUỘC dùng `extract_tokens_with_grad` để gradient chảy về backbone.
    """
    def __init__(self, backbone: nn.Module, dino_head: nn.Module, ibot_head: nn.Module, patch_size: int = 16):
        super().__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head
        self.patch_size = patch_size

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = imagenet_normalize(x)
        cls_token, patch_spatial = extract_tokens_with_grad(
            self.backbone, x, patch_size=self.patch_size, masks=masks
        )
        B, Hp, Wp, D = patch_spatial.shape
        patches = patch_spatial.reshape(B, Hp * Wp, D)
        dino_logits = self.dino_head(cls_token)
        ibot_logits = self.ibot_head(patches)
        return cls_token, patches, dino_logits, ibot_logits


class DirectionGTeacherModel(nn.Module):
    """
    Wrapper gom toàn bộ luồng suy luận của Teacher EMA để tận dụng tối đa Multi-GPU.
    Teacher luôn nhìn ảnh gốc KHÔNG che, KHÔNG gradient.
    """
    def __init__(self, backbone: nn.Module, dino_head: nn.Module, ibot_head: nn.Module, patch_size: int = 16):
        super().__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head
        self.patch_size = patch_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = imagenet_normalize(x)
        cls_token, patch_spatial = extract_tokens(self.backbone, x, patch_size=self.patch_size)
        B, Hp, Wp, D = patch_spatial.shape
        patches = patch_spatial.reshape(B, Hp * Wp, D)
        dino_logits = self.dino_head(cls_token)
        ibot_logits = self.ibot_head(patches)
        return cls_token, patches, dino_logits, ibot_logits


class TrafficCameraDataset(Dataset):
    """
    Dataset nạp ảnh giao thông thực tế từ thư mục (ví dụ: 'output').
    Tự động nhóm các frame theo camera để phục vụ cơ chế ghép vùng tĩnh SRS (Static Region Swap)
    và lấy mẫu hard-negative cùng camera qua CameraGroupedSampler.
    """
    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (256, 448),
        cross_cam_srs: bool = False,
        min_frame_offset: int = 12,
        tam_cache_dir: Optional[str] = None,
        refine_tam: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.H, self.W = img_size
        self.cross_cam_srs = cross_cam_srs
        self.min_frame_offset = min_frame_offset
        self.tam_cache_dir = tam_cache_dir
        self.refine_tam = refine_tam
        if tam_cache_dir and os.path.isdir(tam_cache_dir):
            print(f"⚡ [TAM Cache] Kích hoạt chế độ đọc trực tiếp bản đồ tiền cảnh pi từ: '{tam_cache_dir}'")

        # 1. Tìm tất cả ảnh hợp lệ
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        all_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    all_files.append(os.path.join(root, f))
        all_files.sort()

        if not all_files:
            raise ValueError(f"Không tìm thấy ảnh hợp lệ (.jpg, .png) trong thư mục '{data_dir}'!")

        # 2. Phân nhóm theo camera (dựa trên tên thư mục con hoặc tiền tố tên file)
        self.cam_to_files = defaultdict(list)
        for path in all_files:
            rel = os.path.relpath(path, data_dir)
            parts = rel.split(os.sep)
            if len(parts) > 1:
                cam_key = parts[0]
            else:
                fname = os.path.splitext(parts[0])[0]
                tokens = fname.split("_")
                cam_key = tokens[0] if len(tokens) > 1 else "cam_0"
            self.cam_to_files[cam_key].append(path)

        self.cam_keys = sorted(list(self.cam_to_files.keys()))
        self.cam_to_id = {k: i for i, k in enumerate(self.cam_keys)}
        self.num_cams = len(self.cam_keys)

        # Danh sách phẳng các mẫu và ánh xạ camera_to_indices cho CameraGroupedSampler
        self.samples = []
        self.camera_to_indices = defaultdict(list)
        for cam_key, files in self.cam_to_files.items():
            cid = self.cam_to_id[cam_key]
            for local_pos, f in enumerate(files):
                global_idx = len(self.samples)
                self.samples.append((f, cid, cam_key, local_pos))
                self.camera_to_indices[cid].append(global_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_img(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (self.W, self.H):
                img = img.resize((self.W, self.H), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        path1, cid, cam_key, local_pos = self.samples[idx]
        x1 = self._load_img(path1)

        # Chọn frame x2 phục vụ SRS
        if self.cross_cam_srs and self.num_cams > 1:
            # Cross-Camera SRS: chọn ngẫu nhiên từ camera khác
            other_keys = [k for k in self.cam_keys if k != cam_key]
            chosen_key2 = random.choice(other_keys)
            cid2 = self.cam_to_id[chosen_key2]
            path2 = random.choice(self.cam_to_files[chosen_key2])
        else:
            cid2 = cid
            cam_files = self.cam_to_files[cam_key]
            n_files = len(cam_files)
            if n_files > 1:
                distant_cands = [
                    cam_files[j] for j in range(n_files)
                    if abs(j - local_pos) >= self.min_frame_offset
                ]
                if distant_cands:
                    path2 = random.choice(distant_cands)
                else:
                    other_cands = [p for p in cam_files if p != path1]
                    path2 = random.choice(other_cands) if other_cands else path1
            else:
                path2 = path1

        x2 = self._load_img(path2)
        res = {"x1": x1, "x2": x2, "cid": cid, "cid2": cid2}

        # Nếu có cache pi, nạp trực tiếp giúp loại bỏ forward FrozenExtractor lúc train
        if self.tam_cache_dir:
            rel1 = os.path.relpath(path1, self.data_dir)
            f_pi1 = os.path.join(self.tam_cache_dir, rel1 + ".npy")
            rel2 = os.path.relpath(path2, self.data_dir)
            f_pi2 = os.path.join(self.tam_cache_dir, rel2 + ".npy")
            if os.path.isfile(f_pi1) and os.path.isfile(f_pi2):
                try:
                    p1 = torch.from_numpy(np.load(f_pi1)).float()
                    p2 = torch.from_numpy(np.load(f_pi2)).float()
                    if self.refine_tam:
                        res["pi1"] = refine_traffic_foreground_map(p1)
                        res["pi2"] = refine_traffic_foreground_map(p2)
                    else:
                        res["pi1"] = p1
                        res["pi2"] = p2
                except Exception:
                    pass

        return res


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
      Cột 5: DINOv3 Gốc PCA (Pretrained Baseline - Đối chứng nguyên bản)
      Cột 6: Student PCA Đồng Bộ (Batch-Consistent: màu nền & xe nhất quán qua các camera)
      Cột 7: Student PCA Nội Tại (Per-Image: độ tương phản chi tiết riêng từng ảnh)
      Cột 8: Student Foreground PCA (Object-Centric: bóc tách chi tiết thân xe)
    """
    B = min(4, x_orig.shape[0])
    fig, axes = plt.subplots(B, 8, figsize=(28, 3.2 * B), dpi=150)
    if B == 1:
        axes = np.expand_dims(axes, axis=0)

    corners_idx = [0, 1, 26, 27, 448 - 28, 448 - 27, 448 - 2, 448 - 1]

    # 1. Tiền xử lý Per-Image PCA cho DINO gốc (Vanilla Pretrained)
    vanilla_pca_grid_batch = []
    if vanilla_patch_features is not None:
        v_feats_np = vanilla_patch_features[:B].detach().cpu().numpy()
        for b in range(B):
            f_vb = v_feats_np[b]
            f_vmean = f_vb.mean(axis=0, keepdims=True)
            f_vcent = f_vb - f_vmean
            _, _, vt_v = np.linalg.svd(f_vcent, full_matrices=False)
            pca3_v = f_vcent @ vt_v[:3].T
            for c in range(3):
                if pca3_v[corners_idx, c].mean() > pca3_v[:, c].mean():
                    pca3_v[:, c] = -pca3_v[:, c]
            p_vmin = np.percentile(pca3_v, 1, axis=0, keepdims=True)
            p_vmax = np.percentile(pca3_v, 99, axis=0, keepdims=True)
            pca3_vnorm = np.clip((pca3_v - p_vmin) / np.maximum(p_vmax - p_vmin, 1e-6), 0.0, 1.0)
            vanilla_pca_grid_batch.append(pca3_vnorm.reshape(16, 28, 3))
        vanilla_pca_grid_batch = np.stack(vanilla_pca_grid_batch, axis=0)
    else:
        vanilla_pca_grid_batch = None

    # 2. Tiền xử lý PCA cho Student: Cả Batch-Consistent (đồng bộ camera) và Per-Image (nội tại)
    batch_pca_grid = []
    per_image_pca_grid = []
    fg_grid_batch = []

    if patch_features is not None:
        feats_np = patch_features[:B].detach().cpu().numpy()  # (B, 448, D)

        # A. Batch-Consistent PCA: Chiếu toàn bộ B ảnh lên 1 hệ trục SVD chung -> Đồng bộ màu sắc xuyên camera
        feats_all = feats_np.reshape(B * 448, -1)
        all_mean = feats_all.mean(axis=0, keepdims=True)
        feats_all_cent = feats_all - all_mean
        _, _, vt_batch = np.linalg.svd(feats_all_cent, full_matrices=False)
        pca3_batch_all = feats_all_cent @ vt_batch[:3].T  # (B * 448, 3)

        # Căn chỉnh dấu eigenvector theo các góc của toàn bộ batch để nền ổn định
        corners_batch_idx = [b_idx * 448 + c_idx for b_idx in range(B) for c_idx in corners_idx]
        for c in range(3):
            if pca3_batch_all[corners_batch_idx, c].mean() > pca3_batch_all[:, c].mean():
                pca3_batch_all[:, c] = -pca3_batch_all[:, c]

        bp_min = np.percentile(pca3_batch_all, 1, axis=0, keepdims=True)
        bp_max = np.percentile(pca3_batch_all, 99, axis=0, keepdims=True)
        pca3_batch_norm = np.clip((pca3_batch_all - bp_min) / np.maximum(bp_max - bp_min, 1e-6), 0.0, 1.0)
        batch_pca_grid = pca3_batch_norm.reshape(B, 16, 28, 3)

        # B. Per-Image PCA và Foreground PCA độc lập cho từng ảnh
        for b in range(B):
            f_b = feats_np[b]  # (448, D)
            f_mean = f_b.mean(axis=0, keepdims=True)
            f_cent = f_b - f_mean
            _, _, vt = np.linalg.svd(f_cent, full_matrices=False)
            pca3 = f_cent @ vt[:3].T  # (448, 3)

            for c in range(3):
                if pca3[corners_idx, c].mean() > pca3[:, c].mean():
                    pca3[:, c] = -pca3[:, c]

            p_min = np.percentile(pca3, 1, axis=0, keepdims=True)
            p_max = np.percentile(pca3, 99, axis=0, keepdims=True)
            pca3_norm = np.clip((pca3 - p_min) / np.maximum(p_max - p_min, 1e-6), 0.0, 1.0)
            per_image_pca_grid.append(pca3_norm.reshape(16, 28, 3))

            # Foreground-focused DINOv2 PCA (SVD bậc 2 trên các patch xe)
            pi_b = pi[b].detach().cpu().numpy().reshape(-1)
            fg_mask_b = (pi_b > 0.40)
            if fg_mask_b.sum() < 4:
                fg_mask_b = (pi_b >= np.percentile(pi_b, 80))

            fg_map = np.full((448, 3), 0.06, dtype=np.float32)  # Nền tối dịu (dark backdrop)
            if fg_mask_b.sum() >= 4:
                f_fg = f_b[fg_mask_b]
                f_fg_mean = f_fg.mean(axis=0, keepdims=True)
                f_fg_cent = f_fg - f_fg_mean
                _, _, vt_fg = np.linalg.svd(f_fg_cent, full_matrices=False)
                fg_pca3 = f_fg_cent @ vt_fg[:3].T  # (N_fg, 3)

                f_min = np.percentile(fg_pca3, 2, axis=0, keepdims=True)
                f_max = np.percentile(fg_pca3, 98, axis=0, keepdims=True)
                fg_norm = np.clip((fg_pca3 - f_min) / np.maximum(f_max - f_min, 1e-6), 0.0, 1.0)
                fg_map[fg_mask_b] = fg_norm
            else:
                fg_map = pca3_norm * np.clip(pi_b[:, None] * 2.0, 0.1, 1.0)

            fg_grid_batch.append(fg_map.reshape(16, 28, 3))

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
        pi_np = pi[b].detach().cpu().numpy()
        axes[b, 1].imshow(img_np)
        axes[b, 1].imshow(pi_np, cmap="jet", alpha=0.6, extent=[0, W_orig, H_orig, 0])
        axes[b, 1].set_title("Bản Đồ Tiền Cảnh TAM $\\pi$", fontsize=10, fontweight="bold")
        axes[b, 1].axis("off")

        # 3. AGM Mask (448 patches -> 16x28)
        mask_np = mask[b].view(16, 28).detach().cpu().numpy().astype(float)
        axes[b, 2].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[b, 2].set_title("Mặt Nạ Che Phân Tầng AGM", fontsize=10, fontweight="bold")
        axes[b, 2].axis("off")

        # 4. SRS Swapped Image & Sai khác nền
        srs_np = x_srs[b].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        diff_val = float(np.abs(srs_np - img_np).mean() * 100.0)
        axes[b, 3].imshow(srs_np)
        axes[b, 3].set_title(f"Ảnh Ghép Nền SRS (Δ={diff_val:.1f}%)", fontsize=10, fontweight="bold")
        axes[b, 3].axis("off")

        # 5. DINO Gốc PCA (Pretrained Baseline - Đối chứng nguyên bản)
        if vanilla_pca_grid_batch is not None:
            v_tensor = torch.from_numpy(vanilla_pca_grid_batch[b]).permute(2, 0, 1).unsqueeze(0).float()
            v_smooth = F.interpolate(v_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
            v_smooth_np = v_smooth.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
            axes[b, 4].imshow(v_smooth_np)
            axes[b, 4].set_title("DINO Gốc PCA (Pretrained)", fontsize=10, fontweight="bold", color="#1f77b4")
        else:
            axes[b, 4].axis("off")
        axes[b, 4].axis("off")

        # 6. Student PCA Đồng Bộ (Batch-Consistent: Màu sắc mặt đường & xe nhất quán giữa các camera)
        if batch_pca_grid is not None:
            bp_tensor = torch.from_numpy(batch_pca_grid[b]).permute(2, 0, 1).unsqueeze(0).float()
            bp_smooth = F.interpolate(bp_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
            bp_smooth_np = bp_smooth.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
            axes[b, 5].imshow(bp_smooth_np)
            axes[b, 5].set_title("Student PCA Đồng Bộ Cam", fontsize=10, fontweight="bold", color="#008080")
        else:
            axes[b, 5].axis("off")
        axes[b, 5].axis("off")

        # 7. Student PCA Nội Tại (Per-Image: Độ tương phản chi tiết riêng trong từng ảnh)
        if per_image_pca_grid is not None:
            pca_tensor = torch.from_numpy(per_image_pca_grid[b]).permute(2, 0, 1).unsqueeze(0).float()
            pca_smooth = F.interpolate(pca_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
            pca_smooth_np = pca_smooth.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
            axes[b, 6].imshow(pca_smooth_np)
            axes[b, 6].set_title("Student PCA Nội Tại", fontsize=10, fontweight="bold", color="#2ca02c")
        else:
            axes[b, 6].axis("off")
        axes[b, 6].axis("off")

        # 8. Student Foreground PCA Feature Map (Object-Centric bóc tách chi tiết phương tiện)
        if fg_grid_batch is not None:
            fg_tensor = torch.from_numpy(fg_grid_batch[b]).permute(2, 0, 1).unsqueeze(0).float()
            fg_smooth = F.interpolate(fg_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
            fg_smooth_np = fg_smooth.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
            axes[b, 7].imshow(fg_smooth_np)
            axes[b, 7].set_title("Student Foreground PCA", fontsize=10, fontweight="bold", color="#d62728")
        else:
            axes[b, 7].axis("off")
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


class SyntheticTrafficDataset(Dataset):
    """Dataset giả lập phục vụ chạy demo hoặc khi chưa liên kết dataset thực tế."""
    def __init__(self, num_samples: int = 200, num_cams: int = 4):
        self.num_samples = num_samples
        self.num_cams = num_cams
        self.camera_to_indices = defaultdict(list)
        for idx in range(num_samples):
            cid = idx % num_cams
            self.camera_to_indices[cid].append(idx)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        cid = idx % self.num_cams
        # Frame 1: 3 x 256 x 448
        x1 = torch.rand(3, 256, 448) * 0.6 + 0.2
        # Frame 2 (cùng camera, khác ngày) để phục vụ SRS
        x2 = x1 * 0.9 + 0.1 * torch.rand(3, 256, 448)
        return {"x1": x1, "x2": x2, "cid": cid, "cid2": cid}


def resolve_resume_checkpoint(resume_target: str) -> str:
    """
    Chuẩn hóa và tìm kiếm file checkpoint an toàn trên cả Local lẫn Kaggle/Colab.
    Hỗ trợ đường dẫn file trực tiếp, URL Kaggle copy-paste nhầm, hoặc đường dẫn thư mục.
    """
    if not resume_target:
        raise ValueError("Đường dẫn resume không được để trống!")

    # 1. Tự động sửa đường dẫn Kaggle nếu người dùng copy nhầm URL 'models/<user>/<slug>/'
    if not os.path.exists(resume_target):
        norm_path = resume_target.replace("\\", "/")
        if "/kaggle/input/" in norm_path:
            subparts = norm_path.split("/kaggle/input/")[1].split("/")
            if len(subparts) >= 3 and subparts[0] == "models":
                alt_path = os.path.join("/kaggle/input", *subparts[2:])
                if os.path.exists(alt_path):
                    print(f"💡 [Smart Path] Tự động chuẩn hóa đường dẫn Kaggle:")
                    print(f"   Từ: '{resume_target}'")
                    print(f"   Sang: '{alt_path}'")
                    resume_target = alt_path

    # 2. Nếu là thư mục, tự động quét tìm file checkpoint bên trong
    if os.path.isdir(resume_target):
        found_cand = None
        for candidate in ["last_checkpoint.pth", "dinov3_direction1_new_latest.pth", "dinov3_directionG_latest.pth", "best_checkpoint.pth"]:
            cand_path = os.path.join(resume_target, candidate)
            if os.path.isfile(cand_path):
                found_cand = cand_path
                break
        if found_cand is None:
            import glob
            pths = glob.glob(os.path.join(resume_target, "**", "*.pth"), recursive=True)
            if pths:
                found_cand = pths[0]
        if found_cand:
            print(f"📂 [Smart Path] Đã tìm thấy checkpoint trong thư mục: {found_cand}")
            resume_target = found_cand
        else:
            raise FileNotFoundError(f"❌ [Resume Error] Thư mục '{resume_target}' không chứa bất kỳ file checkpoint (.pth) nào!")

    # 3. Báo lỗi rõ ràng nếu không tìm thấy file
    if not os.path.isfile(resume_target):
        import glob
        pths_found = glob.glob("/kaggle/input/**/*.pth", recursive=True)[:5] if os.path.isdir("/kaggle/input") else []
        err_msg = (
            f"\n❌ [LỖI RESUME] Không tìm thấy file checkpoint tại: '{resume_target}'\n"
            f"   Nguyên nhân: Đường dẫn file không tồn tại trên hệ thống.\n"
        )
        if pths_found:
            err_msg += f"   💡 Gợi ý các file .pth hiện có trong /kaggle/input:\n"
            for p in pths_found:
                err_msg += f"      - {p}\n"
        err_msg += f"   👉 Vui lòng kiểm tra lại đường dẫn file checkpoint.\n"
        raise FileNotFoundError(err_msg)

    return resume_target


def resolve_or_precompute_tam_cache(
    data_dir: Optional[str],
    requested_cache_dir: Optional[str] = None,
    output_dir: str = "checkpoints/direction1_new",
    model_name: str = "dinov3_vits16",
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    hf_token: str = "",
    tam_warmup_frames: Optional[int] = None,
    gmm_buffer_size: int = 300000,
    gmm_ema_alpha: float = 0.15,
) -> Optional[str]:
    """
    Tự động tìm kiếm hoặc kích hoạt tiền xử lý bộ nhớ đệm TAM pi (tam_cache):
    1. Quét các vị trí tiềm năng (requested_cache_dir, tam_cache, data_dir/../tam_cache, output_dir/tam_cache).
    2. Nếu tìm thấy thư mục chứa file .npy hợp lệ: tự động nạp trực tiếp.
    3. Nếu chưa có: tự động chạy precompute_tam_cache để trích xuất trước toàn bộ pi rồi nạp vào.
    """
    if not data_dir or not os.path.isdir(data_dir):
        return None

    data_parent = os.path.dirname(os.path.abspath(data_dir))
    candidate_dirs = []
    if requested_cache_dir:
        candidate_dirs.append(requested_cache_dir)
    candidate_dirs.extend([
        os.path.join(data_parent, "tam_cache"),
        os.path.join(output_dir, "tam_cache"),
        os.path.abspath("tam_cache"),
        "/workspace/tam_cache",
    ])

    # 1. Tìm xem có thư mục nào đã chứa cache hoàn chỉnh không
    for c_dir in candidate_dirs:
        if os.path.isdir(c_dir):
            meta_file = os.path.join(c_dir, "cache_meta.json")
            has_cache = False
            if os.path.isfile(meta_file):
                has_cache = True
            else:
                for root, _, files in os.walk(c_dir):
                    if any(f.endswith(".npy") for f in files):
                        has_cache = True
                        break
            if has_cache:
                print(f"⚡ [TAM Cache Auto-Detect] Đã tìm thấy bộ nhớ đệm TAM tại: '{c_dir}'. Tự động nạp trực tiếp!")
                return c_dir

    # 2. Nếu chưa có bất kỳ cache nào, tự động kích hoạt tiền xử lý precompute
    chosen_cache_dir = requested_cache_dir or os.path.join(data_parent, "tam_cache")
    print("\n" + "=" * 80)
    print(" 🔍 [TAM Cache Auto-Detect] Chưa tìm thấy bộ nhớ đệm TAM pi.")
    print(f" 🚀 Tự động kích hoạt tiền xử lý (precompute_tam_cache) vào: '{chosen_cache_dir}'")
    print("=" * 80)

    try:
        from direction1_new.precompute_tam import precompute_tam_cache
        precompute_tam_cache(
            data_dir=data_dir,
            cache_dir=chosen_cache_dir,
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            device_str=device,
            hf_token=hf_token,
            tam_warmup_frames=tam_warmup_frames,
            gmm_buffer_size=gmm_buffer_size,
            gmm_ema_alpha=gmm_ema_alpha,
        )
        return chosen_cache_dir
    except Exception as e:
        print(f"⚠️ [TAM Cache Precompute] Không thể tự động tính trước cache ({e}). Chuyển sang trích xuất online...")
        return None


def get_cli_specified_args(argv_list: List[str]) -> set:
    """
    Xác định tập hợp các tham số được người dùng chỉ định tường minh trên dòng lệnh CLI.
    Giúp phân biệt giữa tham số người dùng muốn ghi đè và tham số mặc định cần kế thừa từ checkpoint.
    """
    specified = set()
    cli_map = {
        "--data_dir": "data_dir", "--origin_dir": "data_dir",
        "--model_name": "model_name", "--backbone": "model_name",
        "--batch_size": "batch_size",
        "--epochs": "epochs",
        "--lr": "lr",
        "--backbone_lr_scale": "backbone_lr_scale",
        "--weight_decay": "weight_decay",
        "--phi": "phi",
        "--q_max": "q_max",
        "--mask_ratio": "mask_ratio",
        "--mask_mode": "mask_mode",
        "--p_srs": "p_srs",
        "--srs_continuous": "srs_continuous", "--no_srs_continuous": "srs_continuous",
        "--cross_cam_srs": "cross_cam_srs",
        "--random_pi": "random_pi",
        "--ibot_beta": "ibot_beta",
        "--use_ric": "use_ric", "--no_ric": "use_ric",
        "--lambda_ric": "lambda_ric",
        "--use_cam_adv": "use_cam_adv", "--no_cam_adv": "use_cam_adv",
        "--lambda_cam_adv": "lambda_cam_adv",
        "--use_cam_sampler": "use_cam_sampler", "--no_cam_sampler": "use_cam_sampler",
        "--cams_per_batch": "cams_per_batch",
        "--frames_per_cam": "frames_per_cam",
        "--out_dim": "out_dim",
        "--output_dir": "output_dir", "--save_dir": "output_dir",
        "--weights": "weights",
        "--device": "device",
        "--num_workers": "num_workers",
        "--seed": "seed",
        "--use_amp": "use_amp", "--no_amp": "use_amp",
        "--amp_dtype": "amp_dtype",
        "--resume": "resume",
        "--hf_token": "hf_token",
        "--tam_warmup_frames": "tam_warmup_frames",
        "--tam_cache_dir": "tam_cache_dir",
        "--tam_kmeans_iters": "tam_kmeans_iters",
        "--gmm_buffer_size": "gmm_buffer_size",
        "--gmm_refit_steps": "gmm_refit_steps",
        "--gmm_iters": "gmm_iters",
        "--gmm_ema_alpha": "gmm_ema_alpha",
        "--gmm_subsample_ratio": "gmm_subsample_ratio",
        "--skip_tam_warmup": "skip_tam_warmup",
        "--session_epochs": "session_epochs", "--max_session_epochs": "session_epochs",
        "--warmup_epochs": "warmup_epochs",
    }
    for arg in argv_list:
        clean_arg = arg.split("=")[0].strip()
        if clean_arg in cli_map:
            specified.add(cli_map[clean_arg])
    return specified


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Tự Giám Sát DINOv3 Hướng G (TAM + AGM + SRS + RIC + CamAdv)")
    parser.add_argument("--data_dir", "--origin_dir", dest="data_dir", type=str, default=None, help="Thư mục ảnh giao thông thực tế (ví dụ: output)")
    parser.add_argument("--model_name", "--backbone", dest="model_name", type=str, default="dinov3_vits16", help="Tên backbone DINOv3")
    parser.add_argument("--batch_size", type=int, default=4, help="Kích thước batch (hoặc số camera mỗi batch nếu dùng CameraSampler)")
    parser.add_argument("--epochs", type=int, default=12, help="Tổng số epochs huấn luyện theo kế hoạch toàn bộ (mặc định: 12)")
    parser.add_argument("--session_epochs", "--max_session_epochs", dest="session_epochs", type=int, default=0,
                        help="Số epochs tối đa chạy trong phiên làm việc hiện tại (0 = chạy hết toàn bộ --epochs, ví dụ: 4)")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Số epochs khởi động ấm Learning Rate tuyến tính (mặc định: 1)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Tốc độ học cơ sở cho Projection Heads")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.2, help="Tỷ lệ giảm LR cho pretrained backbone (tránh làm lệch feature)")
    parser.add_argument("--weight_decay", type=float, default=0.04, help="Hệ số weight decay")
    parser.add_argument("--phi", type=float, default=0.5, help="Tỷ lệ che ngân sách xe AGM")
    parser.add_argument("--q_max", type=float, default=0.6, help="Tỷ lệ tối đa patch xe bị che AGM")
    parser.add_argument("--mask_ratio", type=float, default=0.35, help="Tỷ lệ tổng số patch bị che trong AGM (mặc định 0.35)")
    parser.add_argument("--mask_mode", type=str, default="block", choices=["patch", "block", "random"],
                        help="Chế độ che AGM: 'block' (liền khối xe), 'patch' (phân tầng rời rạc), 'random' (ngẫu nhiên)")
    parser.add_argument("--p_srs", type=float, default=0.5, help="Xác suất áp dụng SRS hoán nền")
    parser.add_argument("--srs_continuous", action="store_true", default=True, help="Hoán đổi toàn bộ nền tĩnh tự nhiên (thay vì chắp vá từng patch)")
    parser.add_argument("--no_srs_continuous", dest="srs_continuous", action="store_false")
    parser.add_argument("--cross_cam_srs", action="store_true", default=False, help="Ablation: Hoán nền từ camera khác để triệt góc máy")
    parser.add_argument("--refine_tam", action="store_true", default=True, help="Tự động áp dụng bộ lọc làm sắc nét tiền cảnh và khử nhiễu mặt đường khi đọc cache TAM.")
    parser.add_argument("--no_refine_tam", dest="refine_tam", action="store_false", help="Tắt bộ lọc làm sắc nét tiền cảnh khi đọc cache.")
    parser.add_argument("--random_pi", action="store_true", default=False, help="Ablation: Sử dụng bản đồ pi ngẫu nhiên thay cho TAM")
    parser.add_argument("--ibot_beta", type=float, default=0.5, help="Trọng số tiền cảnh beta trong iBOT Patch Loss (1 + beta * pi)")
    parser.add_argument("--use_ric", action="store_true", default=True, help="Bật Regional Instance Contrastive (RIC) cùng camera")
    parser.add_argument("--no_ric", dest="use_ric", action="store_false")
    parser.add_argument("--lambda_ric", type=float, default=0.15, help="Hệ số trọng số RIC Loss")
    parser.add_argument("--use_cam_adv", action="store_true", default=True, help="Bật Camera Adversarial Discriminator (GRL) chống shortcut")
    parser.add_argument("--no_cam_adv", dest="use_cam_adv", action="store_false")
    parser.add_argument("--lambda_cam_adv", type=float, default=0.05, help="Hệ số trọng số Camera Adversarial Loss")
    parser.add_argument("--use_cam_sampler", action="store_true", default=True, help="Bật CameraGroupedSampler gom nhóm theo camera")
    parser.add_argument("--no_cam_sampler", dest="use_cam_sampler", action="store_false")
    parser.add_argument("--cams_per_batch", type=int, default=4, help="Số camera trong 1 batch của CameraGroupedSampler")
    parser.add_argument("--frames_per_cam", type=int, default=2, help="Số frame mỗi camera trong 1 batch của CameraGroupedSampler")
    parser.add_argument("--batches_per_epoch", type=int, default=0,
                        help="Số batches mỗi epoch khi bật CameraGroupedSampler (0 = 1 lượt camera ~152 batches; -1 = duyệt trọn vẹn số mẫu dataset; >0 = số batches cụ thể, v.d. 1000, 2000)")
    parser.add_argument("--out_dim", type=int, default=4096, help="Số prototype DINO/iBOT head")
    parser.add_argument("--output_dir", "--save_dir", dest="output_dir", type=str, default="checkpoints/direction1_new", help="Thư mục lưu weights")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn custom checkpoint ban đầu (.pth / .safetensors)")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán ('cuda', 'cuda:0', hoặc 'cpu')")
    parser.add_argument("--num_workers", type=int, default=0, help="Số luồng nạp dữ liệu DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Sử dụng Mixed Precision (AMP) để tối ưu VRAM và tăng tốc độ")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Tắt Mixed Precision")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16"],
        help="Kiểu dữ liệu AMP: 'auto' (tự động dùng bfloat16 nếu GPU hỗ trợ Tensor Cores BF16, ngược lại dùng fp16), 'bf16', hoặc 'fp16'",
    )
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint (.pth) để tiếp tục huấn luyện")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace Token")
    parser.add_argument("--tam_warmup_frames", type=int, default=200, help="Số frame mỗi camera dùng để khởi động TAM (fit PCA & K-means, mặc định: 200 frame)")
    parser.add_argument("--tam_cache_dir", type=str, default=None, help="Thư mục chứa cache pi đã tính trước bởi precompute_tam.py (tăng tốc độ nạp 2.5x)")
    parser.add_argument("--tam_kmeans_iters", type=int, default=20, help="Số vòng lặp K-means khi khởi tạo cụm trạng thái nền")
    parser.add_argument("--gmm_buffer_size", type=int, default=300000, help="Dung lượng bộ đệm mẫu atypicality để fit GMM Calibrator (mặc định: 300000 mẫu)")
    parser.add_argument("--gmm_refit_steps", type=int, default=100, help="Chu kỳ số bước huấn luyện cập nhật lại GMM Calibrator (mặc định: 100)")
    parser.add_argument("--gmm_iters", type=int, default=30, help="Số vòng lặp EM khi fit GMM Calibrator")
    parser.add_argument("--gmm_ema_alpha", type=float, default=0.15, help="Hệ số EMA làm mượt tham số GMM khi refit (mặc định: 0.15)")
    parser.add_argument("--gmm_subsample_ratio", type=float, default=0.15, help="Tỷ lệ subsample patch ngẫu nhiên khi train để giảm tương quan không gian (mặc định: 0.15)")
    parser.add_argument("--skip_tam_warmup", action="store_true", default=False, help="Bỏ qua bước khởi động TAM offline")
    parser.add_argument("--log_file", type=str, default=None, help="Tên hoặc đường dẫn file log ghi lại toàn bộ output (mặc định: train.log trong --output_dir)")
    parsed, unknown = parser.parse_known_args()
    if unknown:
        print(f"⚠️ [CLI Warning] Bỏ qua các đối số chưa khai báo: {unknown}")
    if parsed.hf_token:
        os.environ["HF_TOKEN"] = parsed.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = parsed.hf_token
    return parsed


def warmup_tam_offline(
    frozen_extractor: Union[nn.Module, FrozenExtractor],
    pos_stats: PositionStats,
    calibrator: GMMCalibrator,
    dataset: Dataset,
    device: torch.device,
    num_frames_per_cam: int = 200,
    kmeans_iters: int = 20,
    gmm_iters: int = 30,
    batch_size: int = 8,
):
    """
    Khởi động TAM (Temporal Atypicality Map) trước khi bước vào huấn luyện chính thức:
    1. Trích xuất tokens đa camera để fit ma trận chiếu PCA tối ưu (embed_dim -> 64) cho FrozenExtractor.
    2. Chạy offline K-means cosine (K=4) cho từng camera để thiết lập tâm cụm mu, trọng số w và độ phân tán s thực tế.
    3. Thu thập các giá trị atypicality ban đầu theo hạn ngạch cân bằng (Equal Quota) qua 100% camera,
       sau đó hiệu chuẩn phân phối GMM (mu_bg, mu_fg, std_bg, std_fg).
    """
    print("\n" + "=" * 80)
    print(" 🛠️  [TAM WARMUP] BẮT ĐẦU KHỞI ĐỘNG THỐNG KÊ KHÔNG-THỜI GIAN TAM OFFLINE")
    print("=" * 80)
    start_t = time.time()
    raw_fe = unwrap_model(frozen_extractor)

    # Chuẩn bị danh sách ảnh theo từng camera với phân tầng thời gian trải rộng
    cam_images = defaultdict(list)
    if hasattr(dataset, "cam_to_files") and hasattr(dataset, "cam_to_id"):
        for cam_key, files in dataset.cam_to_files.items():
            cid = dataset.cam_to_id[cam_key]
            n_f = len(files)
            if n_f <= num_frames_per_cam:
                subset = list(files)
            else:
                # Lấy mẫu phân tầng đều trải từ sáng đến tối / qua các ngày
                indices = np.linspace(0, n_f - 1, num_frames_per_cam, dtype=int)
                subset = [files[i] for i in indices]
            cam_images[cid] = subset
    else:
        # Trường hợp dataset tổng quát hoặc Synthetic
        for idx in range(min(len(dataset), num_frames_per_cam * 16)):
            item = dataset[idx]
            cid = int(item["cid"])
            if len(cam_images[cid]) < num_frames_per_cam:
                cam_images[cid].append(item["x1"])

    num_active_cams = len(cam_images)
    print(f" 📹 Số camera phát hiện cho Warmup: {num_active_cams} camera (tối đa {num_frames_per_cam} frame/cam)")
    pos_stats.ensure_cam_capacity(max(cam_images.keys(), default=0))

    # GIAI ĐOẠN 1: FIT PCA CHO FROZENEXTRACTOR
    # Thu thập patch tokens đa dạng từ tất cả camera để fit ma trận chiếu 64 chiều
    if not getattr(raw_fe.is_pca_fitted, "item", lambda: False)():
        print(" 🔍 [Giai Đoạn 1/3] Trích xuất patch tokens để Fit PCA ma trận chiếu 64 chiều...")
        all_sample_tokens = []
        raw_fe.backbone.eval()

        for cid, img_list in cam_images.items():
            # Chọn tối đa 10 ảnh mỗi camera để fit PCA
            sample_imgs = img_list[:10]
            tensors = []
            for item in sample_imgs:
                if isinstance(item, str):
                    tensors.append(dataset._load_img(item))
                else:
                    tensors.append(item)
            if not tensors:
                continue

            batch_x = torch.stack(tensors, dim=0).to(device)
            target_H = 16 * raw_fe.patch_size
            target_W = 28 * raw_fe.patch_size
            if batch_x.shape[-2] != target_H or batch_x.shape[-1] != target_W:
                batch_x = F.interpolate(batch_x, size=(target_H, target_W), mode="bilinear", align_corners=False)

            with torch.no_grad():
                x_norm = imagenet_normalize(batch_x)
                _, patch_spatial = extract_tokens(raw_fe.backbone, x_norm, patch_size=raw_fe.patch_size)
                # patch_spatial: (B, 16, 28, D)
                B_cur, Hp, Wp, D_cur = patch_spatial.shape
                tokens_flat = patch_spatial.reshape(B_cur * Hp * Wp, D_cur)
                # Lấy mẫu ngẫu nhiên tối đa 1000 tokens mỗi camera
                if tokens_flat.shape[0] > 1000:
                    perm = torch.randperm(tokens_flat.shape[0], device=device)[:1000]
                    tokens_flat = tokens_flat[perm]
                all_sample_tokens.append(tokens_flat.cpu())

        if all_sample_tokens:
            collected_tokens = torch.cat(all_sample_tokens, dim=0)
            if collected_tokens.shape[0] > 30000:
                perm_all = torch.randperm(collected_tokens.shape[0])[:30000]
                collected_tokens = collected_tokens[perm_all]
            print(f"    -> Đã thu thập {collected_tokens.shape[0]} tokens đặc trưng. Đang tính phân rã PCA...")
            raw_fe.fit_pca(collected_tokens.to(device))
        else:
            print("    ⚠️ Không thu thập được tokens, sử dụng ma trận trực giao ban đầu.")
    else:
        print(" ⏩ [Giai Đoạn 1/3] Ma trận chiếu PCA đã được fit từ trước -> Bỏ qua.")

    # GIAI ĐOẠN 2: KHỞI TẠO K-MEANS CHO TỪNG CAMERA VỚI EQUAL QUOTA BUFFER SAMPLING
    print(f" 🎯 [Giai Đoạn 2/3] Chạy K-means cosine (K={pos_stats.num_states}, iters={kmeans_iters}) cho từng camera...")
    # Tính hạn ngạch mẫu công bằng (Equal Quota) cho từng camera để đảm bảo 100% camera đều có mặt trong buffer
    quota_per_cam = max(100, int(np.ceil(calibrator.max_buffer_size / max(1, num_active_cams))))
    print(f"    -> Cơ chế Equal Quota Sampling: Tối đa {quota_per_cam} mẫu/cam (đảm bảo cân bằng qua toàn bộ {num_active_cams} camera)")

    cams_initialized = 0
    cams_skipped = 0
    for cid, img_list in cam_images.items():
        if len(img_list) == 0:
            continue
        if cid < pos_stats.num_cams and bool(pos_stats.is_initialized[cid].item()):
            cams_skipped += 1
            continue

        # Nạp toàn bộ frame warmup của camera này
        cam_tensors = []
        for item in img_list:
            if isinstance(item, str):
                cam_tensors.append(dataset._load_img(item))
            else:
                cam_tensors.append(item)

        # Trích xuất đặc trưng PCA 64 chiều theo từng batch nhỏ
        all_u = []
        with torch.no_grad():
            for i in range(0, len(cam_tensors), batch_size):
                sub_batch = torch.stack(cam_tensors[i:i+batch_size], dim=0).to(device)
                u_sub = raw_fe(sub_batch)  # (B_sub, 448, 64)
                all_u.append(u_sub)
        U_cam = torch.cat(all_u, dim=0)  # (N_cam, 448, 64)

        # Chạy K-means cosine cho camera cid
        pos_stats.init_kmeans(cid, U_cam, n_iters=kmeans_iters)
        cams_initialized += 1

        # Đưa các điểm khác thường atypicality vào calibrator buffer theo quota công bằng
        with torch.no_grad():
            cids_tensor = torch.full((U_cam.shape[0],), cid, dtype=torch.long, device=device)
            a_cam, valid_cam = pos_stats.atypicality(cids_tensor, U_cam)
            calibrator.push(a_cam, valid_cam, max_samples=quota_per_cam)

    print(f"    -> Đã khởi tạo K-means mới cho {cams_initialized} camera (giữ nguyên {cams_skipped} camera đã có sẵn).")

    # GIAI ĐOẠN 3: HIỆU CHUẨN SỚM PHÂN PHỐI GMM
    current_samples = getattr(calibrator, "size", len(calibrator.buffer))
    print(f" 📊 [Giai Đoạn 3/3] Hiệu chuẩn GMM 2 thành phần (EM iters={gmm_iters}) từ buffer ({current_samples} mẫu đại diện)...")
    calibrator.refit(max_iters=gmm_iters)
    if calibrator.is_fitted:
        print(f"    ✅ GMM Hiệu chuẩn thành công: mu_bg={calibrator.mu_bg:.3f}, mu_fg={calibrator.mu_fg:.3f}, "
              f"std_bg={calibrator.std_bg:.3f}, std_fg={calibrator.std_fg:.3f}, pi_weight={calibrator.pi_weight:.3f}")
    else:
        print("    ℹ️ Cần thêm mẫu atypicality trong quá trình train để hoàn tất GMM.")

    elapsed = time.time() - start_t
    print(f" ✅ [TAM WARMUP HOÀN TẤT] Thời gian thực thi: {elapsed:.2f}s")
    print("=" * 80 + "\n")


def build_smooth_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 1,
    eta_min: float = 1e-6,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Xây dựng Lịch trình Learning Rate mượt mà kết hợp Linear Warmup và Cosine Annealing dài hạn.
    Đảm bảo tính liên tục tuyệt đối C^0 Continuity qua mọi phiên làm việc (Resume Sessions):
    - Tại bất kỳ epoch t trong [0, total_epochs - 1], LR là hàm đơn trị phụ thuộc vào total_epochs kế hoạch.
    - Khi resume session tiếp theo (start_epoch), LR xuất phát chính xác từ giá trị tiếp nối,
      triệt tiêu hoàn toàn hiện tượng sốc gradient (LR Shock / Kick).
    """
    base_lr = float(optimizer.param_groups[0]["initial_lr"] if "initial_lr" in optimizer.param_groups[0] else optimizer.param_groups[0]["lr"])
    eta_min_ratio = float(eta_min) / max(1e-12, base_lr)

    def lr_lambda(current_epoch: int) -> float:
        # current_epoch là số nguyên 0-indexed
        if warmup_epochs > 0 and current_epoch < warmup_epochs:
            # Linear Warmup êm ái từ mức 0.5 lên 1.0 của base_lr
            alpha = float(current_epoch + 1) / float(warmup_epochs + 1)
            return eta_min_ratio + (1.0 - eta_min_ratio) * alpha
        else:
            # Cosine Annealing suy giảm mượt mà về eta_min tại total_epochs - 1
            cosine_steps = max(1, total_epochs - max(0, warmup_epochs))
            progress = float(current_epoch - max(0, warmup_epochs)) / float(cosine_steps)
            progress = min(max(progress, 0.0), 1.0)
            return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + np.cos(np.pi * progress))

    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


def train_direction_g():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Tự động kích hoạt cơ chế lưu toàn bộ log phiên huấn luyện vào file
    setup_training_logger(
        save_dir=args.output_dir,
        log_file=args.log_file,
        session_title=f"HUẤN LUYỆN TỰ GIÁM SÁT DINOv3 HƯỚNG G ({args.model_name})",
    )

    # 0. Tiền xử lý Resume: Nạp checkpoint sớm để đồng bộ siêu tham số (Smart Hyperparameters Restoration)
    preloaded_ckpt = None
    if args.resume:
        resolved_resume_path = resolve_resume_checkpoint(args.resume)
        args.resume = resolved_resume_path
        print(f"🔄 [Resume Preload] Đang kiểm tra checkpoint từ: {resolved_resume_path}")
        try:
            preloaded_ckpt = torch.load(resolved_resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            preloaded_ckpt = torch.load(resolved_resume_path, map_location="cpu")

        if not isinstance(preloaded_ckpt, dict):
            raise ValueError(f"❌ Checkpoint file không đúng định dạng dictionary (type={type(preloaded_ckpt)})")

        # Tự động kế thừa Hyperparameters từ checkpoint cũ nếu người dùng không truyền trên CLI
        ckpt_args = preloaded_ckpt.get("args", {}) if isinstance(preloaded_ckpt.get("args"), dict) else {}
        cli_specified = get_cli_specified_args(sys.argv[1:])
        sync_keys = ["epochs", "warmup_epochs", "lr", "weight_decay", "phi", "q_max", "p_srs", "out_dim", "batch_size", "model_name", "seed"]
        restored_info = []
        override_info = []

        # Ưu tiên lấy total_epochs trực tiếp từ checkpoint nếu có
        if "total_epochs" in preloaded_ckpt and "epochs" not in cli_specified:
            args.epochs = int(preloaded_ckpt["total_epochs"])
            restored_info.append(("epochs", args.epochs))

        for key in sync_keys:
            if key in ckpt_args:
                if key not in cli_specified:
                    setattr(args, key, ckpt_args[key])
                    restored_info.append((key, ckpt_args[key]))
                else:
                    override_info.append((key, getattr(args, key), ckpt_args[key]))

            print("\n" + "=" * 80)
            print(" ⚙️ [SMART HYPERPARAMETER RESTORATION] ĐỒNG BỘ SIÊU THAM SỐ TỪ CHECKPOINT")
            print("=" * 80)
            if restored_info:
                print("   ✅ Tự động kế thừa các Hyperparameters từ checkpoint cũ:")
                for k, v in restored_info:
                    print(f"      - {k:<15}: {v}")
            if override_info:
                print("   ⚡ Ưu tiên tham số người dùng chỉ định trên CLI (ghi đè checkpoint cũ):")
                for k, curr_v, old_v in override_info:
                    print(f"      - {k:<15}: {curr_v} (thay vì {old_v} từ checkpoint)")
            print("=" * 80 + "\n")

    # 1. Phát hiện phần cứng và thiết lập Multi-GPU tự động
    primary_device, num_gpus, gpu_names = get_available_devices()
    if args.device.lower() == "cpu" or num_gpus == 0:
        device = torch.device("cpu")
        num_gpus = 0
    else:
        device = primary_device

    # Kích hoạt Tensor Cores TF32 cho các phép tính ma trận FP32 trên Ampere / Ada Lovelace / Hopper
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("🚀 [Tensor Cores] Đã kích hoạt TensorFloat-32 (TF32) cho cuBLAS & cuDNN.")
        except Exception:
            pass

    print("\n" + "=" * 80)
    print(" 🚀 [DIRECTION G] VEHICLE-CENTRIC SSL PRETRAINING (TAM + AGM + SRS)")
    print("=" * 80)
    print(f" Kiến trúc Backbone   : {args.model_name}")
    print(f" Thiết bị chính        : {device}")
    print(f" Số GPU khả dụng       : {num_gpus} ({', '.join(gpu_names) if gpu_names else 'CPU Only'})")
    print(f" Thư mục lưu kết quả   : {args.output_dir}")
    print("=" * 80)
    os.makedirs(args.output_dir, exist_ok=True)

    # Khởi tạo Mixed Precision Scaler & Kiểu dữ liệu Tensor Cores
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = args.use_amp and (device.type == "cuda")
    amp_dtype = torch.float16
    use_scaler = True

    if use_amp:
        req_dtype = getattr(args, "amp_dtype", "auto").lower()
        bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        if req_dtype in ["auto", "bf16", "bfloat16"] and bf16_supported:
            amp_dtype = torch.bfloat16
            use_scaler = False  # BFloat16 có dải số 8-bit exponent tương đương FP32, miễn nhiễm tràn số, không cần GradScaler
            print("⚡ [Tensor Cores BFloat16] Phát hiện GPU hỗ trợ native BF16 -> Kích hoạt BFloat16 AMP (Bỏ qua GradScaler, chống tràn số tuyệt đối)!")
        elif req_dtype in ["bf16", "bfloat16"]:
            print("⚠️ [AMP Notice] GPU không hỗ trợ native BF16 phần cứng, tự động dùng Float16 (FP16).")
            amp_dtype = torch.float16
            use_scaler = True
        else:
            amp_dtype = torch.float16
            use_scaler = True
            print("⚡ [AMP Float16] Kích hoạt FP16 AMP kèm GradScaler.")

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device_type, enabled=(use_amp and use_scaler))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and use_scaler))

    # 2. Khởi tạo Student & Teacher backbone và Projection Heads
    effective_backbone_weights = args.weights or (args.resume if (args.resume and os.path.isfile(args.resume)) else None)
    print(f"📦 [Backbone] Nạp mô hình Student DINOv3 '{args.model_name}'...")
    student_backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.model_name,
        pretrained=True,
        weights_path=effective_backbone_weights,
        device=device,
        hf_token=args.hf_token,
    )

    print("📦 [Backbone] Khởi tạo EMA Teacher DINOv3...")
    teacher_backbone = copy.deepcopy(student_backbone)
    for p in teacher_backbone.parameters():
        p.requires_grad = False

    student_dino_head = DINOHead(in_dim=embed_dim, out_dim=args.out_dim)
    teacher_dino_head = copy.deepcopy(student_dino_head)
    for p in teacher_dino_head.parameters():
        p.requires_grad = False

    student_ibot_head = DINOHead(in_dim=embed_dim, out_dim=args.out_dim)
    teacher_ibot_head = copy.deepcopy(student_ibot_head)
    for p in teacher_ibot_head.parameters():
        p.requires_grad = False

    # Đóng gói Student & Teacher thành module hoàn chỉnh
    student_base = DirectionGStudentModel(
        backbone=student_backbone,
        dino_head=student_dino_head,
        ibot_head=student_ibot_head,
        patch_size=patch_size,
    )
    teacher_base = DirectionGTeacherModel(
        backbone=teacher_backbone,
        dino_head=teacher_dino_head,
        ibot_head=teacher_ibot_head,
        patch_size=patch_size,
    )

    # Tự động nhận diện toàn bộ GPU và kích hoạt Multi-GPU DataParallel
    student_model, device, num_gpus, effective_batch_size, effective_lr = setup_multi_gpu(
        model=student_base,
        batch_size_per_gpu=args.batch_size,
        base_lr=args.lr,
        device_arg=args.device,
        scale_lr=True,
    )
    teacher_model = teacher_base.to(device)
    if num_gpus > 1:
        teacher_model = nn.DataParallel(teacher_model)

    raw_s = unwrap_model(student_model)
    raw_t = unwrap_model(teacher_model)

    # 3. Module FrozenExtractor & PositionStats cho TAM
    print("🔍 [TAM] Khởi tạo FrozenExtractor & PositionStats...")
    # Tạo bản sao độc lập của teacher_backbone để đảm bảo:
    # 1. FrozenExtractor đóng băng tuyệt đối, không can thiệp vào requires_grad của Student
    # 2. Không cần tải lại trọng số từ HuggingFace Hub lần 2
    # 3. Toàn bộ tham số và buffers nằm chuẩn trên primary device (cuda:0)
    frozen_backbone = copy.deepcopy(raw_t.backbone)
    for p in frozen_backbone.parameters():
        p.requires_grad = False

    frozen_extractor = FrozenExtractor(
        model_name=args.model_name,
        pca_dim=64,
        device=device,
        hf_token=args.hf_token,
        backbone=frozen_backbone,
        embed_dim=embed_dim,
        patch_size=patch_size,
    ).to(device)

    for b in frozen_extractor.buffers():
        b.data = b.data.to(device)
    for p in frozen_extractor.parameters():
        p.data = p.data.to(device)

    if num_gpus > 1:
        frozen_extractor = nn.DataParallel(frozen_extractor)

    pos_stats = PositionStats(num_cams=16, num_patches=448, num_states=4, feat_dim=64).to(device)
    calibrator = GMMCalibrator(max_buffer_size=args.gmm_buffer_size, ema_alpha=args.gmm_ema_alpha)

    # 4. Losses, Metric Heads & Optimizer
    dino_loss_fn = DINOLoss(out_dim=args.out_dim, nepochs=args.epochs).to(device)
    ibot_loss_fn = iBOTPatchLoss(out_dim=args.out_dim).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)
    ric_loss_fn = RICLoss().to(device)
    cam_adv_head = CameraAdversarialHead(in_dim=embed_dim, num_cams=pos_stats.num_cams).to(device)

    # Phân nhóm tham số khoa học (Parameter Groups):
    # - Tách Backbone (giảm LR theo backbone_lr_scale) khỏi Heads (full LR) để bảo vệ pretrained DINOv3
    # - Tách bias và norm (weight_decay = 0.0) theo chuẩn kiến trúc Transformer
    decay_backbone, nodecay_backbone = [], []
    decay_heads, nodecay_heads = [], []

    raw_s = unwrap_model(student_model)
    for name, param in raw_s.backbone.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            nodecay_backbone.append(param)
        else:
            decay_backbone.append(param)

    for head in [raw_s.dino_head, raw_s.ibot_head, cam_adv_head]:
        for name, param in head.named_parameters():
            if not param.requires_grad:
                continue
            # Lớp prototype cuối cùng (last_layer.weight) được chuẩn hóa L2, TUYỆT ĐỐI KHÔNG áp dụng weight decay
            # theo chuẩn thiết kế Meta AI DINOv1/v2 để tránh bất ổn định số học khi chia cho norm nhỏ.
            if param.ndim <= 1 or name.endswith(".bias") or "last_layer" in name:
                nodecay_heads.append(param)
            else:
                decay_heads.append(param)

    backbone_lr = effective_lr * args.backbone_lr_scale
    param_groups = [
        {"params": decay_backbone, "lr": backbone_lr, "weight_decay": args.weight_decay},
        {"params": nodecay_backbone, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": decay_heads, "lr": effective_lr, "weight_decay": args.weight_decay},
        {"params": nodecay_heads, "lr": effective_lr, "weight_decay": 0.0},
    ]
    # Lọc bỏ các group rỗng
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    trainable_params = [p for g in param_groups for p in g["params"]]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = build_smooth_lr_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=1e-6,
    )

    # Khôi phục trạng thái từ file resume nếu có (chuẩn an toàn tuyệt đối)
    start_epoch = 0
    best_loss = float("inf")
    history = {"total": [], "dino": [], "ibot": [], "koleo": [], "ric": [], "cam_adv": []}

    resume_target = args.resume
    if resume_target:
        if preloaded_ckpt is not None:
            ckpt = preloaded_ckpt
        else:
            resolved_resume_path = resolve_resume_checkpoint(resume_target)
            print(f"🔄 [Resume] Đang nạp checkpoint từ: {resolved_resume_path}")
            try:
                ckpt = torch.load(resolved_resume_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(resolved_resume_path, map_location="cpu")

        if not isinstance(ckpt, dict):
            raise ValueError(f"❌ Checkpoint file không đúng định dạng dictionary (type={type(ckpt)})")

        raw_curr_s = unwrap_model(student_model)
        raw_curr_t = unwrap_model(teacher_model)
        available_keys = list(ckpt.keys())
        print(f"   🔑 [Checkpoint Keys] Các trường dữ liệu tìm thấy: {available_keys}")

        # 1. Khôi phục Student Backbone
        if "student_backbone" in ckpt:
            smart_load_state_dict(raw_curr_s.backbone, ckpt["student_backbone"], strict=False, verbose=True)
        elif "model_state" in ckpt or "state_dict" in ckpt or "model" in ckpt:
            sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt.get("model")))
            bb_sd = {k.replace("backbone.", ""): v for k, v in sd.items() if "backbone." in k}
            if bb_sd:
                smart_load_state_dict(raw_curr_s.backbone, bb_sd, strict=False, verbose=True)
            else:
                smart_load_state_dict(raw_curr_s.backbone, sd, strict=False, verbose=True)

        # 2. Khôi phục Teacher Backbone
        if "teacher_backbone" in ckpt:
            smart_load_state_dict(raw_curr_t.backbone, ckpt["teacher_backbone"], strict=False, verbose=True)
        else:
            with torch.no_grad():
                for ps, pt in zip(raw_curr_s.backbone.parameters(), raw_curr_t.backbone.parameters()):
                    pt.data.copy_(ps.data)

        # 3. Khôi phục Projection Heads
        if "student_dino_head" in ckpt:
            smart_load_state_dict(raw_curr_s.dino_head, ckpt["student_dino_head"], strict=False, verbose=False)
        if "teacher_dino_head" in ckpt:
            smart_load_state_dict(raw_curr_t.dino_head, ckpt["teacher_dino_head"], strict=False, verbose=False)
        elif "student_dino_head" in ckpt:
            smart_load_state_dict(raw_curr_t.dino_head, ckpt["student_dino_head"], strict=False, verbose=False)

        if "student_ibot_head" in ckpt:
            smart_load_state_dict(raw_curr_s.ibot_head, ckpt["student_ibot_head"], strict=False, verbose=False)
        if "teacher_ibot_head" in ckpt:
            smart_load_state_dict(raw_curr_t.ibot_head, ckpt["teacher_ibot_head"], strict=False, verbose=False)
        elif "student_ibot_head" in ckpt:
            smart_load_state_dict(raw_curr_t.ibot_head, ckpt["student_ibot_head"], strict=False, verbose=False)

        # 4. Khôi phục PositionStats (TAM)
        if "pos_stats" in ckpt:
            pos_dict = clean_state_dict(ckpt["pos_stats"])
            if "mu" in pos_dict and pos_dict["mu"].shape[0] != pos_stats.num_cams:
                cams_in_ckpt = pos_dict["mu"].shape[0]
                print(f"   ⚙️ [TAM Scale] Tự động mở rộng dung lượng camera ({pos_stats.num_cams} -> {cams_in_ckpt} cameras)...")
                pos_stats.set_cam_capacity(cams_in_ckpt)
            pos_stats.load_state_dict(pos_dict)
            print(f"   ✅ [PositionStats] Khôi phục thành công thống kê vị trí cho {pos_stats.num_cams} camera.")

        # 5. Khôi phục GMMCalibrator (Ước lượng xác suất tiền cảnh pi_t)
        if "calibrator" in ckpt and ckpt["calibrator"] is not None:
            calibrator.load_state_dict(ckpt["calibrator"])
            if "gmm_buffer_size" in cli_specified:
                calibrator.max_buffer_size = args.gmm_buffer_size
            if "gmm_ema_alpha" in cli_specified:
                calibrator.ema_alpha = args.gmm_ema_alpha
            current_samples = getattr(calibrator, "size", len(calibrator.buffer))
            print(f"   🎯 [GMMCalibrator] Khôi phục phân phối GMM: mu_bg={calibrator.mu_bg:.2f}, mu_fg={calibrator.mu_fg:.2f}, "
                  f"std_bg={calibrator.std_bg:.2f}, std_fg={calibrator.std_fg:.2f}, pi_weight={calibrator.pi_weight:.2f} "
                  f"(buffer_size={calibrator.max_buffer_size}, samples={current_samples}, ema_alpha={calibrator.ema_alpha})")
        else:
            print("   ℹ️ [GMMCalibrator] Checkpoint trước chưa lưu GMM, hệ thống sẽ tự động hiệu chuẩn sau 20 bước.")

        # 6. Khôi phục FrozenExtractor & PCA components
        raw_fe = unwrap_model(frozen_extractor)
        if "frozen_extractor" in ckpt and ckpt["frozen_extractor"] is not None:
            smart_load_state_dict(raw_fe, ckpt["frozen_extractor"], strict=False, verbose=False)
            print("   🧊 [FrozenExtractor] Khôi phục toàn bộ đặc trưng và ma trận PCA từ checkpoint.")
        elif "pca_components" in ckpt and ckpt["pca_components"] is not None:
            raw_fe.pca_components.copy_(ckpt["pca_components"].to(device))
            if "pca_mean" in ckpt and ckpt["pca_mean"] is not None:
                raw_fe.pca_mean.copy_(ckpt["pca_mean"].to(device))
            if "is_pca_fitted" in ckpt and ckpt["is_pca_fitted"] is not None:
                raw_fe.is_pca_fitted.copy_(ckpt["is_pca_fitted"].to(device))
            print("   🧊 [FrozenExtractor] Khôi phục thành công ma trận PCA buffers từ checkpoint.")

        # 7. Khôi phục Loss Centers
        if "dino_center" in ckpt and hasattr(dino_loss_fn, "center"):
            dino_loss_fn.center.copy_(ckpt["dino_center"].to(device))
        if "ibot_center" in ckpt and hasattr(ibot_loss_fn, "center"):
            ibot_loss_fn.center.copy_(ckpt["ibot_center"].to(device))

        # 8. Khôi phục Optimizer
        if "optimizer" in ckpt or "optimizer_state" in ckpt:
            opt_sd = ckpt.get("optimizer", ckpt.get("optimizer_state"))
            try:
                optimizer.load_state_dict(opt_sd)
                dest_device = torch.device(device)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(dest_device)
                print("   ✅ [Optimizer] Khôi phục toàn bộ trạng thái Optimizer.")
            except Exception as e_opt:
                print(f"   ⚠️ [Optimizer Notice] {e_opt}")

        # 9. Xác định epoch tiếp theo
        found_epoch = None
        for ep_key in ["epoch", "start_epoch", "last_epoch", "current_epoch"]:
            if ep_key in ckpt and ckpt[ep_key] is not None:
                found_epoch = int(ckpt[ep_key])
                break

        if found_epoch is not None:
            start_epoch = found_epoch + 1
            print(f"   ⏱️ [Epoch] Checkpoint ghi nhận đã hoàn thành epoch {found_epoch + 1}. Bắt đầu tiếp tục từ epoch {start_epoch + 1}")
        else:
            print("   ⚠️ [Epoch Notice] Checkpoint không có thông tin epoch. Bắt đầu từ epoch 1 với trọng số đã nạp.")

        # 10. Tự động gia hạn số epochs nếu đã đạt đích cũ
        cli_specified = get_cli_specified_args(sys.argv[1:])
        if start_epoch >= args.epochs:
            if "epochs" not in cli_specified:
                extra_epochs = args.session_epochs if args.session_epochs > 0 else 5
                if "args" in ckpt and isinstance(ckpt["args"], dict) and "session_epochs" in ckpt["args"]:
                    try:
                        extra_epochs = int(ckpt["args"]["session_epochs"])
                    except Exception:
                        extra_epochs = 5
                new_total_epochs = start_epoch + extra_epochs
                print(f"\n⚠️ [Smart Resume Notice] Checkpoint đã hoàn thành {start_epoch} epochs (mục tiêu cũ: {args.epochs}).")
                print(f"   💡 Tự động gia hạn tổng số epochs: {args.epochs} -> {new_total_epochs} (+{extra_epochs} epochs tiếp theo).")
                args.epochs = new_total_epochs
            else:
                print(f"\n⚠️ [Cảnh Báo Resume] Checkpoint đã hoàn thành {start_epoch}/{args.epochs} epochs.")
                print(f"   👉 Bạn đã chỉ định --epochs {args.epochs} <= epoch hiện tại ({start_epoch}). Vui lòng tăng --epochs > {start_epoch} để tiếp tục train!\n")

        # 11. Tái lập Lịch trình Smooth LR Scheduler liên tục (CHỐNG HIỆN TƯỢNG LOSS BẬT NGƯỢC TĂNG VỌT)
        # Sử dụng build_smooth_lr_scheduler dựa trên total_epochs kế hoạch và last_epoch = start_epoch - 1.
        # Điều này đảm bảo LR xuất phát chính xác từ điểm kết thúc của session trước, đường dốc C^0 liên tục,
        # không bao giờ bị nhảy vọt LR lên đỉnh.
        scheduler = build_smooth_lr_scheduler(
            optimizer,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            eta_min=1e-6,
            last_epoch=start_epoch - 1 if start_epoch > 0 else -1,
        )
        curr_lr = optimizer.param_groups[0]["lr"]
        remaining_epochs = max(1, args.epochs - start_epoch)
        print(f"   📈 [Scheduler Restored] Đồng bộ Lịch trình Smooth LR Scheduler: Kế hoạch tổng {args.epochs} epochs (còn {remaining_epochs} epochs).")
        print(f"   🎯 [Current Learning Rate] LR chuẩn xác cho epoch {start_epoch + 1}: {curr_lr:.3e} (liên tục 100%, triệt tiêu gradient shock).")

        # 12. Cập nhật lịch nhiệt độ teacher trong DINOLoss phù hợp với args.epochs
        if hasattr(dino_loss_fn, "teacher_temp_schedule"):
            warmup_temp_epochs = min(10, max(2, int(args.epochs * 0.35)))
            dino_loss_fn.teacher_temp_schedule = [
                0.04 + (0.07 - 0.04) * (i / max(1, warmup_temp_epochs))
                if i < warmup_temp_epochs else 0.07
                for i in range(max(args.epochs, len(dino_loss_fn.teacher_temp_schedule)))
            ]

        # 13. Khôi phục Scaler, History, Best Loss
        if "scaler" in ckpt and ckpt["scaler"] is not None and hasattr(scaler, "load_state_dict"):
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass

        if "history" in ckpt and isinstance(ckpt["history"], dict):
            history = {k: list(v) for k, v in ckpt["history"].items()}
            print(f"   📊 [History] Khôi phục toàn bộ lịch sử loss ({len(history.get('total', []))} epochs trước).")

        if "best_loss" in ckpt and ckpt["best_loss"] is not None:
            best_loss = float(ckpt["best_loss"])
        print("✅ [Resume] Đã khôi phục hoàn chỉnh và an toàn tuyệt đối toàn bộ trạng thái mô hình!")

    # 5. Dataloader: Tự động phát hiện hoặc tạo trước TAM cache, sau đó nạp dataset
    active_tam_cache = None
    if args.data_dir and os.path.isdir(args.data_dir):
        active_tam_cache = resolve_or_precompute_tam_cache(
            data_dir=args.data_dir,
            requested_cache_dir=args.tam_cache_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            device=args.device,
            batch_size=min(64, max(16, args.batch_size)),
            num_workers=args.num_workers,
            hf_token=args.hf_token,
            tam_warmup_frames=args.tam_warmup_frames,
            gmm_buffer_size=args.gmm_buffer_size,
            gmm_ema_alpha=args.gmm_ema_alpha,
        )

        # Nếu tìm thấy trọng số TAM đã fit từ cache, nạp luôn để bỏ qua warmup offline
        if active_tam_cache:
            stats_pth = os.path.join(active_tam_cache, "tam_stats.pth")
            if os.path.isfile(stats_pth):
                try:
                    s_ckpt = torch.load(stats_pth, map_location=device, weights_only=False)
                except TypeError:
                    s_ckpt = torch.load(stats_pth, map_location=device)
                if "pos_stats" in s_ckpt:
                    pos_stats.load_state_dict(s_ckpt["pos_stats"])
                if "calibrator" in s_ckpt:
                    calibrator.load_state_dict(s_ckpt["calibrator"])
                args.skip_tam_warmup = True
                print("⚡ [TAM Cache] Đã khôi phục thành công trọng số PositionStats & GMMCalibrator từ cache -> Tự động bỏ qua Warm-up offline!")

        try:
            dataset = TrafficCameraDataset(
                data_dir=args.data_dir,
                img_size=(256, 448),
                cross_cam_srs=args.cross_cam_srs,
                tam_cache_dir=active_tam_cache,
                refine_tam=args.refine_tam,
            )
            print(f"✅ [Data] Nạp thành công {len(dataset)} ảnh thực tế từ '{args.data_dir}' (phân bổ qua {dataset.num_cams} camera).")
            if dataset.num_cams > pos_stats.num_cams:
                pos_stats.ensure_cam_capacity(max(0, dataset.num_cams - 1))
        except Exception as e_data:
            print(f"⚠️ [Data] Không thể nạp ảnh từ '{args.data_dir}': {e_data}. Chuyển sang SyntheticTrafficDataset...")
            dataset = SyntheticTrafficDataset(num_samples=16, num_cams=4)
    else:
        print("💡 [Data] Chưa truyền --data_dir -> Sử dụng SyntheticTrafficDataset mô phỏng...")
        dataset = SyntheticTrafficDataset(num_samples=16, num_cams=4)

    # Thiết lập Bộ lấy mẫu phân nhóm theo Camera (CameraGroupedSampler) cho hard negatives cùng camera
    batch_sampler = None
    if args.use_cam_sampler and hasattr(dataset, "camera_to_indices"):
        valid_cam_dict = {c: idxs for c, idxs in dataset.camera_to_indices.items() if len(idxs) >= 2}
        if len(valid_cam_dict) >= 2:
            frames_per_camera = max(2, args.frames_per_cam)
            # Tự động đồng bộ cams_per_batch nếu người dùng chỉ định batch_size lớn hơn
            cams_per_batch = max(1, args.cams_per_batch)
            if args.batch_size > cams_per_batch * frames_per_camera:
                auto_cams = max(1, args.batch_size // frames_per_camera)
                print(f"⚙️ [Batch Sync] Tự động nâng cams_per_batch ({cams_per_batch} -> {auto_cams}) "
                      f"để khớp với --batch_size {args.batch_size} (với {frames_per_camera} frames/cam).")
                cams_per_batch = auto_cams

            actual_batch_size = cams_per_batch * frames_per_camera
            target_batches = None
            if args.batches_per_epoch == -1:
                target_batches = max(1, len(dataset) // actual_batch_size)
                print(f"📊 [Epoch Scale] Chế độ Full Dataset Coverage: {target_batches} batches/epoch (duyệt trọn vẹn ~{len(dataset)} ảnh).")
            elif args.batches_per_epoch > 0:
                target_batches = args.batches_per_epoch
                print(f"📊 [Epoch Scale] Thiết lập thủ công: {target_batches} batches/epoch (~{target_batches * actual_batch_size} ảnh).")
            else:
                default_batches = len(valid_cam_dict) // cams_per_batch
                print(f"ℹ️ [Epoch Notice] Chế độ Camera-Round: {default_batches} batches/epoch (mỗi epoch duyệt qua 1 lượt 608 camera, ~{default_batches * actual_batch_size} ảnh). "
                      f"Để duyệt toàn bộ {len(dataset)} ảnh mỗi epoch, hãy dùng --batches_per_epoch -1 hoặc tắt --no_cam_sampler.")

            batch_sampler = CameraGroupedSampler(
                camera_to_indices=valid_cam_dict,
                batch_cameras=cams_per_batch,
                frames_per_camera=frames_per_camera,
                batches_per_epoch=target_batches,
                shuffle=True,
                drop_last=True,
            )
            print(f"📦 [CameraGroupedSampler] Đã kích hoạt bộ lấy mẫu theo nhóm Camera: "
                  f"{cams_per_batch} cams x {frames_per_camera} frames/cam = batch {actual_batch_size} ảnh.")

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    if batch_sampler is not None:
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            drop_last=(len(dataset) > effective_batch_size),
            **loader_kwargs,
        )

    # 6. Tự động khởi động TAM Warmup offline (Fit PCA, K-means đa camera & GMM sơ bộ)
    raw_fe = unwrap_model(frozen_extractor)
    active_cams = min(getattr(dataset, "num_cams", pos_stats.num_cams), pos_stats.num_cams)
    need_tam_warmup = not args.skip_tam_warmup and (
        (not getattr(raw_fe.is_pca_fitted, "item", lambda: False)())
        or (not pos_stats.is_initialized[:active_cams].all())
        or (not calibrator.is_fitted)
    )
    if need_tam_warmup:
        warmup_tam_offline(
            frozen_extractor=frozen_extractor,
            pos_stats=pos_stats,
            calibrator=calibrator,
            dataset=dataset,
            device=device,
            num_frames_per_cam=args.tam_warmup_frames,
            kmeans_iters=args.tam_kmeans_iters,
            gmm_iters=args.gmm_iters,
            batch_size=args.batch_size,
        )

    print("🏁 [Huấn Luyện] Bắt đầu vòng lặp huấn luyện Hướng G...")
    step = start_epoch * len(loader) if start_epoch > 0 else 0
    start_time = time.time()
    last_visual_data = None

    best_checkpoint_path = os.path.join(args.output_dir, "best_checkpoint.pth")
    last_checkpoint_path = os.path.join(args.output_dir, "last_checkpoint.pth")
    latest_checkpoint_path = os.path.join(args.output_dir, "dinov3_direction1_new_latest.pth")

    def save_g_checkpoint(save_path: str, epoch_num: int, is_best: bool = False):
        raw_curr_s = unwrap_model(student_model)
        raw_curr_t = unwrap_model(teacher_model)
        raw_curr_fe = unwrap_model(frozen_extractor)
        checkpoint_dict = {
            "epoch": epoch_num,
            "total_epochs": args.epochs,
            "session_epochs": args.session_epochs,
            "warmup_epochs": args.warmup_epochs,
            "completed_epoch": epoch_num,
            "student_backbone": clean_state_dict(raw_curr_s.backbone.state_dict()),
            "teacher_backbone": clean_state_dict(raw_curr_t.backbone.state_dict()),
            "student_dino_head": clean_state_dict(raw_curr_s.dino_head.state_dict()),
            "teacher_dino_head": clean_state_dict(raw_curr_t.dino_head.state_dict()),
            "student_ibot_head": clean_state_dict(raw_curr_s.ibot_head.state_dict()),
            "teacher_ibot_head": clean_state_dict(raw_curr_t.ibot_head.state_dict()),
            "cam_adv_head": clean_state_dict(cam_adv_head.state_dict()),
            "pos_stats": clean_state_dict(pos_stats.state_dict()),
            "calibrator": calibrator.state_dict(),
            "frozen_extractor": clean_state_dict(raw_curr_fe.state_dict()),
            "pca_components": getattr(raw_curr_fe, "pca_components", None),
            "pca_mean": getattr(raw_curr_fe, "pca_mean", None),
            "is_pca_fitted": getattr(raw_curr_fe, "is_pca_fitted", None),
            "dino_center": dino_loss_fn.center.detach().cpu(),
            "ibot_center": ibot_loss_fn.center.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if (use_amp and use_scaler) else None,
            "amp_dtype": str(amp_dtype),
            "history": history,
            "best_loss": best_loss,
            "num_gpus": num_gpus,
            "effective_batch_size": effective_batch_size,
            "args": vars(args),
        }
        torch.save(checkpoint_dict, save_path)
        if is_best:
            print(f"🏆 [Checkpoint] Đã lưu mô hình tốt nhất mới (Best Loss: {best_loss:.4f}) tại: {save_path}")

    # Xác định giới hạn epoch cho phiên (session) hiện tại
    if args.session_epochs > 0:
        session_target_epoch = min(start_epoch + args.session_epochs, args.epochs)
        print(f"\n🎯 [Session Kế Hoạch] Phiên hiện tại sẽ chạy {session_target_epoch - start_epoch} epochs: "
              f"Epoch {start_epoch + 1} -> Epoch {session_target_epoch} (Kế hoạch tổng: {args.epochs} epochs)\n")
    else:
        session_target_epoch = args.epochs
        print(f"\n🎯 [Full Training] Huấn luyện liên tục từ Epoch {start_epoch + 1} -> Epoch {session_target_epoch}\n")

    total_train_steps = max(1, len(loader) * max(1, args.epochs))

    try:
        for epoch in range(start_epoch, session_target_epoch):
            student_model.train()
            epoch_losses = []
            epoch_dino_losses = []
            epoch_ibot_losses = []
            epoch_koleo_losses = []
            epoch_ric_losses = []
            epoch_cam_adv_losses = []

            pbar = tqdm(
                loader,
                desc=f"Epoch [{epoch+1}/{args.epochs}]",
                dynamic_ncols=True,
                leave=True,
            )
            for batch in pbar:
                x1 = batch["x1"].to(device, non_blocking=(device.type == "cuda"))
                x2 = batch["x2"].to(device, non_blocking=(device.type == "cuda"))
                cids = batch["cid"].to(device, non_blocking=(device.type == "cuda"))
                cids2 = batch.get("cid2", cids).to(device, non_blocking=(device.type == "cuda"))
                B = x1.shape[0]

                # Bước A: Xác định bản đồ tiền cảnh TAM (ưu tiên nạp từ cache precomputed để tăng tốc 2.5x)
                if "pi1" in batch and "pi2" in batch and batch["pi1"] is not None:
                    pi1 = batch["pi1"].to(device, non_blocking=(device.type == "cuda"))
                    pi2 = batch["pi2"].to(device, non_blocking=(device.type == "cuda"))
                else:
                    with torch.no_grad():
                        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                            u1 = frozen_extractor(x1)  # (B, 448, 64)
                            u2 = frozen_extractor(x2)  # (B, 448, 64)
                        u1_f = u1.float()
                        u2_f = u2.float()

                        a1, valid1 = pos_stats.atypicality(cids, u1_f)
                        a2, valid2 = pos_stats.atypicality(cids2, u2_f)

                        # Đẩy mẫu vào hồ chứa Reservoir Sampling với subsampling để giảm tương quan không gian nội frame
                        calibrator.push(a1, valid1, subsample_ratio=args.gmm_subsample_ratio)
                        if step % args.gmm_refit_steps == 0:
                            calibrator.refit(max_iters=args.gmm_iters)

                        if args.random_pi:
                            # Ablation: pi ngẫu nhiên
                            pi1 = torch.rand((B, 16, 28), device=device)
                            pi2 = torch.rand((B, 16, 28), device=device)
                        else:
                            pi1 = calibrator.posterior(a1).view(B, 16, 28)
                            pi2 = calibrator.posterior(a2).view(B, 16, 28)

                        # Gated Update: Bảo vệ cụm tĩnh không bị xe ở làn đông làm sai lệch
                        pos_stats.update(cids, u1_f, gate_pi=pi1.view(B, 448))

                # Bước B: Áp dụng SRS với xác suất p_srs
                x_student = x1.clone()
                for b in range(B):
                    if torch.rand(1).item() < args.p_srs:
                        x_swapped, _ = static_region_swap(
                            x1[b], x2[b], pi1[b], pi2[b],
                            ratio=0.6, feather=4,
                            continuous_bg=args.srs_continuous,
                            color_match=args.cross_cam_srs,
                        )
                        x_student[b] = x_swapped

                # Bước C: Áp dụng AGM sinh mask che phân tầng (hoặc block / random)
                masks = []
                for b in range(B):
                    m_b = agm_sample(
                        pi=pi1[b].view(-1),
                        ratio=args.mask_ratio,
                        phi=args.phi,
                        q_max=args.q_max,
                        mode=args.mask_mode,
                    )
                    masks.append(m_b)
                mask_batch = torch.stack(masks, dim=0).to(device)  # (B, 448) bool

                # Bước D: Forward Student & Teacher trong AMP autocast (Song song đa GPU)
                target_H = 16 * patch_size
                target_W = 28 * patch_size
                if x1.shape[-2] != target_H or x1.shape[-1] != target_W:
                    x1_vit = F.interpolate(x1, size=(target_H, target_W), mode="bilinear", align_corners=False)
                    x_student_vit = F.interpolate(x_student, size=(target_H, target_W), mode="bilinear", align_corners=False)
                else:
                    x1_vit = x1
                    x_student_vit = x_student

                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    # Teacher nhận ảnh x_student_vit (SRS hoán nền) SẠCH KHÔNG CHE
                    # Đồng bộ hóa bối cảnh giữa Teacher và Student: Cả hai đều nhìn cùng phương tiện trên nền mới,
                    # triệt tiêu hoàn toàn sự lệ thuộc vào góc máy camera gốc và giải quyết dứt điểm xung đột với CamAdv GRL!
                    with torch.no_grad():
                        t_cls, t_patches, t_dino_logits, t_ibot_logits = teacher_model(x_student_vit)

                    # Student nhận ảnh x_student_vit (SRS) cùng mặt nạ che phân tầng masks (AGM iBOT)
                    s_cls, s_patches, s_dino_logits, s_ibot_logits = student_model(x_student_vit, masks=mask_batch)

                    # Bước E: Tính toán toàn bộ hệ thống Loss
                    loss_dino = dino_loss_fn(s_dino_logits, t_dino_logits, epoch=epoch)
                    # Kích hoạt nhánh trọng số beta * pi trong iBOT
                    loss_ibot = ibot_loss_fn(s_ibot_logits, t_ibot_logits, mask_batch, pi=pi1.reshape(B, 448), beta=args.ibot_beta)
                    loss_koleo = koleo_loss_fn(s_cls)

                    # Regional Instance Contrastive (RIC): ép mô hình tập trung vào xe qua hard-negatives cùng camera
                    if args.use_ric:
                        z_veh_s = pool_vehicle_embedding(s_patches, pi1.reshape(B, -1))
                        z_veh_t = pool_vehicle_embedding(t_patches, pi1.reshape(B, -1))
                        loss_ric = ric_loss_fn(z_veh_s, z_veh_t, cids)
                    else:
                        loss_ric = torch.tensor(0.0, device=device)

                    # Camera Adversarial Loss (GRL): ép xóa bỏ thông tin camera ID trong CLS token
                    if args.use_cam_adv:
                        current_progress = float(epoch * len(loader) + step) / float(max(1, total_train_steps))
                        current_grl_alpha = compute_dann_alpha(current_progress, max_alpha=0.1, gamma=10.0)
                        loss_cam_adv = cam_adv_head(s_cls, cids, grl_alpha=current_grl_alpha)
                    else:
                        loss_cam_adv = torch.tensor(0.0, device=device)

                    total_loss = (
                        loss_dino
                        + 1.0 * loss_ibot
                        + 0.02 * loss_koleo
                        + args.lambda_ric * loss_ric
                        + args.lambda_cam_adv * loss_cam_adv
                    )

                # Bước F: Tối ưu Gradient qua GradScaler (cho FP16) hoặc direct backward (cho Tensor Cores BF16)
                optimizer.zero_grad()
                if use_amp and use_scaler:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
                    optimizer.step()

                # Bước G: Cập nhật EMA Teacher với Lịch trình Cosine Momentum (0.996 -> 1.0)
                cosine_progress = float(epoch) / float(max(1, args.epochs))
                momentum = 1.0 - (1.0 - 0.996) * 0.5 * (1.0 + np.cos(np.pi * cosine_progress))
                with torch.no_grad():
                    raw_curr_s = unwrap_model(student_model)
                    raw_curr_t = unwrap_model(teacher_model)
                    for param_s, param_t in zip(raw_curr_s.parameters(), raw_curr_t.parameters()):
                        param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)

                epoch_losses.append(total_loss.item())
                epoch_dino_losses.append(loss_dino.item())
                epoch_ibot_losses.append(loss_ibot.item())
                epoch_koleo_losses.append(loss_koleo.item())
                epoch_ric_losses.append(loss_ric.item())
                epoch_cam_adv_losses.append(loss_cam_adv.item())

                # Lưu mẫu dữ liệu visual của batch cuối (dùng t_patches của Teacher - ảnh sạch!)
                # Đồng thời trích xuất đặc trưng DINO gốc để phục vụ đối chứng 7 cột
                B_vis = min(4, x1.shape[0])
                with torch.no_grad():
                    x1_vis = x1[:B_vis].to(device)
                    x1_vis_norm = imagenet_normalize(x1_vis)
                    _, v_patch_spatial = extract_tokens(frozen_backbone, x1_vis_norm, patch_size=patch_size)
                    v_patch_tokens = v_patch_spatial.view(B_vis, -1, embed_dim).cpu().float()

                    # Đảm bảo các mẫu visual luôn được áp dụng SRS để hiển thị tác động hoán nền thực tế
                    x_srs_vis = x_student[:B_vis].clone()
                    for b_idx in range(B_vis):
                        diff_check = float(torch.abs(x_srs_vis[b_idx] - x1[b_idx]).mean().item() * 100.0)
                        if diff_check < 0.5:
                            # Mẫu này bị bỏ qua p_srs trong lúc train, thực hiện swap bổ sung để trực quan hóa
                            x_sw_vis, _ = static_region_swap(
                                x1[b_idx].to(device), x2[b_idx].to(device),
                                pi1[b_idx].to(device), pi2[b_idx].to(device),
                                ratio=0.7, feather=4, continuous_bg=True, color_match=True,
                            )
                            x_srs_vis[b_idx] = x_sw_vis.cpu()

                last_visual_data = {
                    "x_orig": x1[:B_vis].detach().cpu(),
                    "pi": pi1[:B_vis].detach().cpu(),
                    "mask": mask_batch[:B_vis].detach().cpu(),
                    "x_srs": x_srs_vis.detach().cpu(),
                    "patch_features": t_patches[:B_vis].detach().cpu().float(),
                    "vanilla_patch_features": v_patch_tokens,
                }

                step += 1
                cur_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "dino": f"{loss_dino.item():.4f}",
                    "ibot": f"{loss_ibot.item():.4f}",
                    "koleo": f"{loss_koleo.item():.3f}",
                    "ric": f"{loss_ric.item():.3f}",
                    "lr": f"{cur_lr:.1e}",
                })

            # Cập nhật lịch trình Learning Rate Cosine Annealing
            scheduler.step()

            # Kết thúc epoch: Tính giá trị loss trung bình
            avg_tot = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            avg_dino = float(np.mean(epoch_dino_losses)) if epoch_dino_losses else 0.0
            avg_ibot = float(np.mean(epoch_ibot_losses)) if epoch_ibot_losses else 0.0
            avg_koleo = float(np.mean(epoch_koleo_losses)) if epoch_koleo_losses else 0.0
            avg_ric = float(np.mean(epoch_ric_losses)) if epoch_ric_losses else 0.0
            avg_cam_adv = float(np.mean(epoch_cam_adv_losses)) if epoch_cam_adv_losses else 0.0

            history["total"].append(avg_tot)
            history["dino"].append(avg_dino)
            history["ibot"].append(avg_ibot)
            history["koleo"].append(avg_koleo)
            history["ric"].append(avg_ric)
            history["cam_adv"].append(avg_cam_adv)

            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"📊 [Epoch {epoch+1}/{args.epochs} Hoàn Tất] LR: {cur_lr:.2e} — Loss TB: {avg_tot:.4f} "
                  f"(DINO: {avg_dino:.4f}, iBOT: {avg_ibot:.4f}, KoLeo: {avg_koleo:.4f}, "
                  f"RIC: {avg_ric:.4f}, CamAdv: {avg_cam_adv:.4f})")

            # Giám sát phân bố Teacher-Student để phát hiện sớm bất thường
            try:
                dist_metrics = dino_loss_fn.compute_distribution_metrics(s_dino_logits, t_dino_logits, epoch=epoch)
                print(f"   📈 [DINO Giám Sát] Teacher Temp: {dist_metrics['teacher_temp']} | Teacher H: {dist_metrics['teacher_entropy']} | "
                      f"Student H: {dist_metrics['student_entropy']} | KL Div: {dist_metrics['kl_div']} | Prototypes: {dist_metrics['active_prototypes']}")
            except Exception:
                pass

            # Cập nhật hình ảnh trực quan hóa và đồ thị sau mỗi epoch
            if last_visual_data is not None:
                vis_save_path = os.path.join(args.output_dir, "direction1_new_progress.png")
                save_direction_g_visuals(
                    x_orig=last_visual_data["x_orig"],
                    pi=last_visual_data["pi"],
                    mask=last_visual_data["mask"],
                    x_srs=last_visual_data["x_srs"],
                    patch_features=last_visual_data["patch_features"],
                    save_path=vis_save_path,
                    epoch=epoch + 1,
                    vanilla_patch_features=last_visual_data.get("vanilla_patch_features"),
                )

            # Cập nhật đồ thị đường cong hàm mất mát sau mỗi epoch (Live Loss Curve)
            loss_curve_path = os.path.join(args.output_dir, "loss_curve.png")
            save_loss_curves(history, loss_curve_path)

            # Cập nhật file tóm tắt chỉ số Metrics JSON sau mỗi epoch (đảm bảo an toàn khi crash hoặc restart)
            metrics_path = os.path.join(args.output_dir, "training_metrics.json")
            try:
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "history": history,
                        "current_epoch": epoch + 1,
                        "total_epochs": args.epochs,
                        "best_loss": best_loss,
                        "last_lr": cur_lr,
                    }, f, indent=2, ensure_ascii=False)
            except Exception as e_json:
                print(f"⚠️ [JSON Metrics Error] Không thể lưu metrics.json: {e_json}")

            # Lưu checkpoint từng epoch: last_checkpoint và latest_checkpoint
            save_g_checkpoint(last_checkpoint_path, epoch_num=epoch)
            save_g_checkpoint(latest_checkpoint_path, epoch_num=epoch)

            # Nếu đạt kỷ lục loss thấp nhất, lưu best_checkpoint.pth
            if avg_tot < best_loss:
                best_loss = avg_tot
                save_g_checkpoint(best_checkpoint_path, epoch_num=epoch, is_best=True)

    except KeyboardInterrupt:
        print("\n⚠️ [Dừng Sớm] Nhận tín hiệu ngắt (Ctrl+C). Đang tiến hành lưu khẩn cấp trạng thái và hình ảnh...")

    # 6. Xuất đồ thị hàm mất mát Loss Curve
    loss_curve_path = os.path.join(args.output_dir, "loss_curve.png")
    save_loss_curves(history, loss_curve_path)

    # 7. Xuất file tóm tắt chỉ số Metrics JSON
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "final_epoch": len(history["total"]),
            "session_target_epoch": session_target_epoch,
            "total_planned_epochs": args.epochs,
            "best_loss": best_loss,
            "num_gpus": num_gpus,
            "effective_batch_size": effective_batch_size,
            "args": vars(args),
            "total_time_seconds": time.time() - start_time,
        }, f, indent=2, ensure_ascii=False)
    print(f"📈 [Metrics] Đã lưu thông số chi tiết tại: {metrics_path}")

    # 8. Lưu checkpoint toàn diện cuối cùng
    save_g_checkpoint(latest_checkpoint_path, epoch_num=len(history["total"]) - 1)

    elapsed_time = time.time() - start_time
    vis_path = os.path.join(args.output_dir, "direction1_new_progress.png")

    if session_target_epoch < args.epochs:
        next_session_epochs = min(args.session_epochs, args.epochs - session_target_epoch) if args.session_epochs > 0 else (args.epochs - session_target_epoch)
        next_end = session_target_epoch + next_session_epochs
        print("\n" + "╔" + "═" * 78 + "╗")
        print(f"║  🎉 [SESSION HOÀN TẤT] Đã chạy xong {session_target_epoch}/{args.epochs} epochs cho phiên làm việc này!  ║")
        print(f"║  💾 Checkpoint an toàn đã được lưu tại:                                      ║")
        print(f"║     {latest_checkpoint_path:<72} ║")
        print(f"║  📊 Đồ thị loss tích lũy: {loss_curve_path:<50} ║")
        print(f"║                                                                              ║")
        print(f"║  👉 Để chạy tiếp Session kế tiếp (Epoch {session_target_epoch + 1} -> {next_end}), bạn chỉ cần chạy lệnh:    ║")
        cmd_resume = f"python train.py --resume {latest_checkpoint_path}"
        if args.data_dir:
            cmd_resume += f" --data_dir {args.data_dir}"
        if args.session_epochs > 0:
            cmd_resume += f" --session_epochs {args.session_epochs}"
        print(f"║     {cmd_resume:<72} ║")
        print("╚" + "═" * 78 + "╝\n")
    else:
        print("\n" + "╔" + "═" * 78 + "╗")
        print(f"║  🏆 [XUẤT SẮC] ĐÃ HOÀN TẤT TOÀN BỘ {args.epochs}/{args.epochs} EPOCHS THEO KẾ HOẠCH!              ║")
        print(f"║  💾 Checkpoint tối ưu nhất (Best Loss) : {best_checkpoint_path:<36} ║")
        print(f"║  💾 Checkpoint phiên bản mới nhất      : {latest_checkpoint_path:<36} ║")
        print(f"║  🖼️ Ảnh trực quan hóa chất lượng cao  : {vis_path:<36} ║")
        print(f"║  📊 Toàn bộ đồ thị hàm mất mát Loss    : {loss_curve_path:<36} ║")
        print(f"║  ⏱️ Tổng thời gian thực thi            : {elapsed_time:.2f}s                                  ║")
        print("╚" + "═" * 78 + "╝\n")

    return latest_checkpoint_path


if __name__ == "__main__":
    train_direction_g()
