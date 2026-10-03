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

from common.backbone_loader import get_dino_backbone, extract_tokens
from directionG_camera_ssl.features import FrozenExtractor
from directionG_camera_ssl.tam import PositionStats, GMMCalibrator
from directionG_camera_ssl.masking import agm_sample
from directionG_camera_ssl.srs import static_region_swap
from directionG_camera_ssl.losses import DINOLoss, iBOTPatchLoss, KoLeoLoss


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


class TrafficCameraDataset(Dataset):
    """
    Dataset nạp ảnh giao thông thực tế từ thư mục (ví dụ: 'output').
    Tự động nhóm các frame theo camera để phục vụ cơ chế ghép vùng tĩnh SRS (Static Region Swap).
    """
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (256, 448)):
        super().__init__()
        self.data_dir = data_dir
        self.H, self.W = img_size

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

        # Danh sách phẳng các mẫu
        self.samples = []
        for cam_key, files in self.cam_to_files.items():
            cid = self.cam_to_id[cam_key]
            for f in files:
                self.samples.append((f, cid, cam_key))

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
        path1, cid, cam_key = self.samples[idx]
        x1 = self._load_img(path1)

        # Chọn frame x2 cùng camera (khác thời điểm) để phục vụ SRS
        cam_files = self.cam_to_files[cam_key]
        if len(cam_files) > 1:
            candidates = [p for p in cam_files if p != path1]
            path2 = random.choice(candidates) if candidates else path1
        else:
            path2 = path1

        x2 = self._load_img(path2)
        return {"x1": x1, "x2": x2, "cid": cid}


def save_direction_g_visuals(
    x_orig: torch.Tensor,
    pi: torch.Tensor,
    mask: torch.Tensor,
    x_srs: torch.Tensor,
    patch_features: Optional[torch.Tensor],
    save_path: str,
    epoch: int = 1,
):
    """
    Xuất biểu đồ 5 cột trực quan hóa quá trình học Tự Giám Sát Hướng G:
      Cột 1: Ảnh gốc frame x1
      Cột 2: Bản đồ tiền cảnh bất thường TAM pi (Heatmap)
      Cột 3: Mặt nạ che phân tầng AGM (Mask)
      Cột 4: Ảnh sau khi hoán đổi vùng tĩnh SRS
      Cột 5: Bản đồ đặc trưng DINOv3 PCA Feature Map (RGB)
    """
    B = min(4, x_orig.shape[0])
    fig, axes = plt.subplots(B, 5, figsize=(18, 3.2 * B), dpi=150)
    if B == 1:
        axes = np.expand_dims(axes, axis=0)

    for b in range(B):
        # 1. Ảnh gốc
        img_np = x_orig[b].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        axes[b, 0].imshow(img_np)
        axes[b, 0].set_title(f"Mẫu #{b+1} — Ảnh Gốc $x_1$", fontsize=10, fontweight="bold")
        axes[b, 0].axis("off")

        # 2. TAM Heatmap (16, 28) nội suy lên kích thước ảnh
        pi_np = pi[b].detach().cpu().numpy()
        axes[b, 1].imshow(img_np)
        axes[b, 1].imshow(pi_np, cmap="jet", alpha=0.6, extent=[0, img_np.shape[1], img_np.shape[0], 0])
        axes[b, 1].set_title("Bản Đồ Tiền Cảnh TAM $\\pi$", fontsize=10, fontweight="bold")
        axes[b, 1].axis("off")

        # 3. AGM Mask (448 patches -> 16x28)
        mask_np = mask[b].view(16, 28).detach().cpu().numpy().astype(float)
        axes[b, 2].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[b, 2].set_title("Mặt Nạ Che Phân Tầng AGM", fontsize=10, fontweight="bold")
        axes[b, 2].axis("off")

        # 4. SRS Swapped Image
        srs_np = x_srs[b].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        axes[b, 3].imshow(srs_np)
        axes[b, 3].set_title("Ảnh Ghép Nền Tĩnh SRS", fontsize=10, fontweight="bold")
        axes[b, 3].axis("off")

        # 5. DINOv3 PCA Feature Map
        if patch_features is not None:
            feats = patch_features[b].detach().cpu().numpy()  # (448, D)
            feats_norm = feats - feats.mean(axis=0)
            u, s, vt = np.linalg.svd(feats_norm, full_matrices=False)
            pca3 = u[:, :3]  # (448, 3)
            pca3 = (pca3 - pca3.min(axis=0)) / (pca3.max(axis=0) - pca3.min(axis=0) + 1e-6)
            pca_img = pca3.reshape(16, 28, 3)
            axes[b, 4].imshow(pca_img)
            axes[b, 4].set_title("DINOv3 PCA Representation", fontsize=10, fontweight="bold")
        else:
            axes[b, 4].axis("off")
        axes[b, 4].axis("off")

    plt.suptitle(f"Tiến Trình Huấn Luyện SSL Hướng G — Epoch {epoch}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"🖼️ [Trực Quan Hóa] Đã xuất biểu đồ học biểu diễn tại: {save_path}")


def save_loss_curves(history: Dict[str, List[float]], save_path: str):
    """Vẽ và lưu biểu đồ đường cong hàm mất mát theo từng epoch."""
    if not history["total"]:
        return
    epochs = list(range(1, len(history["total"]) + 1))
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(epochs, history["total"], "b-o", linewidth=2, label="Total Loss")
    plt.plot(epochs, history["dino"], "g--s", linewidth=1.5, label="DINO CLS Loss")
    plt.plot(epochs, history["ibot"], "m-.^", linewidth=1.5, label="iBOT Patch Loss")
    plt.plot(epochs, history["koleo"], "r:x", linewidth=1.5, label="KoLeo Entropy Loss")

    plt.title("Diễn Biến Hàm Mất Mát Huấn Luyện SSL DINOv3 (Hướng G)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss Value", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"📊 [Đồ Thị] Đã lưu đường cong tổn thất tại: {save_path}")


class SyntheticTrafficDataset(Dataset):
    """Dataset giả lập phục vụ chạy demo hoặc khi chưa liên kết dataset thực tế."""
    def __init__(self, num_samples: int = 200, num_cams: int = 4):
        self.num_samples = num_samples
        self.num_cams = num_cams

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        cid = idx % self.num_cams
        # Frame 1: 3 x 256 x 448
        x1 = torch.rand(3, 256, 448) * 0.6 + 0.2
        # Frame 2 (cùng camera, khác ngày) để phục vụ SRS
        x2 = x1 * 0.9 + 0.1 * torch.rand(3, 256, 448)
        return {"x1": x1, "x2": x2, "cid": cid}


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Tự Giám Sát DINOv3 Hướng G (TAM + AGM + SRS)")
    parser.add_argument("--data_dir", "--origin_dir", dest="data_dir", type=str, default=None, help="Thư mục ảnh giao thông thực tế (ví dụ: output)")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16", help="Tên backbone DINOv3")
    parser.add_argument("--batch_size", type=int, default=4, help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=5, help="Số epochs huấn luyện")
    parser.add_argument("--lr", type=float, default=5e-5, help="Tốc độ học")
    parser.add_argument("--weight_decay", type=float, default=0.04, help="Hệ số weight decay")
    parser.add_argument("--phi", type=float, default=0.5, help="Tỷ lệ che ngân sách xe AGM")
    parser.add_argument("--q_max", type=float, default=0.6, help="Tỷ lệ tối đa patch xe bị che AGM")
    parser.add_argument("--p_srs", type=float, default=0.5, help="Xác suất áp dụng SRS")
    parser.add_argument("--out_dim", type=int, default=4096, help="Số prototype DINO/iBOT head")
    parser.add_argument("--output_dir", type=str, default="checkpoints/directionG", help="Thư mục lưu weights")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn custom checkpoint ban đầu (.pth / .safetensors)")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace Token")
    parsed = parser.parse_args()
    if parsed.hf_token:
        os.environ["HF_TOKEN"] = parsed.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = parsed.hf_token
    return parsed


def train_direction_g():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [Direction G] Khởi chạy huấn luyện SSL trên thiết bị: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Khởi tạo Student & Teacher backbone
    print(f"📦 [Backbone] Nạp mô hình Student DINOv3 '{args.model_name}'...")
    student_backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.model_name,
        pretrained=True,
        weights_path=args.weights,
        device=device,
        hf_token=args.hf_token,
    )

    print("📦 [Backbone] Khởi tạo EMA Teacher DINOv3...")
    teacher_backbone = copy.deepcopy(student_backbone)
    for p in teacher_backbone.parameters():
        p.requires_grad = False

    # 2. Đầu chiếu DINO và iBOT Heads
    student_dino_head = DINOHead(in_dim=embed_dim, out_dim=args.out_dim).to(device)
    teacher_dino_head = DINOHead(in_dim=embed_dim, out_dim=args.out_dim).to(device)
    teacher_dino_head.load_state_dict(student_dino_head.state_dict())
    for p in teacher_dino_head.parameters():
        p.requires_grad = False

    student_ibot_head = DINOHead(in_dim=embed_dim, out_dim=args.out_dim).to(device)
    teacher_ibot_head = DINOHead(in_dim=embed_dim, out_dim=args.out_dim).to(device)
    teacher_ibot_head.load_state_dict(student_ibot_head.state_dict())
    for p in teacher_ibot_head.parameters():
        p.requires_grad = False

    # 3. Module FrozenExtractor & PositionStats cho TAM
    print("🔍 [TAM] Khởi tạo FrozenExtractor & PositionStats...")
    frozen_extractor = FrozenExtractor(
        model_name=args.model_name,
        pca_dim=64,
        device=device,
        hf_token=args.hf_token,
        backbone=student_backbone,
        embed_dim=embed_dim,
        patch_size=patch_size,
    )
    pos_stats = PositionStats(num_cams=16, num_patches=448, num_states=4, feat_dim=64).to(device)
    calibrator = GMMCalibrator()

    # 4. Losses & Optimizer
    dino_loss_fn = DINOLoss(out_dim=args.out_dim, nepochs=args.epochs).to(device)
    ibot_loss_fn = iBOTPatchLoss(out_dim=args.out_dim).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)

    trainable_params = (
        list(student_backbone.parameters())
        + list(student_dino_head.parameters())
        + list(student_ibot_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # 5. Dataloader: Ưu tiên nạp dữ liệu thực tế từ args.data_dir
    if args.data_dir and os.path.isdir(args.data_dir):
        try:
            dataset = TrafficCameraDataset(data_dir=args.data_dir, img_size=(256, 448))
            print(f"✅ [Data] Nạp thành công {len(dataset)} ảnh thực tế từ '{args.data_dir}' (phân bổ qua {dataset.num_cams} camera).")
            if dataset.num_cams > pos_stats.num_cams:
                pos_stats.ensure_cam_capacity(dataset.num_cams)
        except Exception as e_data:
            print(f"⚠️ [Data] Không thể nạp ảnh từ '{args.data_dir}': {e_data}. Chuyển sang SyntheticTrafficDataset...")
            dataset = SyntheticTrafficDataset(num_samples=16, num_cams=4)
    else:
        print("💡 [Data] Chưa truyền --data_dir -> Sử dụng SyntheticTrafficDataset mô phỏng...")
        dataset = SyntheticTrafficDataset(num_samples=16, num_cams=4)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    print("🏁 [Huấn Luyện] Bắt đầu vòng lặp huấn luyện Hướng G...")
    step = 0
    start_time = time.time()
    history = {"total": [], "dino": [], "ibot": [], "koleo": []}
    last_visual_data = None

    try:
        for epoch in range(args.epochs):
            student_backbone.train()
            student_dino_head.train()
            student_ibot_head.train()

            epoch_losses = []
            epoch_dino_losses = []
            epoch_ibot_losses = []
            epoch_koleo_losses = []

            for batch in loader:
                x1 = batch["x1"].to(device)
                x2 = batch["x2"].to(device)
                cids = batch["cid"].to(device)
                B = x1.shape[0]

                # Bước A: Trích xuất đặc trưng đóng băng cho TAM
                with torch.no_grad():
                    u1 = frozen_extractor(x1)  # (B, 448, 64)
                    u2 = frozen_extractor(x2)  # (B, 448, 64)
                    pos_stats.update(cids, u1)
                    a1, valid1 = pos_stats.atypicality(cids, u1)
                    a2, valid2 = pos_stats.atypicality(cids, u2)

                    calibrator.push(a1, valid1)
                    if step % 20 == 0:
                        calibrator.refit()
                    pi1 = calibrator.posterior(a1).view(B, 16, 28)
                    pi2 = calibrator.posterior(a2).view(B, 16, 28)

                # Bước B: Áp dụng SRS với xác suất p_srs
                x_student = x1.clone()
                for b in range(B):
                    if torch.rand(1).item() < args.p_srs:
                        x_swapped, _ = static_region_swap(
                            x1[b], x2[b], pi1[b], pi2[b],
                            ratio=0.6, feather=4,
                        )
                        x_student[b] = x_swapped

                # Bước C: Áp dụng AGM sinh mask che phân tầng
                masks = []
                for b in range(B):
                    m_b = agm_sample(
                        pi=pi1[b].view(-1),
                        ratio=0.35,
                        phi=args.phi,
                        q_max=args.q_max,
                    )
                    masks.append(m_b)
                mask_batch = torch.stack(masks, dim=0).to(device)  # (B, 448) bool

                # Bước D: Forward Student & Teacher
                target_H = 16 * patch_size
                target_W = 28 * patch_size
                if x1.shape[-2] != target_H or x1.shape[-1] != target_W:
                    x1_vit = F.interpolate(x1, size=(target_H, target_W), mode="bilinear", align_corners=False)
                    x_student_vit = F.interpolate(x_student, size=(target_H, target_W), mode="bilinear", align_corners=False)
                else:
                    x1_vit = x1
                    x_student_vit = x_student

                # Teacher nhận ảnh gốc x1_vit không che
                with torch.no_grad():
                    t_cls, t_patches_spatial = extract_tokens(teacher_backbone, x1_vit, patch_size=patch_size)
                    t_patches = t_patches_spatial.reshape(B, 448, embed_dim)
                    t_dino_logits = teacher_dino_head(t_cls)
                    t_ibot_logits = teacher_ibot_head(t_patches)

                # Student nhận ảnh x_student_vit (đã qua SRS)
                s_cls, s_patches_spatial = extract_tokens(student_backbone, x_student_vit, patch_size=patch_size)
                s_patches = s_patches_spatial.reshape(B, 448, embed_dim)
                s_dino_logits = student_dino_head(s_cls)
                s_ibot_logits = student_ibot_head(s_patches)

                # Bước E: Tính tổn thất Loss
                loss_dino = dino_loss_fn(s_dino_logits, t_dino_logits, epoch=epoch)
                loss_ibot = ibot_loss_fn(s_ibot_logits, t_ibot_logits, mask_batch, pi=pi1.reshape(B, 448))
                loss_koleo = koleo_loss_fn(s_cls)
                total_loss = loss_dino + 1.0 * loss_ibot + 0.1 * loss_koleo

                # Bước F: Tối ưu Gradient
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
                optimizer.step()

                # Bước G: Cập nhật EMA Teacher (momentum 0.996)
                momentum = 0.996
                with torch.no_grad():
                    for param_s, param_t in zip(student_backbone.parameters(), teacher_backbone.parameters()):
                        param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)
                    for param_s, param_t in zip(student_dino_head.parameters(), teacher_dino_head.parameters()):
                        param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)
                    for param_s, param_t in zip(student_ibot_head.parameters(), teacher_ibot_head.parameters()):
                        param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.detach().data)

                epoch_losses.append(total_loss.item())
                epoch_dino_losses.append(loss_dino.item())
                epoch_ibot_losses.append(loss_ibot.item())
                epoch_koleo_losses.append(loss_koleo.item())

                # Lưu mẫu dữ liệu visual của batch cuối
                last_visual_data = {
                    "x_orig": x1.detach().cpu(),
                    "pi": pi1.detach().cpu(),
                    "mask": mask_batch.detach().cpu(),
                    "x_srs": x_student.detach().cpu(),
                    "patch_features": s_patches.detach().cpu(),
                }

                step += 1
                if step % 2 == 0 or step == 1:
                    print(f"   [Epoch {epoch+1}/{args.epochs} | Step {step}] Loss: {total_loss.item():.4f} "
                          f"(DINO: {loss_dino.item():.4f}, iBOT: {loss_ibot.item():.4f}, KoLeo: {loss_koleo.item():.4f})")

            # Kết thúc epoch: Tính giá trị loss trung bình
            avg_tot = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            avg_dino = float(np.mean(epoch_dino_losses)) if epoch_dino_losses else 0.0
            avg_ibot = float(np.mean(epoch_ibot_losses)) if epoch_ibot_losses else 0.0
            avg_koleo = float(np.mean(epoch_koleo_losses)) if epoch_koleo_losses else 0.0

            history["total"].append(avg_tot)
            history["dino"].append(avg_dino)
            history["ibot"].append(avg_ibot)
            history["koleo"].append(avg_koleo)

            print(f"🌟 [Epoch {epoch+1}/{args.epochs} Hoàn Tất] Loss TB: {avg_tot:.4f} "
                  f"(DINO: {avg_dino:.4f}, iBOT: {avg_ibot:.4f}, KoLeo: {avg_koleo:.4f})")

            # Cập nhật hình ảnh trực quan hóa và đồ thị sau mỗi epoch (hoặc epoch cuối)
            if last_visual_data is not None:
                vis_save_path = os.path.join(args.output_dir, "directionG_progress.png")
                save_direction_g_visuals(
                    x_orig=last_visual_data["x_orig"],
                    pi=last_visual_data["pi"],
                    mask=last_visual_data["mask"],
                    x_srs=last_visual_data["x_srs"],
                    patch_features=last_visual_data["patch_features"],
                    save_path=vis_save_path,
                    epoch=epoch + 1,
                )

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
            "args": vars(args),
            "total_time_seconds": time.time() - start_time,
        }, f, indent=2, ensure_ascii=False)
    print(f"📈 [Metrics] Đã lưu thông số chi tiết tại: {metrics_path}")

    # 8. Lưu checkpoint toàn diện
    save_path = os.path.join(args.output_dir, "dinov3_directionG_latest.pth")
    torch.save({
        "student_backbone": student_backbone.state_dict(),
        "teacher_backbone": teacher_backbone.state_dict(),
        "student_dino_head": student_dino_head.state_dict(),
        "student_ibot_head": student_ibot_head.state_dict(),
        "pos_stats": pos_stats.state_dict(),
        "history": history,
        "args": vars(args),
    }, save_path)

    elapsed_time = time.time() - start_time
    vis_path = os.path.join(args.output_dir, "directionG_progress.png")

    print("\n" + "=" * 80)
    print(" 🎉 [HOÀN TẤT HUẤN LUYỆN TỰ GIÁM SÁT HƯỚNG G]")
    print("=" * 80)
    print(f" 💾 Checkpoint Weights        : {save_path}")
    print(f" 🖼️ Ảnh Trực Quan Hóa (Visual): {vis_path}")
    print(f" 📊 Đồ Thị Hàm Mất Mát (Loss) : {loss_curve_path}")
    print(f" 📈 Nhật Ký Huấn Luyện (JSON) : {metrics_path}")
    print(f" ⏱️ Tổng Thời Gian Thực Thi   : {elapsed_time:.2f}s")
    print("=" * 80 + "\n")

    return save_path


if __name__ == "__main__":
    train_direction_g()
