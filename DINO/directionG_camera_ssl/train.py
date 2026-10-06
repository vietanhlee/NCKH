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

        # 5. DINOv3 PCA Feature Map (Nội suy Bilinear mượt mà chuẩn Meta DINOv2/v3)
        if patch_features is not None:
            feats = patch_features[b].detach().cpu().numpy()  # (448, D)
            feats_norm = feats - feats.mean(axis=0, keepdims=True)
            u, s, vt = np.linalg.svd(feats_norm, full_matrices=False)
            pca3 = u[:, :3]  # (448, 3)
            # Chuẩn hóa min-max trên từng kênh thành phần chính
            p_min = pca3.min(axis=0, keepdims=True)
            p_max = pca3.max(axis=0, keepdims=True)
            pca3_scaled = (pca3 - p_min) / np.maximum(p_max - p_min, 1e-6)
            pca_grid = pca3_scaled.reshape(16, 28, 3)

            # Nội suy song tuyến tính (Bilinear Interpolation) lên kích thước ảnh thật
            # Khắc phục hoàn toàn hiện tượng rỗ ô vuông 16x16 thô ráp
            H_orig, W_orig = img_np.shape[0], img_np.shape[1]
            pca_tensor = torch.from_numpy(pca_grid).permute(2, 0, 1).unsqueeze(0).float()
            pca_smooth = F.interpolate(pca_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
            pca_smooth_np = pca_smooth.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)

            axes[b, 4].imshow(pca_smooth_np)
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
    parser.add_argument("--model_name", "--backbone", dest="model_name", type=str, default="dinov3_vits16", help="Tên backbone DINOv3")
    parser.add_argument("--batch_size", type=int, default=4, help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=5, help="Số epochs huấn luyện")
    parser.add_argument("--lr", type=float, default=5e-5, help="Tốc độ học")
    parser.add_argument("--weight_decay", type=float, default=0.04, help="Hệ số weight decay")
    parser.add_argument("--phi", type=float, default=0.5, help="Tỷ lệ che ngân sách xe AGM")
    parser.add_argument("--q_max", type=float, default=0.6, help="Tỷ lệ tối đa patch xe bị che AGM")
    parser.add_argument("--p_srs", type=float, default=0.5, help="Xác suất áp dụng SRS")
    parser.add_argument("--out_dim", type=int, default=4096, help="Số prototype DINO/iBOT head")
    parser.add_argument("--output_dir", "--save_dir", dest="output_dir", type=str, default="checkpoints/directionG", help="Thư mục lưu weights")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn custom checkpoint ban đầu (.pth / .safetensors)")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán ('cuda', 'cuda:0', hoặc 'cpu')")
    parser.add_argument("--num_workers", type=int, default=0, help="Số luồng nạp dữ liệu DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Sử dụng Mixed Precision (AMP) để tối ưu VRAM và tăng tốc độ")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Tắt Mixed Precision")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint (.pth) để tiếp tục huấn luyện")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace Token")
    parsed, unknown = parser.parse_known_args()
    if unknown:
        print(f"⚠️ [CLI Warning] Bỏ qua các đối số chưa khai báo: {unknown}")
    if parsed.hf_token:
        os.environ["HF_TOKEN"] = parsed.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = parsed.hf_token
    return parsed


def train_direction_g():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1. Phát hiện phần cứng và thiết lập Multi-GPU tự động
    primary_device, num_gpus, gpu_names = get_available_devices()
    if args.device.lower() == "cpu" or num_gpus == 0:
        device = torch.device("cpu")
        num_gpus = 0
    else:
        device = primary_device

    print("\n" + "=" * 80)
    print(" 🚀 [DIRECTION G] VEHICLE-CENTRIC SSL PRETRAINING (TAM + AGM + SRS)")
    print("=" * 80)
    print(f" Kiến trúc Backbone   : {args.model_name}")
    print(f" Thiết bị chính        : {device}")
    print(f" Số GPU khả dụng       : {num_gpus} ({', '.join(gpu_names) if gpu_names else 'CPU Only'})")
    print(f" Thư mục lưu kết quả   : {args.output_dir}")
    print("=" * 80)
    os.makedirs(args.output_dir, exist_ok=True)

    # Khởi tạo Mixed Precision Scaler
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = args.use_amp and (device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device_type, enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 2. Khởi tạo Student & Teacher backbone và Projection Heads
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
    calibrator = GMMCalibrator()

    # 4. Losses & Optimizer
    dino_loss_fn = DINOLoss(out_dim=args.out_dim, nepochs=args.epochs).to(device)
    ibot_loss_fn = iBOTPatchLoss(out_dim=args.out_dim).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)

    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=effective_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Khôi phục trạng thái từ file resume nếu có (chuẩn an toàn tuyệt đối)
    start_epoch = 0
    best_loss = float("inf")
    history = {"total": [], "dino": [], "ibot": [], "koleo": []}

    resume_target = args.resume
    if resume_target:
        # 1. Hỗ trợ tự động sửa đường dẫn Kaggle nếu người dùng copy nhầm URL 'models/<user>/<slug>/'
        if not os.path.exists(resume_target):
            norm_path = resume_target.replace("\\", "/")
            if "/kaggle/input/" in norm_path:
                subparts = norm_path.split("/kaggle/input/")[1].split("/")
                # Nếu có dạng 'models/<user>/<slug>/...' -> Thử chuyển thành '/kaggle/input/<slug>/...'
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
            for candidate in ["last_checkpoint.pth", "dinov3_directionG_latest.pth", "best_checkpoint.pth"]:
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
                raise FileNotFoundError(f"❌ [Resume Error] Thư mục '{args.resume}' không chứa bất kỳ file checkpoint (.pth) nào!")

        # 3. Báo lỗi rõ ràng nếu không tìm thấy file thay vì im lặng train mới
        if not os.path.isfile(resume_target):
            import glob
            pths_found = glob.glob("/kaggle/input/**/*.pth", recursive=True)[:5] if os.path.isdir("/kaggle/input") else []
            err_msg = (
                f"\n❌ [LỖI RESUME] Không tìm thấy file checkpoint tại: '{resume_target}'\n"
                f"   Nguyên nhân: Đường dẫn file trên Kaggle không tồn tại.\n"
            )
            if pths_found:
                err_msg += f"   💡 Gợi ý các file .pth hiện có trong /kaggle/input:\n"
                for p in pths_found:
                    err_msg += f"      - {p}\n"
            err_msg += f"   👉 Vui lòng kiểm tra lại đường dẫn bằng lệnh: !ls -R {os.path.dirname(resume_target) if os.path.dirname(resume_target) else '/kaggle/input'}\n"
            raise FileNotFoundError(err_msg)

        # 4. Tiến hành nạp checkpoint
        print(f"🔄 [Resume] Đang nạp checkpoint từ: {resume_target}")
        try:
            ckpt = torch.load(resume_target, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(resume_target, map_location="cpu")

        if not isinstance(ckpt, dict):
            raise ValueError(f"❌ Checkpoint file không đúng định dạng dictionary (type={type(ckpt)})")

        raw_curr_s = unwrap_model(student_model)
        raw_curr_t = unwrap_model(teacher_model)
        available_keys = list(ckpt.keys())
        print(f"   🔑 [Checkpoint Keys] Các trường dữ liệu tìm thấy: {available_keys}")

        # Khôi phục Student Backbone
        if "student_backbone" in ckpt:
            smart_load_state_dict(raw_curr_s.backbone, ckpt["student_backbone"], strict=False, verbose=True)
        elif "model_state" in ckpt or "state_dict" in ckpt or "model" in ckpt:
            sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt.get("model")))
            bb_sd = {k.replace("backbone.", ""): v for k, v in sd.items() if "backbone." in k}
            if bb_sd:
                smart_load_state_dict(raw_curr_s.backbone, bb_sd, strict=False, verbose=True)
            else:
                smart_load_state_dict(raw_curr_s.backbone, sd, strict=False, verbose=True)

        # Khôi phục Teacher Backbone
        if "teacher_backbone" in ckpt:
            smart_load_state_dict(raw_curr_t.backbone, ckpt["teacher_backbone"], strict=False, verbose=True)
        else:
            # Đồng bộ lại teacher từ student nếu checkpoint không lưu riêng teacher
            with torch.no_grad():
                for ps, pt in zip(raw_curr_s.backbone.parameters(), raw_curr_t.backbone.parameters()):
                    pt.data.copy_(ps.data)

        # Khôi phục Heads
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

        if "pos_stats" in ckpt:
            pos_dict = clean_state_dict(ckpt["pos_stats"])
            if "mu" in pos_dict and pos_dict["mu"].shape[0] != pos_stats.num_cams:
                cams_in_ckpt = pos_dict["mu"].shape[0]
                print(f"   ⚙️ [TAM Scale] Tự động mở rộng dung lượng camera ({pos_stats.num_cams} -> {cams_in_ckpt} cameras)...")
                pos_stats.set_cam_capacity(cams_in_ckpt)
            pos_stats.load_state_dict(pos_dict)
            print(f"   ✅ [PositionStats] Khôi phục thành công thống kê vị trí cho {pos_stats.num_cams} camera.")
        if "dino_center" in ckpt and hasattr(dino_loss_fn, "center"):
            dino_loss_fn.center.copy_(ckpt["dino_center"].to(device))
        if "ibot_center" in ckpt and hasattr(ibot_loss_fn, "center"):
            ibot_loss_fn.center.copy_(ckpt["ibot_center"].to(device))

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

        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
                print("   ✅ [Scheduler] Khôi phục lịch trình Learning Rate (CosineAnnealingLR).")
            except Exception as e_sched:
                print(f"   ⚠️ [Scheduler Notice] {e_sched}")

        if "scaler" in ckpt and ckpt["scaler"] is not None and hasattr(scaler, "load_state_dict"):
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass

        if "history" in ckpt and isinstance(ckpt["history"], dict):
            history = {k: list(v) for k, v in ckpt["history"].items()}
            print(f"   📊 [History] Khôi phục toàn bộ lịch sử loss ({len(history.get('total', []))} epochs trước).")

        # Xác định epoch tiếp theo
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

        if "best_loss" in ckpt and ckpt["best_loss"] is not None:
            best_loss = float(ckpt["best_loss"])
        print("✅ [Resume] Đã khôi phục thành công trạng thái mô hình!")

        if start_epoch >= args.epochs:
            print(f"\n⚠️ [Cảnh Báo Resume] Checkpoint đã hoàn thành {start_epoch}/{args.epochs} epochs.")
            print(f"💡 Nếu muốn tiếp tục huấn luyện, vui lòng đặt --epochs lớn hơn {start_epoch} (ví dụ: --epochs {start_epoch + 5}).\n")

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

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    print("🏁 [Huấn Luyện] Bắt đầu vòng lặp huấn luyện Hướng G...")
    step = start_epoch * len(loader) if start_epoch > 0 else 0
    start_time = time.time()
    last_visual_data = None

    best_checkpoint_path = os.path.join(args.output_dir, "best_checkpoint.pth")
    last_checkpoint_path = os.path.join(args.output_dir, "last_checkpoint.pth")
    latest_checkpoint_path = os.path.join(args.output_dir, "dinov3_directionG_latest.pth")

    def save_g_checkpoint(save_path: str, epoch_num: int, is_best: bool = False):
        raw_curr_s = unwrap_model(student_model)
        raw_curr_t = unwrap_model(teacher_model)
        checkpoint_dict = {
            "epoch": epoch_num,
            "student_backbone": clean_state_dict(raw_curr_s.backbone.state_dict()),
            "teacher_backbone": clean_state_dict(raw_curr_t.backbone.state_dict()),
            "student_dino_head": clean_state_dict(raw_curr_s.dino_head.state_dict()),
            "teacher_dino_head": clean_state_dict(raw_curr_t.dino_head.state_dict()),
            "student_ibot_head": clean_state_dict(raw_curr_s.ibot_head.state_dict()),
            "teacher_ibot_head": clean_state_dict(raw_curr_t.ibot_head.state_dict()),
            "pos_stats": clean_state_dict(pos_stats.state_dict()),
            "dino_center": dino_loss_fn.center.detach().cpu(),
            "ibot_center": ibot_loss_fn.center.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "history": history,
            "best_loss": best_loss,
            "num_gpus": num_gpus,
            "effective_batch_size": effective_batch_size,
            "args": vars(args),
        }
        torch.save(checkpoint_dict, save_path)
        if is_best:
            print(f"🏆 [Checkpoint] Đã lưu mô hình tốt nhất mới (Best Loss: {best_loss:.4f}) tại: {save_path}")

    try:
        for epoch in range(start_epoch, args.epochs):
            student_model.train()
            epoch_losses = []
            epoch_dino_losses = []
            epoch_ibot_losses = []
            epoch_koleo_losses = []

            for batch in loader:
                x1 = batch["x1"].to(device, non_blocking=(device.type == "cuda"))
                x2 = batch["x2"].to(device, non_blocking=(device.type == "cuda"))
                cids = batch["cid"].to(device, non_blocking=(device.type == "cuda"))
                B = x1.shape[0]

                # Bước A: Trích xuất đặc trưng đóng băng cho TAM
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                        u1 = frozen_extractor(x1)  # (B, 448, 64)
                        u2 = frozen_extractor(x2)  # (B, 448, 64)
                    u1_f = u1.float()
                    u2_f = u2.float()
                    pos_stats.update(cids, u1_f)
                    a1, valid1 = pos_stats.atypicality(cids, u1_f)
                    a2, valid2 = pos_stats.atypicality(cids, u2_f)

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

                # Bước D: Forward Student & Teacher trong AMP autocast (Song song đa GPU)
                target_H = 16 * patch_size
                target_W = 28 * patch_size
                if x1.shape[-2] != target_H or x1.shape[-1] != target_W:
                    x1_vit = F.interpolate(x1, size=(target_H, target_W), mode="bilinear", align_corners=False)
                    x_student_vit = F.interpolate(x_student, size=(target_H, target_W), mode="bilinear", align_corners=False)
                else:
                    x1_vit = x1
                    x_student_vit = x_student

                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    # Teacher nhận ảnh gốc x1_vit không che
                    with torch.no_grad():
                        t_cls, t_patches, t_dino_logits, t_ibot_logits = teacher_model(x1_vit)

                    # Student nhận ảnh x_student_vit (SRS) cùng mặt nạ che phân tầng masks (AGM iBOT)
                    s_cls, s_patches, s_dino_logits, s_ibot_logits = student_model(x_student_vit, masks=mask_batch)

                    # Bước E: Tính tổn thất Loss (KoLeo weight cân bằng 0.02 chống áp đảo)
                    loss_dino = dino_loss_fn(s_dino_logits, t_dino_logits, epoch=epoch)
                    loss_ibot = ibot_loss_fn(s_ibot_logits, t_ibot_logits, mask_batch, pi=pi1.reshape(B, 448))
                    loss_koleo = koleo_loss_fn(s_cls)
                    total_loss = loss_dino + 1.0 * loss_ibot + 0.02 * loss_koleo

                # Bước F: Tối ưu Gradient qua GradScaler
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()

                # Bước G: Cập nhật EMA Teacher (momentum 0.996)
                momentum = 0.996
                with torch.no_grad():
                    raw_curr_s = unwrap_model(student_model)
                    raw_curr_t = unwrap_model(teacher_model)
                    for param_s, param_t in zip(raw_curr_s.parameters(), raw_curr_t.parameters()):
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
                    "patch_features": s_patches.detach().cpu().float(),
                }

                step += 1
                if step % 2 == 0 or step == 1:
                    print(f"   [Epoch {epoch+1}/{args.epochs} | Step {step}] Loss: {total_loss.item():.4f} "
                          f"(DINO: {loss_dino.item():.4f}, iBOT: {loss_ibot.item():.4f}, KoLeo: {loss_koleo.item():.4f})")

            # Cập nhật lịch trình Learning Rate Cosine Annealing
            scheduler.step()

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

            # Cập nhật hình ảnh trực quan hóa và đồ thị sau mỗi epoch
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
    vis_path = os.path.join(args.output_dir, "directionG_progress.png")

    print("\n" + "=" * 80)
    print(" 🎉 [HOÀN TẤT HUẤN LUYỆN TỰ GIÁM SÁT HƯỚNG G]")
    print("=" * 80)
    print(f" 💾 Checkpoint Weights (Latest): {latest_checkpoint_path}")
    print(f" 💾 Checkpoint Weights (Best)  : {best_checkpoint_path}")
    print(f" 💾 Checkpoint Weights (Last)  : {last_checkpoint_path}")
    print(f" 🖼️ Ảnh Trực Quan Hóa (Visual) : {vis_path}")
    print(f" 📊 Đồ Thị Hàm Mất Mát (Loss)  : {loss_curve_path}")
    print(f" 📈 Nhật Ký Huấn Luyện (JSON)  : {metrics_path}")
    print(f" ⏱️ Tổng Thời Gian Thực Thi    : {elapsed_time:.2f}s")
    print("=" * 80 + "\n")

    return latest_checkpoint_path


if __name__ == "__main__":
    train_direction_g()
