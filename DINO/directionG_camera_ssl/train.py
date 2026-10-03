"""
=============================================================================
 Hướng G: Training Pipeline (directionG_camera_ssl/train.py)
 Pipeline huấn luyện tự giám sát DINOv3 kết hợp TAM, AGM và SRS
 Hỗ trợ DDP / Single-GPU / CPU, EMA Teacher và lưu checkpoint toàn diện
=============================================================================
"""

import argparse
import copy
import os
import sys
import time

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

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if len(shape) == 3:  # (B, N, D)
            B, N, D = shape
            x = x.view(B * N, D)
            feat = self.mlp(x)
            feat = F.normalize(feat, dim=-1, p=2)
            out = self.last_layer(feat)
            return out.view(B, N, -1)
        else:  # (B, D)
            feat = self.mlp(x)
            feat = F.normalize(feat, dim=-1, p=2)
            return self.last_layer(feat)


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
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace Token")
    return parser.parse_args()


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
        device=device,
        hf_token=args.hf_token,
    )

    print("📦 [Backbone] Khởi tạo EMA Teacher DINOv3...")
    teacher_backbone = copy.deepcopy(student_backbone)
    for p in teacher_backbone.parameters():
        p.requires_grad = False

    # 2. Đầu chiếu DINO và iBOT Heads
    student_dino_head = DINOHead(in_dim=embed_dim, out_dim=args.out_dim).to(device)
    teacher_dino_head = copy.deepcopy(student_dino_head)
    for p in teacher_dino_head.parameters():
        p.requires_grad = False

    student_ibot_head = DINOHead(in_dim=embed_dim, out_dim=args.out_dim).to(device)
    teacher_ibot_head = copy.deepcopy(student_ibot_head)
    for p in teacher_ibot_head.parameters():
        p.requires_grad = False

    # 3. Module FrozenExtractor & PositionStats cho TAM
    print("🔍 [TAM] Khởi tạo FrozenExtractor & PositionStats...")
    frozen_extractor = FrozenExtractor(model_name=args.model_name, pca_dim=64, device=device)
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

    # 5. Dataloader giả lập mẫu
    dataset = SyntheticTrafficDataset(num_samples=16, num_cams=4)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("🏁 [Huấn Luyện] Bắt đầu vòng lặp huấn luyện Hướng G...")
    step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        student_backbone.train()
        student_dino_head.train()
        student_ibot_head.train()

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
            # Teacher nhận ảnh gốc x1 không che
            with torch.no_grad():
                t_cls, t_patches_spatial = extract_tokens(teacher_backbone, x1, patch_size=patch_size)
                t_patches = t_patches_spatial.view(B, 448, embed_dim)
                t_dino_logits = teacher_dino_head(t_cls)
                t_ibot_logits = teacher_ibot_head(t_patches)

            # Student nhận ảnh x_student (đã qua SRS)
            s_cls, s_patches_spatial = extract_tokens(student_backbone, x_student, patch_size=patch_size)
            s_patches = s_patches_spatial.view(B, 448, embed_dim)
            s_dino_logits = student_dino_head(s_cls)
            s_ibot_logits = student_ibot_head(s_patches)

            # Bước E: Tính tổn thất Loss
            loss_dino = dino_loss_fn(s_dino_logits, t_dino_logits, epoch=epoch)
            loss_ibot = ibot_loss_fn(s_ibot_logits, t_ibot_logits, mask_batch, pi=pi1.view(B, 448))
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

            step += 1
            if step % 2 == 0 or step == 1:
                print(f"   [Epoch {epoch+1}/{args.epochs} | Step {step}] Loss: {total_loss.item():.4f} "
                      f"(DINO: {loss_dino.item():.4f}, iBOT: {loss_ibot.item():.4f}, KoLeo: {loss_koleo.item():.4f})")

    # 6. Lưu checkpoint toàn diện
    save_path = os.path.join(args.output_dir, "dinov3_directionG_latest.pth")
    torch.save({
        "student_backbone": student_backbone.state_dict(),
        "teacher_backbone": teacher_backbone.state_dict(),
        "student_dino_head": student_dino_head.state_dict(),
        "student_ibot_head": student_ibot_head.state_dict(),
        "pos_stats": pos_stats.state_dict(),
        "args": vars(args),
    }, save_path)
    print(f"💾 [Direction G] Đã lưu thành công checkpoint tại: {save_path}")
    print(f"⏱️ Tổng thời gian chạy: {time.time() - start_time:.2f}s")
    return save_path


if __name__ == "__main__":
    train_direction_g()
