"""
=============================================================================
 Hướng 1: BG-Guided DINO — Training Pipeline (Production-Ready)
 Huấn luyện Tiền huấn luyện Tự giám sát (SSL Continual Pre-training)
 với Foreground-Aware Masking từ ảnh Background giao thông
=============================================================================
"""

import argparse
import math
import os
import sys
import time

# Chống xung đột OpenMP trên Windows Anaconda và đảm bảo hiển thị ký tự tiếng Việt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*dinov2.*")

from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.backbone_loader import get_dino_backbone
from common.gpu_utils import setup_multi_gpu, unwrap_model, save_checkpoint, clean_state_dict
from dataset import BGGuidedDINODataset
from models import BGGuidedDINOModel
from losses import BGGuidedDINOLoss


def get_cosine_schedule(base_val: float, final_val: float, total_iters: int, warmup_iters: int = 0) -> np.ndarray:
    """Tạo lịch trình biến thiên Cosine với Warmup tuyến tính."""
    warmup_schedule = np.linspace(0, base_val, warmup_iters) if warmup_iters > 0 else np.array([])
    iters = np.arange(max(1, total_iters - warmup_iters))
    schedule = final_val + 0.5 * (base_val - final_val) * (1 + np.cos(np.pi * iters / len(iters)))
    return np.concatenate((warmup_schedule, schedule))


def train_bg_guided_dino(args):
    # Khởi tạo hạt giống ngẫu nhiên đảm bảo tính tái lập (Reproducibility)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(" 🚀 STARTING BG-GUIDED DINO SSL TRAINING")
    print("=" * 78)
    print(f" Backbone Architecture : {args.backbone}")
    print(f" Background Directory  : {args.bg_dir}")
    print(f" Origin Directory      : {args.origin_dir}")
    print(f" Epochs                : {args.epochs}")
    print(f" Base Batch Size/GPU   : {args.batch_size}")
    print(f" Base Learning Rate    : {args.lr}")
    print(f" Foreground Alpha      : {args.alpha_fg} (Mask bias ratio)")
    print(f" Output Checkpoint Dir : {args.save_dir}")
    print("=" * 78)

    # 1. Khởi tạo Backbone và Tự động cấu hình Toàn bộ GPU (Multi-GPU Engine)
    print(f"🧠 [Model] Nạp Backbone '{args.backbone}'...")
    student_backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        pretrained=True,
        weights_path=args.weights,
        device=args.device,
    )

    base_model = BGGuidedDINOModel(
        student_backbone=student_backbone,
        embed_dim=embed_dim,
        out_dim=args.out_dim,
    )

    # Tự động nhận diện toàn bộ số lượng GPU và bọc Multi-GPU song song
    model, device, num_gpus, effective_batch_size, effective_lr = setup_multi_gpu(
        model=base_model,
        batch_size_per_gpu=args.batch_size,
        base_lr=args.lr,
        device_arg=args.device,
    )

    # Tự động chuẩn hóa kích thước crop đảm bảo luôn là bội số nguyên của patch_size
    if args.size_global % patch_size != 0:
        old_val = args.size_global
        args.size_global = max(patch_size, round(args.size_global / patch_size) * patch_size)
        print(f"📐 [Auto-Align] Điều chỉnh size_global: {old_val} -> {args.size_global} (bội số của patch_size={patch_size})")

    if args.size_local % patch_size != 0:
        old_val = args.size_local
        args.size_local = max(patch_size, round(args.size_local / patch_size) * patch_size)
        print(f"📐 [Auto-Align] Điều chỉnh size_local: {old_val} -> {args.size_local} (bội số của patch_size={patch_size})")

    # 2. Khởi tạo Dataset
    print("📦 [Data] Khởi tạo BGGuidedDINODataset...")
    try:
        dataset = BGGuidedDINODataset(
            bg_dir=args.bg_dir,
            origin_dir=args.origin_dir,
            match_strategy=args.match_strategy,
            patch_size=patch_size,
            size_global=args.size_global,
            size_local=args.size_local,
            local_crops_number=args.local_crops,
            mask_ratio=args.mask_ratio,
            alpha_fg=args.alpha_fg,
            max_samples=args.max_samples,
        )
        print(f"✅ [Data] Đã nạp thành công {len(dataset)} cặp ảnh hợp lệ.")
    except Exception as e:
        print(f"❌ [Data Error] {e}")
        return

    def collate_dino(batch):
        n_crops = len(batch[0]["crops"])
        crops_stacked = [torch.stack([b["crops"][i] for b in batch]) for i in range(n_crops)]
        masks_stacked = torch.stack([b["fg_mask"] for b in batch])
        return crops_stacked, masks_stacked

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_dino,
    )

    # 3. Khởi tạo Hàm mất mát DINO Loss
    total_crops = 2 + args.local_crops
    dino_loss_fn = BGGuidedDINOLoss(
        out_dim=args.out_dim,
        ncrops=total_crops,
        nepochs=args.epochs,
    ).to(device)

    # 4. Cấu hình Optimizer và Lịch trình Lr / Momentum
    raw_model = unwrap_model(model)
    optimizer = torch.optim.AdamW(
        raw_model.student_backbone.parameters(),
        lr=effective_lr,
        weight_decay=0.04,
    )

    # Mixed precision scaler (Sử dụng torch.amp chuẩn PyTorch 2.x+ thay thế API cũ đã deprecated)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = (device.type == "cuda" and args.use_amp)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 5. Khôi phục trạng thái từ Checkpoint nếu có cờ --resume
    start_epoch = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Không tìm thấy file checkpoint resume: {args.resume}")
        print(f"\n🔄 [Resume] Đang khôi phục toàn bộ trạng thái huấn luyện từ checkpoint: {args.resume}")
        ckpt_data = load_checkpoint(
            load_path=args.resume,
            model=raw_model.student_backbone,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            strict=False,
            verbose=True,
        )

        if "teacher_state" in ckpt_data:
            smart_load_state_dict(raw_model.teacher_backbone, ckpt_data["teacher_state"], strict=False, verbose=False)
            print("   ✅ [Teacher] Đã khôi phục thành công trạng thái Teacher EMA.")
        else:
            raw_model.teacher_backbone.load_state_dict(raw_model.student_backbone.state_dict())

        if "head_state" in ckpt_data:
            smart_load_state_dict(raw_model.student_head, ckpt_data["head_state"], strict=False, verbose=False)
            raw_model.teacher_head.load_state_dict(raw_model.student_head.state_dict())
            print("   ✅ [Projection Head] Đã khôi phục thành công Student & Teacher DINO Head.")

        if "epoch" in ckpt_data and ckpt_data["epoch"] is not None:
            start_epoch = int(ckpt_data["epoch"])
            print(f"   ⏱️ [Epoch] Khôi phục tại epoch {start_epoch}. Sẽ tiếp tục chạy từ epoch {start_epoch + 1}.")
            if args.epochs <= start_epoch:
                target_epochs = start_epoch + args.epochs
                print(f"   💡 [Gia hạn Epochs] Số epochs cài đặt ({args.epochs}) <= epoch checkpoint ({start_epoch}).")
                print(f"      -> Tự động huấn luyện thêm {args.epochs} epochs (Tổng mới: {target_epochs} epochs).")
                args.epochs = target_epochs

    total_iters = len(loader) * args.epochs
    lr_schedule = get_cosine_schedule(effective_lr, 1e-6, total_iters, warmup_iters=len(loader) * 2)
    momentum_schedule = get_cosine_schedule(0.996, 1.0, total_iters)
    global_step = start_epoch * len(loader)

    # 6. Training Loop
    print(f"\n🏁 [Train] Bắt đầu huấn luyện từ Epoch [{start_epoch+1}/{args.epochs}] trên {max(1, num_gpus)} thiết bị...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for crops, masks in pbar:
            cur_lr = lr_schedule[min(global_step, total_iters - 1)]
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            crops = [c.to(device, non_blocking=True) for c in crops]
            global_crops = crops[:2]

            # Quản lý Autocast theo chuẩn PyTorch 2.x+
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type=device_type, enabled=amp_enabled)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled)

            with autocast_ctx:
                masks_dev = masks.to(device, non_blocking=True)
                # Teacher forward 2 global views song song trên toàn bộ GPU
                with torch.no_grad():
                    teacher_res = model(global_crops, mode="teacher")
                    if isinstance(teacher_res, tuple):
                        teacher_cls, teacher_patch = teacher_res
                    else:
                        teacher_cls, teacher_patch = teacher_res, None

                # Student forward toàn bộ views và trích xuất thêm patch logits cho view bị che
                student_res = model(crops, mask=masks_dev, mode="student")
                if isinstance(student_res, tuple):
                    student_cls, student_patch = student_res
                else:
                    student_cls, student_patch = student_res, None

                # Tính DINO CLS loss + iBOT Patch loss
                loss = dino_loss_fn(
                    student_cls=student_cls,
                    teacher_cls=teacher_cls,
                    student_patch=student_patch,
                    teacher_patch=teacher_patch,
                    mask=masks_dev,
                    epoch=epoch,
                )

            # Cập nhật Gradient
            optimizer.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.student_backbone.parameters(), max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_model.student_backbone.parameters(), max_norm=3.0)
                optimizer.step()

            # Cập nhật Teacher EMA đồng bộ
            cur_momentum = momentum_schedule[min(global_step, total_iters - 1)]
            raw_model.update_teacher(cur_momentum)

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{cur_lr:.2e}",
                "entropy": f"{dino_loss_fn.last_entropy:.2f}",
                "gpus": num_gpus if num_gpus > 0 else "CPU",
            })

        avg_loss = epoch_loss / len(loader)
        print(f"📊 Epoch [{epoch+1}/{args.epochs}] Hoàn tất — Loss TB: {avg_loss:.4f} | Teacher Entropy: {dino_loss_fn.last_entropy:.3f}")

        # Lưu Checkpoint (chuẩn hóa không dính tiền tố 'module.')
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"bg_dino_{args.backbone}_ep{epoch+1}.pth")
            save_checkpoint(
                save_path=ckpt_path,
                model=raw_model.student_backbone,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                metrics={"loss": avg_loss, "entropy": dino_loss_fn.last_entropy},
                extra_dict={
                    "teacher_state": clean_state_dict(raw_model.teacher_backbone.state_dict()),
                    "head_state": clean_state_dict(raw_model.student_head.state_dict()),
                    "num_gpus": num_gpus,
                    "args": vars(args),
                },
                verbose=True,
            )

    # 5. Tự động xuất ảnh trực quan hóa PCA Feature Map kiểm chứng mô hình đã học
    try:
        from direction2_zero_shot_segmentation.pca_extractor import DINOPCAExtractor
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        print("\n🎨 [Visualization] Đang trích xuất PCA Feature Map kiểm chứng chất lượng biểu diễn...")
        vis_sample = dataset.pairs[0]
        sample_img = Image.open(vis_sample["origin_path"]).convert("RGB")
        sample_bg = Image.open(vis_sample["bg_path"]).convert("RGB")

        pca_ext = DINOPCAExtractor(
            backbone=raw_model.student_backbone,
            patch_size=patch_size,
            img_size=args.size_global,
            device=device,
        )
        pca_rgb, pc1_mask = pca_ext.compute_pca_maps(sample_img)
        delta_norm, _ = dataset.subtractor.compute_delta(sample_img, sample_bg)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)
        axes[0].imshow(sample_bg)
        axes[0].set_title(f"1. Background Nền ({vis_sample['route_id']})", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(sample_img)
        axes[1].set_title("2. Ảnh gốc Giao thông", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(delta_norm, cmap="inferno")
        axes[2].set_title("3. Bản đồ sai khác Δ (Trừ nền)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(pca_rgb)
        axes[3].set_title(f"4. DINO PCA Feature Map ({args.backbone})", fontsize=11, fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        vis_path = os.path.join(args.save_dir, "bg_guided_dino_comparison.png")
        plt.savefig(vis_path, bbox_inches="tight")
        plt.close()
        print(f"✅ [Visualization] Đã lưu ảnh đối chiếu 4 ô tại: {vis_path}")

        # Xuất biểu đồ Emergent PCA Feature Maps (4 ảnh x 2 cột) chuẩn báo cáo khoa học
        from visualize_pca import generate_emergent_pca_maps
        sample_paths = [p["origin_path"] for p in dataset.pairs[:4]]
        title_pfx = "DINOv3" if "dinov3" in args.backbone.lower() else "DINOv2"
        grid_save_path = os.path.join(args.save_dir, "emergent_pca_feature_maps.png")
        generate_emergent_pca_maps(
            backbone=raw_model.student_backbone,
            image_paths=sample_paths,
            save_path=grid_save_path,
            device=device,
            img_size=args.size_global,
            title_prefix=title_pfx,
        )
    except Exception as e_vis:
        print(f"💡 [Visualization Notice] {e_vis}")

    print("\n🎉 [Train Complete] Quá trình huấn luyện SSL hoàn tất thành công!")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện BG-Guided DINO SSL")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục ảnh background")
    parser.add_argument("--origin_dir", type=str, default="output", help="Thư mục ảnh origin")
    parser.add_argument("--save_dir", type=str, default="checkpoints/direction1_bg_dino", help="Thư mục lưu checkpoint")
    parser.add_argument("--match_strategy", type=str, default="route_hourly", choices=["route_hourly", "same_name", "camera_id"])
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone (dinov3_vits16 / dinov2_vits14)")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn custom checkpoint ban đầu (chỉ nạp backbone)")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint (.pth) để khôi phục toàn bộ trạng thái (epoch, optimizer, teacher, head, scaler) và tiếp tục huấn luyện")
    parser.add_argument("--epochs", type=int, default=10, help="Số epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size mỗi step")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate cực đại")
    parser.add_argument("--alpha_fg", type=float, default=0.75, help="Hệ số tập trung foreground (0.0: ngẫu nhiên, 1.0: thuần xe)")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Tỷ lệ diện tích patch bị che")
    parser.add_argument("--size_global", type=int, default=224, help="Kích thước crop toàn cảnh")
    parser.add_argument("--size_local", type=int, default=96, help="Kích thước crop cục bộ")
    parser.add_argument("--local_crops", type=int, default=4, help="Số lượng local crops")
    parser.add_argument("--out_dim", type=int, default=4096, help="Kích thước vector prototype")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Sử dụng Mixed Precision (FP16)")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị ('cuda' hoặc 'cpu')")
    parser.add_argument("--num_workers", type=int, default=0, help="Số luồng nạp dữ liệu")
    parser.add_argument("--save_every", type=int, default=5, help="Lưu checkpoint sau mỗi N epochs")
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn số mẫu thử nghiệm nhanh")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_bg_guided_dino(args)
