"""
=============================================================================
 ST-BMB: Streaming Background Memory Bank — Training Pipeline
 Huấn luyện Mô hình Phân rã Cảnh & Phân đoạn Phương tiện Luồng Thời Gian Thực
=============================================================================
Cải tiến đột phá chuẩn Production:
  1. Tích hợp Memory Dropout (K in [0, K_max]) chống quá khớp vào bộ nhớ.
  2. Lịch trình ấm dần (Warm-up Schedule) cho Cross-frame Loss và Binarization Loss.
  3. Quản lý luồng thời gian thực: truyền timestamps vào mô hình và Memory Bank.
  4. Hỗ trợ Train / Val split theo Camera ID chống rò rỉ dữ liệu.
  5. Xuất ảnh trực quan 6 hàng và biểu đồ học tập chi tiết.
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dino_root not in sys.path:
    sys.path.insert(0, dino_root)

from common.backbone_loader import get_dino_backbone
from common.gpu_utils import setup_multi_gpu, unwrap_model, save_checkpoint, load_checkpoint
from direction_streaming_bmb.models import StreamingDecompositionNet
from direction_streaming_bmb.losses import StreamingDecompositionLoss
from direction_streaming_bmb.dataset import SlidingWindowTrafficDataset


def save_streaming_visual_sample(
    frames: torch.Tensor,
    step_outputs: list,
    save_path: str,
    epoch: int,
):
    """
    Xuất ma trận trực quan hóa chuỗi W khung hình:
      Hàng 1: Ảnh gốc đầu vào I_t
      Hàng 2: Mặt nạ phân đoạn xe alpha_t
      Hàng 3: Tiền cảnh xe dự đoán F_t
      Hàng 4: Nền đường sạch dự đoán B_t (BMB Clean Road)
      Hàng 5: Bản đồ độ bất định sigma_t
      Hàng 6: Ảnh tái tạo vật lý I_recon_t
    """
    W_len = frames.shape[1]
    fig, axes = plt.subplots(6, W_len, figsize=(3.5 * W_len, 18), dpi=150)

    def to_rgb(t):
        arr = t.squeeze().detach().cpu().permute(1, 2, 0).numpy()
        return np.clip(arr, 0.0, 1.0)

    def to_gray(t):
        arr = t.squeeze().detach().cpu().numpy()
        return np.clip(arr, 0.0, 1.0)

    if W_len == 1:
        axes = np.expand_dims(axes, axis=1)

    row_titles = [
        "Input Frame $I_t$",
        "Vehicle Alpha $\\hat{\\alpha}_t$",
        "Foreground $\\hat{F}_t$",
        "Clean Road $\\hat{B}_t$ (BMB)",
        "Uncertainty $\\hat{\\sigma}_t$",
        "Physical Recon $\\hat{I}_t$",
    ]

    for t in range(W_len):
        cur_in = frames[0, t]
        out_t = step_outputs[t]

        alpha_t = out_t["alpha_mask"][0]
        fg_t = out_t["pred_fg"][0]
        bg_t = out_t["pred_bg"][0]
        sigma_t = out_t["sigma"][0]
        recon_t = out_t["recon_origin"][0]

        sigma_np = sigma_t.squeeze().detach().cpu().numpy()
        sigma_norm = np.clip((sigma_np - 0.01) / (0.50 - 0.01), 0.0, 1.0)

        axes[0, t].imshow(to_rgb(cur_in))
        axes[0, t].set_title(f"Time Step $t={t+1}$", fontsize=11, fontweight="bold")
        axes[0, t].axis("off")

        axes[1, t].imshow(to_gray(alpha_t), cmap="inferno")
        axes[1, t].axis("off")

        axes[2, t].imshow(to_rgb(fg_t))
        axes[2, t].axis("off")

        axes[3, t].imshow(to_rgb(bg_t))
        axes[3, t].axis("off")

        axes[4, t].imshow(sigma_norm, cmap="magma")
        axes[4, t].axis("off")

        axes[5, t].imshow(to_rgb(recon_t))
        axes[5, t].axis("off")

    for r in range(6):
        axes[r, 0].text(
            -0.1, 0.5, row_titles[r],
            transform=axes[r, 0].transAxes,
            rotation=90, verticalalignment="center", horizontalalignment="right",
            fontsize=12, fontweight="bold",
        )

    plt.suptitle(
        f"ST-BMB Streaming Decomposition — Epoch {epoch} (Point-wise Temporal Memory Sequence)",
        fontsize=14, fontweight="bold", y=0.99,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def train_streaming_bmb(args):
    print("=" * 75)
    print(" 🚀 HUẤN LUYỆN ST-BMB: SPATIO-TEMPORAL BACKGROUND MEMORY BANK")
    print("=" * 75)
    print(f"📁 Thư mục ảnh đầu vào     : {args.data_dir}")
    print(f"📁 Thư mục lưu kết quả     : {args.save_dir}")
    print(f"⚙️  Backbone Vision         : {args.backbone}")
    print(f"⏱️  Kích thước cửa sổ (W)   : {args.window_size} frames (cách nhau 5 phút)")
    print(f"🖼️  Kích thước ảnh          : {args.img_size}x{args.img_size}")
    print(f"🔢 Số Epochs               : {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"⚖️  Trọng số Temporal Loss  : {args.lambda_temp} | Cross-Frame: {args.lambda_cross}")
    print(f"🎲 Memory Dropout          : {args.memory_dropout} | Warmup Epochs: {args.warmup_epochs}")
    print("=" * 75)

    os.makedirs(args.save_dir, exist_ok=True)
    vis_dir = os.path.join(args.save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Khởi tạo Backbone
    print(f"\n🧠 [Backbone] Đang tải mô hình {args.backbone}...")
    backbone, feat_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        weights_path=args.weights,
        freeze=args.freeze_backbone,
    )

    # 2. Khởi tạo Mô hình StreamingDecompositionNet
    model = StreamingDecompositionNet(
        backbone=backbone,
        backbone_dim=feat_dim,
        embed_dim=args.embed_dim,
        patch_size=patch_size,
        max_recent_frames=args.max_recent,
        max_anchor_frames=args.max_anchor,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        num_heads=args.num_heads,
        ema_eta=args.ema_eta,
    )

    # 3. Thiết lập Thiết bị & Multi-GPU (Thread-Safe Per-Device Memory Bank)
    if args.data_parallel:
        print("⚡ [Multi-GPU DataParallel] Kích hoạt nn.DataParallel với Per-Device Memory Bank Isolation.")
        print("   Hệ thống tự động phân tách độc lập buffer bộ nhớ cho từng GPU để triệt tiêu 100% race condition.")
        model, device, num_gpus, effective_batch_size, effective_lr = setup_multi_gpu(
            model=model,
            batch_size_per_gpu=args.batch_size,
            base_lr=args.lr,
            device_arg=args.device,
        )
    else:
        # Chế độ Single GPU Chuyên Dụng
        if args.device.lower() == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
            num_gpus = 0
            print("🖥️ [Hardware] Chạy trên CPU.")
        else:
            dev_idx = 0
            if ":" in args.device:
                try:
                    dev_idx = int(args.device.split(":")[-1])
                except ValueError:
                    dev_idx = 0
            device = torch.device(f"cuda:{dev_idx}")
            num_gpus = 1
            gpu_name = torch.cuda.get_device_name(dev_idx)
            mem_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
            print(f"⚡ [Hardware] ST-BMB chạy trên GPU chuyên dụng: {gpu_name} ({device}) | VRAM: {mem_gb:.2f} GB")
            if torch.cuda.device_count() > 1:
                print(f"ℹ️ [Multi-GPU Info] Phát hiện {torch.cuda.device_count()} GPUs. ST-BMB tự động vận hành trên GPU đơn ({device}) (hoặc dùng --data_parallel để kích hoạt song song {torch.cuda.device_count()} GPUs).")

        model = model.to(device)
        effective_batch_size = args.batch_size
        effective_lr = args.lr

    raw_model = unwrap_model(model)
    raw_model.set_memory_dropout(args.memory_dropout)

    # 4. Khởi tạo Dataset huấn luyện & Validation với effective_batch_size chuẩn
    train_dataset = SlidingWindowTrafficDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        img_size=args.img_size,
        split=args.split,
        val_ratio=args.val_ratio,
        max_gap=args.max_gap,
        is_train=(args.split == "train"),
        max_samples=args.max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=(len(train_dataset) > effective_batch_size),
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if args.val_ratio > 0.0 and args.split == "train":
        try:
            val_dataset = SlidingWindowTrafficDataset(
                data_dir=args.data_dir,
                window_size=args.window_size,
                stride=args.stride,
                img_size=args.img_size,
                split="val",
                val_ratio=args.val_ratio,
                max_gap=args.max_gap,
                is_train=False,
                max_samples=args.max_samples,
                apply_jitter=False,
                apply_crop=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=effective_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            print(f"📊 [Validation] Đã khởi tạo tập kiểm định: {len(val_dataset)} cửa sổ.")
        except Exception as e_v:
            print(f"ℹ️ [Validation Note] Bỏ qua val split: {e_v}")

    # 5. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=effective_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Hàm mất mát đa mục tiêu chuẩn Production
    loss_fn = StreamingDecompositionLoss(
        lambda_temp=args.lambda_temp,
        lambda_cross=args.lambda_cross,
        lambda_fam=args.lambda_fam,
        lambda_chroma=args.lambda_chroma,
        lambda_sparse=args.lambda_sparse,
        lambda_tv=args.lambda_tv,
        lambda_excl=args.lambda_excl,
        lambda_bin=args.lambda_bin,
    ).to(device)

    # 6. Vòng lặp huấn luyện
    history = {
        "epochs": [],
        "loss_total": [],
        "loss_rec": [],
        "loss_temp_bg": [],
        "loss_cross": [],
        "loss_fam": [],
        "loss_sparse": [],
        "mean_sigma": [],
        "mean_soft_weight": [],
        "warmup_factor": [],
    }
    best_loss = float("inf")

    print("\n🚀 Bắt đầu huấn luyện...")
    for epoch in range(args.epochs):
        model.train()
        # Cập nhật Warm-up Factor cho Cross-Frame Loss và Binarization Loss
        warmup_factor = min(1.0, (epoch + 1) / max(1, args.warmup_epochs))
        loss_fn.set_warmup_factor(warmup_factor)

        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_temp = 0.0
        epoch_cross = 0.0
        epoch_fam = 0.0
        epoch_sparse = 0.0
        epoch_sigma = 0.0
        epoch_soft_w = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] (Warmup={warmup_factor:.2f})", leave=True)
        last_cam_id: Optional[str] = None
        for batch_idx, batch in enumerate(pbar):
            frames = batch["frames"].to(device)  # (B, W, 3, H, W)
            timestamps = batch["timestamps"].to(device) if "timestamps" in batch else None
            jitter_g = batch["jitter_gain"].to(device) if "jitter_gain" in batch and torch.is_tensor(batch["jitter_gain"]) else (
                torch.tensor(batch["jitter_gain"], device=device) if "jitter_gain" in batch else None
            )
            jitter_b = batch["jitter_bias"].to(device) if "jitter_bias" in batch and torch.is_tensor(batch["jitter_bias"]) else (
                torch.tensor(batch["jitter_bias"], device=device) if "jitter_bias" in batch else None
            )

            # Cơ chế Stateful Memory: Nếu bật --stateful_memory và batch hiện tại cùng camera với batch trước,
            # giữ nguyên bộ nhớ để mô hình học cách tận dụng bộ nhớ sâu (T >= 5) như khi suy luận thực tế!
            cur_cam = batch.get("cam_id", None)
            if isinstance(cur_cam, (list, tuple)) and len(cur_cam) > 0:
                cur_cam_str = str(cur_cam[0])
            elif isinstance(cur_cam, str):
                cur_cam_str = cur_cam
            else:
                cur_cam_str = None

            if args.stateful_memory and cur_cam_str is not None and cur_cam_str == last_cam_id:
                reset_memory = False
            else:
                reset_memory = True
            last_cam_id = cur_cam_str

            # Xử lý chuỗi W khung hình với BMB Memory Bank
            step_outputs = model(frames, timestamps=timestamps, reset_memory=reset_memory)

            total_loss, loss_dict = loss_fn(
                step_outputs,
                frames,
                target_jitter_gain=jitter_g,
                target_jitter_bias=jitter_b,
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # Tích lũy số liệu
            epoch_loss += loss_dict["loss_total"]
            epoch_rec += loss_dict["loss_rec"]
            epoch_temp += loss_dict["loss_temp_bg"]
            epoch_cross += loss_dict["loss_cross"]
            epoch_fam += loss_dict["loss_fam"]
            epoch_sparse += loss_dict["loss_sparse"]
            epoch_sigma += loss_dict["mean_sigma"]
            epoch_soft_w += loss_dict.get("mean_soft_weight", 1.0)

            pbar.set_postfix({
                "loss": f"{loss_dict['loss_total']:.4f}",
                "rec": f"{loss_dict['loss_rec']:.4f}",
                "temp": f"{loss_dict['loss_temp_bg']:.4f}",
                "cross": f"{loss_dict['loss_cross']:.4f}",
                "fam": f"{loss_dict['loss_fam']:.4f}",
                "sig": f"{loss_dict['mean_sigma']:.3f}",
            })

        scheduler.step()
        num_batches = max(1, len(train_loader))
        avg_loss = epoch_loss / num_batches
        avg_rec = epoch_rec / num_batches
        avg_temp = epoch_temp / num_batches
        avg_cross = epoch_cross / num_batches
        avg_fam = epoch_fam / num_batches
        avg_sparse = epoch_sparse / num_batches
        avg_sigma = epoch_sigma / num_batches
        avg_soft_w = epoch_soft_w / num_batches

        history["epochs"].append(epoch + 1)
        history["loss_total"].append(float(avg_loss))
        history["loss_rec"].append(float(avg_rec))
        history["loss_temp_bg"].append(float(avg_temp))
        history["loss_cross"].append(float(avg_cross))
        history["loss_fam"].append(float(avg_fam))
        history["loss_sparse"].append(float(avg_sparse))
        history["mean_sigma"].append(float(avg_sigma))
        history["mean_soft_weight"].append(float(avg_soft_w))
        history["warmup_factor"].append(float(warmup_factor))

        print(
            f"📊 Epoch [{epoch+1:02d}/{args.epochs:02d}] — Total: {avg_loss:.4f} "
            f"(Rec: {avg_rec:.4f}, TempBG: {avg_temp:.4f}, Cross: {avg_cross:.4f}, Fam: {avg_fam:.4f}, Sigma: {avg_sigma:.3f})"
        )

        # Xuất trực quan hóa chuỗi
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            sample_frames = sample_batch["frames"][:1].to(device)
            sample_ts = sample_batch["timestamps"][:1].to(device) if "timestamps" in sample_batch else None
            sample_outputs = raw_model(sample_frames, timestamps=sample_ts, reset_memory=True)

            epoch_vis_path = os.path.join(vis_dir, f"epoch_{epoch+1:03d}.png")
            progress_path = os.path.join(args.save_dir, "streaming_progress.png")
            save_streaming_visual_sample(sample_frames, sample_outputs, epoch_vis_path, epoch=epoch + 1)
            save_streaming_visual_sample(sample_frames, sample_outputs, progress_path, epoch=epoch + 1)

        # Lưu Best Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_best = os.path.join(args.save_dir, "best_streaming_model.pth")
            save_checkpoint(
                save_path=ckpt_best,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics={"loss": avg_loss, "rec": avg_rec, "cross": avg_cross, "temp_bg": avg_temp},
                extra_dict={"num_gpus": num_gpus, "args": vars(args)},
                verbose=False,
            )

        # Lưu Last Checkpoint
        ckpt_last = os.path.join(args.save_dir, "last_checkpoint.pth")
        save_checkpoint(
            save_path=ckpt_last,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            metrics={"loss": avg_loss, "rec": avg_rec, "cross": avg_cross, "temp_bg": avg_temp},
            extra_dict={"num_gpus": num_gpus, "args": vars(args)},
            verbose=False,
        )

    # 7. Lưu Metrics JSON
    metrics_path = os.path.join(args.save_dir, "training_metrics.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "args": vars(args),
                "history": history,
                "best_loss": float(best_loss),
            }, f, indent=2, ensure_ascii=False)
        print(f"📊 [Metrics] Đã lưu lịch sử huấn luyện tại: {metrics_path}")
    except Exception as e_m:
        print(f"⚠️ [Metrics Warning] Không thể lưu JSON: {e_m}")

    # 8. Vẽ biểu đồ các thành phần hàm mất mát
    try:
        plt.figure(figsize=(11, 6), dpi=150)
        plt.plot(history["epochs"], history["loss_total"], "b-o", linewidth=2, label="Total Loss")
        plt.plot(history["epochs"], history["loss_rec"], "r--s", linewidth=1.5, label="Recon Loss")
        plt.plot(history["epochs"], history["loss_temp_bg"], "g-.^", linewidth=1.5, label="Temporal BG Loss")
        plt.plot(history["epochs"], history["loss_cross"], "c-.*", linewidth=1.5, label="Cross-Frame BG Loss")
        plt.plot(history["epochs"], history["loss_fam"], "m:d", linewidth=1.5, label="Familiarity Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title(f"ST-BMB Point-wise Streaming Curves ({args.backbone}, W={args.window_size})", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        curve_path = os.path.join(args.save_dir, "loss_curves.png")
        plt.savefig(curve_path, bbox_inches="tight")
        plt.close()
        print(f"📈 [Charts] Đã lưu biểu đồ hàm mất mát tại: {curve_path}")
    except Exception as e_p:
        print(f"⚠️ [Plot Warning] {e_p}")

    print("\n🎉 [Complete] Huấn luyện ST-BMB hoàn tất thành công!")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện ST-BMB Point-wise Streaming Scene Decomposition")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="direction_data_article/zenodo_bundle/sample_preview/sample_camera_sequences",
        help="Thư mục chứa các khung hình camera giao thông",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/direction_streaming_bmb",
        help="Thư mục lưu mô hình và trực quan hóa",
    )
    parser.add_argument("--backbone", type=str, default="dinov3_vits16")
    parser.add_argument("--weights", type=str, default="checkpoints/dinov3_vits16_model.safetensors")
    parser.add_argument("--window_size", type=int, default=5, help="Kích thước cửa sổ trượt W khung hình (K=5)")
    parser.add_argument("--stride", type=int, default=1, help="Bước trượt của cửa sổ thời gian")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_gap", type=float, default=3600.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_recent", type=int, default=5)
    parser.add_argument("--max_anchor", type=int, default=1)
    parser.add_argument("--ema_eta", type=float, default=0.1, help="Tốc độ học EMA nền dài hạn")
    parser.add_argument("--memory_dropout", type=float, default=0.1, help="Tỷ lệ Memory Dropout")
    parser.add_argument("--lambda_temp", type=float, default=2.0)
    parser.add_argument("--lambda_cross", type=float, default=1.5, help="Trọng số Cross-frame Background Loss")
    parser.add_argument("--lambda_fam", type=float, default=0.5)
    parser.add_argument("--lambda_chroma", type=float, default=0.5)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--lambda_tv", type=float, default=0.02)
    parser.add_argument("--lambda_excl", type=float, default=0.2)
    parser.add_argument("--lambda_bin", type=float, default=0.05)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze_last_blocks", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--stateful_memory",
        action="store_true",
        default=False,
        help="Duy trì bộ nhớ liên tục qua các sliding window của cùng camera (Stateful Streaming Training) để thu hẹp khoảng cách phân phối huấn luyện vs suy luận",
    )
    parser.add_argument(
        "--data_parallel",
        action="store_true",
        default=False,
        help="Bật nn.DataParallel nếu muốn chia batch qua nhiều GPU (Mặc định False để bảo toàn chuỗi bộ nhớ Streaming)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_streaming_bmb(args)
