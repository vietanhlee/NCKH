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
from common.gpu_utils import (
    setup_multi_gpu,
    unwrap_model,
    save_checkpoint,
    load_checkpoint,
    resolve_checkpoint_path,
    smart_inherit_checkpoint_args,
    get_cli_specified_args,
)
from common.logger import setup_training_logger
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
        arr = t.squeeze().detach().float().cpu().permute(1, 2, 0).numpy()
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def to_gray(t):
        arr = t.squeeze().detach().float().cpu().numpy()
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

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

        sigma_np = sigma_t.squeeze().detach().float().cpu().numpy()
        sigma_norm = np.clip((sigma_np - 0.01) / (0.50 - 0.01), 0.0, 1.0).astype(np.float32)

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
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    except Exception as e_s:
        print(f"   ⚠️ [Visual Sample Error] Không thể lưu ảnh {save_path}: {e_s}")
    finally:
        plt.close()


def train_streaming_bmb(args):
    # Tự động kích hoạt cơ chế lưu toàn bộ log phiên huấn luyện vào file
    setup_training_logger(
        save_dir=args.save_dir,
        log_file=getattr(args, "log_file", None),
        session_title=f"HUẤN LUYỆN ST-BMB STREAMING ({args.backbone})",
    )

    # 0. Tiền xử lý Resume: Nạp checkpoint sớm để đồng bộ siêu tham số (Smart Hyperparameters Restoration)
    preloaded_ckpt = None
    if getattr(args, "resume", None):
        args.resume = resolve_checkpoint_path(args.resume)
        print(f"🔄 [Resume Preload] Đang kiểm tra checkpoint từ: {args.resume}")
        try:
            preloaded_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        except TypeError:
            preloaded_ckpt = torch.load(args.resume, map_location="cpu")

        if isinstance(preloaded_ckpt, dict) and "args" in preloaded_ckpt and isinstance(preloaded_ckpt["args"], dict):
            sync_keys = [
                "lr", "window_size", "stride", "img_size", "embed_dim", "num_heads",
                "max_recent", "max_anchor", "ema_eta", "memory_dropout",
                "lambda_temp", "lambda_cross", "lambda_fam", "lambda_chroma",
                "lambda_sparse", "lambda_tv", "lambda_excl", "lambda_bin",
                "warmup_epochs", "batch_size", "backbone", "weights",
            ]
            smart_inherit_checkpoint_args(args, preloaded_ckpt["args"], sync_keys)

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
    if getattr(args, "hf_token", None):
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    effective_backbone_weights = args.weights or (args.resume if (args.resume and os.path.isfile(args.resume)) else None)
    print(f"\n🧠 [Backbone] Đang tải mô hình {args.backbone}...")
    backbone, feat_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        weights_path=effective_backbone_weights,
        freeze=args.freeze_backbone,
        hf_token=getattr(args, "hf_token", None),
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

    # 3. Thiết lập Thiết bị & Multi-GPU Tự Động (Auto-Detect All Available GPUs)
    # Nhờ Per-Device Memory Bank Isolation (_MemoryBankDeviceState), ST-BMB hỗ trợ 100% DataParallel song song không race condition.
    if getattr(args, "single_gpu", False):
        # Chế độ ép chạy Single GPU do người dùng chỉ định rõ
        if args.device.lower() == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
            num_gpus = 0
            print("🖥️ [Hardware] Chạy trên CPU theo yêu cầu.")
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
            print(f"⚡ [Hardware] ST-BMB ép chạy trên đơn GPU theo yêu cầu (--single_gpu): {gpu_name} ({device}) | VRAM: {mem_gb:.2f} GB")

        model = model.to(device)
        effective_batch_size = args.batch_size
        effective_lr = args.lr
    else:
        # TỰ ĐỘNG nhận diện toàn bộ GPU và kích hoạt Multi-GPU DataParallel
        model, device, num_gpus, effective_batch_size, effective_lr = setup_multi_gpu(
            model=model,
            batch_size_per_gpu=args.batch_size,
            base_lr=args.lr,
            device_arg=args.device,
            scale_lr=getattr(args, "scale_lr", False),
        )

    raw_model = unwrap_model(model)
    raw_model.set_memory_dropout(args.memory_dropout)

    # 3.1. Thiết lập Huấn luyện Mixed Precision (AMP) — Tự động chọn BFloat16 (Ada/Ampere) hoặc Float16 (Tesla T4/Turing)
    def is_native_bf16_supported():
        if not torch.cuda.is_available():
            return False
        try:
            cap = torch.cuda.get_device_capability()
            # Compute Capability >= 8.0: Ampere (A100, RTX 3090), Ada Lovelace (RTX 4090), Hopper
            return cap[0] >= 8 and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        except Exception:
            return False

    use_amp = bool(getattr(args, "amp", True)) and torch.cuda.is_available() and device.type == "cuda"
    use_bf16 = use_amp and is_native_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    # Tương thích PyTorch 2.x (torch.amp) và PyTorch 1.x (torch.cuda.amp)
    # Cấu hình init_scale=2048.0 để chống tràn định kỳ của GradScaler trên Turing / Tesla T4
    scaler_kwargs = {
        "enabled": (use_amp and not use_bf16),
        "init_scale": 2048.0,
        "growth_factor": 1.5,
        "backoff_factor": 0.5,
    }
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", **scaler_kwargs)
    else:
        scaler = torch.cuda.amp.GradScaler(**scaler_kwargs)

    def get_autocast(enabled_flag, dtype_val):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", dtype=dtype_val, enabled=enabled_flag)
        return torch.cuda.amp.autocast(dtype=dtype_val, enabled=enabled_flag)

    if use_amp:
        dtype_desc = "BFloat16 (Native Ada Lovelace/Tensor Cores)" if use_bf16 else "Float16 (GradScaler - Turing/Tesla T4/V100 Tensor Cores)"
        print(f"⚡ [AMP] Đã kích hoạt Mixed Precision Training: {dtype_desc}")

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

    # Tự động tối ưu số workers nếu người dùng chưa chỉ định rõ
    if getattr(args, "num_workers", None) is None or args.num_workers < 0:
        actual_workers = min(8, max(2, (os.cpu_count() or 4) // 2)) if torch.cuda.is_available() else 0
    else:
        actual_workers = int(args.num_workers)

    loader_kwargs = {
        "batch_size": effective_batch_size,
        "num_workers": actual_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if actual_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=(len(train_dataset) > effective_batch_size),
        **loader_kwargs,
    )

    print(f"📦 [Dataloader] Đã tạo DataLoader với {actual_workers} workers | Batch size: {effective_batch_size} | Cửa sổ train: {len(train_dataset)}")
    if args.stride == 1 and len(train_dataset) > 5000:
        print(
            f"💡 [Tối ưu tốc độ] Tập dữ liệu lớn ({len(train_dataset)} cửa sổ) với --stride 1 chứa 80% frame trùng lặp giữa các window kế tiếp.\n"
            f"   Khuyến nghị: Thêm '--stride 3' hoặc '--stride 5' và '--batch_size 8' để giảm số bước 12-20 lần mà vẫn giữ nguyên độ đa dạng cảnh!"
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
                shuffle=False,
                **loader_kwargs,
            )
            print(f"📊 [Validation] Đã khởi tạo tập kiểm định: {len(val_dataset)} cửa sổ.")
        except Exception as e_v:
            print(f"ℹ️ [Validation Note] Bỏ qua val split: {e_v}")

    # 5. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=effective_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Hàm mất mát đa mục tiêu chuẩn Production (kèm Robust Matting chống suy biến)
    loss_fn = StreamingDecompositionLoss(
        lambda_temp=args.lambda_temp,
        lambda_cross=args.lambda_cross,
        lambda_fam=args.lambda_fam,
        lambda_chroma=args.lambda_chroma,
        lambda_sparse=args.lambda_sparse,
        lambda_tv=args.lambda_tv,
        lambda_excl=args.lambda_excl,
        lambda_bin=args.lambda_bin,
        lambda_cost=args.lambda_cost,
        lambda_robust=args.lambda_robust,
    ).to(device)

    # 6. Khôi phục từ checkpoint nếu có cờ --resume
    start_epoch = 0
    best_loss = float("inf")
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

    if getattr(args, "resume", None):
        print(f"\n🔄 [Resume] Khôi phục toàn bộ trạng thái huấn luyện từ checkpoint: {args.resume}")
        ckpt_data = preloaded_ckpt if preloaded_ckpt is not None else load_checkpoint(
            load_path=args.resume,
            model=unwrap_model(model),
            optimizer=optimizer,
            scheduler=None,
            device=device,
            strict=False,
            verbose=True,
        )
        if preloaded_ckpt is not None:
            from common.gpu_utils import smart_load_state_dict
            candidate_keys = ["model_state", "state_dict", "student_state", "model", "student"]
            model_sd = None
            if isinstance(ckpt_data, dict):
                for ck in candidate_keys:
                    if ck in ckpt_data and isinstance(ckpt_data[ck], dict):
                        model_sd = ckpt_data[ck]
                        break
            if model_sd is None:
                model_sd = ckpt_data
            smart_load_state_dict(unwrap_model(model), model_sd, strict=False, verbose=True)
            if "optimizer" in ckpt_data:
                try:
                    optimizer.load_state_dict(ckpt_data["optimizer"])
                    dest_device = torch.device(device)
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(dest_device)
                    print("   ✅ [Optimizer] Khôi phục toàn bộ trạng thái optimizer.")
                except Exception as e_opt:
                    print(f"   ⚠️ [Optimizer Notice] {e_opt}")

        if "epoch" in ckpt_data and ckpt_data["epoch"] is not None:
            start_epoch = int(ckpt_data["epoch"])
            print(f"   ⏱️ [Epoch] Khôi phục tại epoch {start_epoch}. Sẽ tiếp tục chạy từ epoch {start_epoch + 1}.")
            cli_specified = get_cli_specified_args(sys.argv[1:])
            if args.epochs <= start_epoch:
                if "epochs" not in cli_specified:
                    extra_epochs = 5
                    if "args" in ckpt_data and isinstance(ckpt_data["args"], dict) and "epochs" in ckpt_data["args"]:
                        extra_epochs = int(ckpt_data["args"]["epochs"])
                    target_epochs = start_epoch + extra_epochs
                    print(f"   💡 [Gia hạn Epochs] Số epochs cài đặt ({args.epochs}) <= epoch checkpoint ({start_epoch}).")
                    print(f"      -> Tự động huấn luyện thêm {extra_epochs} epochs (Tổng mới: {target_epochs} epochs).")
                    args.epochs = target_epochs
                else:
                    print(f"   ⚠️ [Cảnh báo] --epochs {args.epochs} <= epoch hiện tại ({start_epoch}). Vui lòng tăng --epochs!")

        # Tái lập CosineAnnealingLR mượt mà (chống loss bùng nổ do gradient shock)
        eta_min = 1e-6
        curr_cosine_lr = eta_min + 0.5 * (effective_lr - eta_min) * (1.0 + np.cos(start_epoch * np.pi / max(1, args.epochs)))
        for group in optimizer.param_groups:
            group["initial_lr"] = effective_lr
            group["lr"] = float(curr_cosine_lr)

        remaining_epochs = max(1, args.epochs - start_epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=eta_min,
            last_epoch=start_epoch - 1 if start_epoch > 0 else -1,
        )
        print(f"   📈 [Scheduler Restored] Tái lập CosineAnnealingLR: Mục tiêu {args.epochs} epochs (còn {remaining_epochs} epochs).")
        print(f"   🎯 [Current Learning Rate] LR khởi điểm cho epoch {start_epoch + 1}: {curr_cosine_lr:.3e} (suy giảm mượt mà về 1e-6, triệt tiêu gradient shock).")

        if "metrics" in ckpt_data and isinstance(ckpt_data["metrics"], dict):
            saved_loss = ckpt_data["metrics"].get("loss", None)
            if saved_loss is not None:
                best_loss = float(saved_loss)
                print(f"   🏆 [Best Loss] Khôi phục kỷ lục loss tốt nhất trước đó: {best_loss:.4f}")

    print("\n🚀 Bắt đầu huấn luyện...")
    for epoch in range(start_epoch, args.epochs):
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

        total_train_batches = len(train_loader)
        max_batches = getattr(args, "batches_per_epoch", None)
        steps_this_epoch = min(total_train_batches, max_batches) if (max_batches is not None and max_batches > 0) else total_train_batches

        pbar = tqdm(train_loader, total=steps_this_epoch, desc=f"Epoch [{epoch+1}/{args.epochs}] (Warmup={warmup_factor:.2f})", leave=True)
        last_cam_id: Optional[str] = None
        sample_batch_cache = None
        batches_processed = 0
        batches_valid_in_epoch = 0

        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
                break
            batches_processed += 1

            if sample_batch_cache is None:
                sample_batch_cache = {
                    "frames": batch["frames"][:1].clone(),
                    "timestamps": batch["timestamps"][:1].clone() if "timestamps" in batch else None,
                }

            frames = batch["frames"].to(device, non_blocking=True)  # (B, W, 3, H, W)
            timestamps = batch["timestamps"].to(device, non_blocking=True) if "timestamps" in batch else None
            jitter_g = batch["jitter_gain"].to(device, non_blocking=True) if "jitter_gain" in batch and torch.is_tensor(batch["jitter_gain"]) else (
                torch.tensor(batch["jitter_gain"], device=device) if "jitter_gain" in batch else None
            )
            jitter_b = batch["jitter_bias"].to(device, non_blocking=True) if "jitter_bias" in batch and torch.is_tensor(batch["jitter_bias"]) else (
                torch.tensor(batch["jitter_bias"], device=device) if "jitter_bias" in batch else None
            )
            fov_mask = batch["fov_mask"].to(device, non_blocking=True) if "fov_mask" in batch and batch["fov_mask"] is not None else None

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

            optimizer.zero_grad()
            if use_amp:
                with get_autocast(use_amp, amp_dtype):
                    step_outputs = model(frames, timestamps=timestamps, reset_memory=reset_memory)
                    total_loss, loss_dict = loss_fn(
                        step_outputs,
                        frames,
                        target_jitter_gain=jitter_g,
                        target_jitter_bias=jitter_b,
                        fov_mask=fov_mask,
                    )

                # [Active NaN Guard] Tầng 1: Kiểm tra tính hữu hạn của tổng hàm mất mát
                if not torch.isfinite(total_loss):
                    rec_v = loss_dict.get("loss_rec", 0.0)
                    temp_v = loss_dict.get("loss_temp_bg", 0.0)
                    cross_v = loss_dict.get("loss_cross", 0.0)
                    print(f"\n⚠️ [Active NaN Guard] Phát hiện loss không hợp lệ (total_loss={total_loss.item():.4f} | rec={rec_v:.4f}, temp={temp_v:.4f}, cross={cross_v:.4f}) tại step {batch_idx + 1}. Bỏ qua batch và giải phóng memory bank!")
                    optimizer.zero_grad()
                    raw_model.reset_memory()
                    continue

                if not use_bf16:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.8)
                    # [Active NaN Guard] Tầng 2: Kiểm tra bùng nổ gradient sau khi unscale
                    if not torch.isfinite(grad_norm):
                        print(f"\n⚠️ [Active NaN Guard] Phát hiện gradient không hợp lệ (grad_norm={grad_norm.item()}) tại step {batch_idx + 1}. Bỏ qua cập nhật trọng số và điều chỉnh hạ scale factor!")
                        optimizer.zero_grad()
                        scaler.update()
                        raw_model.reset_memory()
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.8)
                    if not torch.isfinite(grad_norm):
                        print(f"\n⚠️ [Active NaN Guard] Phát hiện gradient không hợp lệ (grad_norm={grad_norm.item()}) tại step {batch_idx + 1}. Bỏ qua cập nhật trọng số!")
                        optimizer.zero_grad()
                        raw_model.reset_memory()
                        continue
                    optimizer.step()
            else:
                step_outputs = model(frames, timestamps=timestamps, reset_memory=reset_memory)
                total_loss, loss_dict = loss_fn(
                    step_outputs,
                    frames,
                    target_jitter_gain=jitter_g,
                    target_jitter_bias=jitter_b,
                    fov_mask=fov_mask,
                )

                if not torch.isfinite(total_loss):
                    rec_v = loss_dict.get("loss_rec", 0.0)
                    temp_v = loss_dict.get("loss_temp_bg", 0.0)
                    cross_v = loss_dict.get("loss_cross", 0.0)
                    print(f"\n⚠️ [Active NaN Guard] Phát hiện loss không hợp lệ (total_loss={total_loss.item():.4f} | rec={rec_v:.4f}, temp={temp_v:.4f}, cross={cross_v:.4f}) tại step {batch_idx + 1}. Bỏ qua batch!")
                    optimizer.zero_grad()
                    raw_model.reset_memory()
                    continue

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.8)
                if not torch.isfinite(grad_norm):
                    print(f"\n⚠️ [Active NaN Guard] Phát hiện gradient không hợp lệ (grad_norm={grad_norm.item()}) tại step {batch_idx + 1}. Bỏ qua cập nhật trọng số!")
                    optimizer.zero_grad()
                    raw_model.reset_memory()
                    continue
                optimizer.step()

            # Tích lũy số liệu chỉ từ các batch hợp lệ
            if math.isfinite(loss_dict["loss_total"]):
                epoch_loss += loss_dict["loss_total"]
                epoch_rec += loss_dict["loss_rec"]
                epoch_temp += loss_dict["loss_temp_bg"]
                epoch_cross += loss_dict["loss_cross"]
                epoch_fam += loss_dict["loss_fam"]
                epoch_sparse += loss_dict["loss_sparse"]
                epoch_sigma += loss_dict["mean_sigma"]
                epoch_soft_w += loss_dict.get("mean_soft_weight", 1.0)
                batches_valid_in_epoch += 1

            pbar.set_postfix({
                "loss": f"{loss_dict['loss_total']:.4f}",
                "rec": f"{loss_dict['loss_rec']:.4f}",
                "temp": f"{loss_dict['loss_temp_bg']:.4f}",
                "cross": f"{loss_dict['loss_cross']:.4f}",
                "fam": f"{loss_dict['loss_fam']:.4f}",
                "sig": f"{loss_dict['mean_sigma']:.3f}",
            })

        scheduler.step()
        num_batches = max(1, batches_valid_in_epoch)
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
            if sample_batch_cache is not None:
                sample_frames = sample_batch_cache["frames"].to(device)
                sample_ts = sample_batch_cache["timestamps"].to(device) if sample_batch_cache["timestamps"] is not None else None
            else:
                sample_batch = next(iter(train_loader))
                sample_frames = sample_batch["frames"][:1].to(device)
                sample_ts = sample_batch["timestamps"][:1].to(device) if "timestamps" in sample_batch else None

            try:
                with get_autocast(use_amp, amp_dtype):
                    sample_outputs = raw_model(sample_frames, timestamps=sample_ts, reset_memory=True)

                epoch_vis_path = os.path.join(vis_dir, f"epoch_{epoch+1:03d}.png")
                progress_path = os.path.join(args.save_dir, "streaming_progress.png")
                save_streaming_visual_sample(sample_frames, sample_outputs, epoch_vis_path, epoch=epoch + 1)
                save_streaming_visual_sample(sample_frames, sample_outputs, progress_path, epoch=epoch + 1)
            except Exception as e_vis:
                print(f"⚠️ [Visual Sample Notice] Bỏ qua xuất ảnh epoch {epoch+1}: {e_vis}")

        # Đánh giá trên tập kiểm định (Validation) nếu có val_loader
        val_loss = None
        if val_loader is not None and len(val_loader) > 0:
            raw_model.eval()
            val_loss_acc = 0.0
            val_rec_acc = 0.0
            val_cross_acc = 0.0
            val_batches = 0
            max_vb = getattr(args, "max_val_batches", 100)
            with torch.no_grad():
                # Tạm thời đặt warmup_factor = 1.0 để đánh giá công bằng tuyệt đối giữa các epoch
                loss_fn.set_warmup_factor(1.0)
                for v_idx, val_batch in enumerate(val_loader):
                    if max_vb is not None and max_vb > 0 and v_idx >= max_vb:
                        break
                    v_frames = val_batch["frames"].to(device, non_blocking=True)
                    v_ts = val_batch["timestamps"].to(device, non_blocking=True) if "timestamps" in val_batch else None
                    v_fov = val_batch["fov_mask"].to(device, non_blocking=True) if "fov_mask" in val_batch and val_batch["fov_mask"] is not None else None
                    with get_autocast(use_amp, amp_dtype):
                        v_outs = raw_model(v_frames, timestamps=v_ts, reset_memory=True)
                        v_loss, v_dict = loss_fn(v_outs, v_frames, fov_mask=v_fov)
                    if torch.isfinite(v_loss):
                        val_loss_acc += v_loss.item()
                        val_rec_acc += v_dict.get("loss_rec", 0.0)
                        val_cross_acc += v_dict.get("loss_cross", 0.0)
                        val_batches += 1
                    else:
                        print(f"⚠️ [Validation Notice] Phát hiện validation loss không hợp lệ tại val batch {v_idx + 1}. Bỏ qua batch này!")

                # Khôi phục warmup_factor cho epoch tiếp theo
                loss_fn.set_warmup_factor(warmup_factor)

            if val_batches > 0:
                val_loss = val_loss_acc / val_batches
                val_rec = val_rec_acc / val_batches
                val_cross = val_cross_acc / val_batches
                history.setdefault("val_loss", []).append(float(val_loss))
                history.setdefault("val_rec", []).append(float(val_rec))
                history.setdefault("val_cross", []).append(float(val_cross))
                print(
                    f"   🧪 [Validation Epoch {epoch+1:02d}] Val Loss: {val_loss:.4f} "
                    f"(Rec: {val_rec:.4f}, Cross: {val_cross:.4f})"
                )
            raw_model.train()

        # Tiêu chí chọn Best Checkpoint: Ưu tiên val_loss (nếu có), nếu không dùng train avg_loss
        eval_metric = val_loss if val_loss is not None else avg_loss
        if eval_metric < best_loss:
            best_loss = eval_metric
            ckpt_best = os.path.join(args.save_dir, "best_streaming_model.pth")
            save_checkpoint(
                save_path=ckpt_best,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics={
                    "loss": eval_metric,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "rec": avg_rec,
                    "cross": avg_cross,
                    "temp_bg": avg_temp,
                },
                extra_dict={"num_gpus": num_gpus, "args": vars(args)},
                verbose=False,
            )
            print(f"   🏆 [Best Checkpoint] Đã cập nhật best_streaming_model.pth (Metric: {best_loss:.4f})")

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
    parser.add_argument("--lambda_cost", type=float, default=0.10, help="Chi phí gán pixel phương tiện (lambda_cost) trong công thức Robust Matting")
    parser.add_argument("--lambda_robust", type=float, default=1.0, help="Trọng số mất mát Robust Background Explanation nội khung hình")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze_last_blocks", type=int, default=0)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Số luồng tải dữ liệu song song (Mặc định: None - Tự động thiết lập 4-8 workers trên GPU để triệt tiêu nghẽn I/O CPU)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Kích hoạt Mixed Precision Training (BFloat16/FP16) trên Tensor Cores để tăng tốc 2.5x-3x (Mặc định: True)",
    )
    parser.add_argument(
        "--no_amp",
        dest="amp",
        action="store_false",
        help="Tắt Mixed Precision Training (ép chạy FP32)",
    )
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=None,
        help="Số batches tối đa huấn luyện trong mỗi epoch (vd: 1000). Giúp 1 epoch hoàn thành nhanh, lưu checkpoint và vẽ ảnh trực quan đều đặn",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=100,
        help="Số batches tối đa để đánh giá trên tập validation (mặc định: 100)",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--stateful_memory",
        action="store_true",
        default=False,
        help="Duy trì bộ nhớ liên tục qua các sliding window của cùng camera (Stateful Streaming Training) để thu hẹp khoảng cách phân phối huấn luyện vs suy luận",
    )
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        default=False,
        help="Ép chạy trên duy nhất 1 GPU ngay cả khi hệ thống có nhiều GPU (Mặc định: False - tự động nhận diện và sử dụng toàn bộ GPU)",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Tự động nhân LR theo số GPU khi dùng DataParallel (Mặc định: False để bảo toàn tính ổn định cho Vision Transformer)",
    )
    parser.add_argument(
        "--data_parallel",
        action="store_true",
        default=False,
        help="Tương thích ngược: kích hoạt DataParallel (Hệ thống hiện tại đã MẶC ĐỊNH tự động bật khi có đa GPU)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Đường dẫn checkpoint (.pth) để tiếp tục huấn luyện ST-BMB",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face user access token (để tải trọng số DINOv3 tự động nếu dùng weights trên HF)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_file", type=str, default=None, help="Tên hoặc đường dẫn file log ghi lại toàn bộ output (mặc định: train.log trong --save_dir)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_streaming_bmb(args)
