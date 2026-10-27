"""
=============================================================================
  Hướng G (direction1_new/train.py):
  Entry-point Huấn luyện Chính cho Vehicle-Centric Self-Supervised Learning
  trên DINOv3 từ chuỗi ảnh camera giao thông thưa (Sparse Static-Camera Feeds).

  Kiến trúc Modular (Clean Production Structure):
    - direction1_new.core          : Mô hình Student/Teacher/DINOHead, TAM, AGM, SRS, Losses
    - direction1_new.data          : TrafficCameraDataset, SyntheticTrafficDataset, CameraGroupedSampler
    - direction1_new.engine        : CLI Config, Smooth LR Scheduler, Checkpoint Manager, TAM Warmup
    - direction1_new.visualization : 8-Column Progress Visualizer & Live Loss Curves
=============================================================================
"""

import os
import sys
import copy
import time
import json

# Khắc phục xung đột OpenMP trên Windows (bắt buộc trước khi import torch/numpy)
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
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Đảm bảo đường dẫn import gốc của DINO
_script_dir = os.path.dirname(os.path.abspath(__file__))
_dino_root = os.path.dirname(_script_dir)
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from common.backbone_loader import get_dino_backbone, extract_tokens, imagenet_normalize
from common.gpu_utils import (
    setup_multi_gpu,
    unwrap_model,
    clean_state_dict,
    smart_load_state_dict,
    get_available_devices,
)
from common.logger import setup_training_logger

# Import các thành phần từ hệ thống subpackage modular của direction1_new
from direction1_new.core import (
    DINOHead,
    DirectionGStudentModel,
    DirectionGTeacherModel,
    FrozenExtractor,
    PositionStats,
    GMMCalibrator,
    agm_sample,
    batch_agm_sample,
    static_region_swap,
    batch_static_region_swap,
    batch_vehicle_photometric_jitter,
    DINOLoss,
    iBOTPatchLoss,
    KoLeoLoss,
    PatchVarianceRegularizer,
    FeatureAnchorLoss,
    RICLoss,
    pool_vehicle_embedding,
    CameraAdversarialHead,
    compute_dann_alpha,
)
from direction1_new.data import (
    TrafficCameraDataset,
    SyntheticTrafficDataset,
    CameraGroupedSampler,
)
from direction1_new.engine import (
    parse_args,
    get_cli_specified_args,
    build_smooth_lr_scheduler,
    resolve_resume_checkpoint,
    warmup_tam_offline,
    resolve_or_precompute_tam_cache,
)
from direction1_new.visualization import (
    save_direction_g_visuals,
    save_loss_curves,
)


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

    # Xác định sớm số lượng camera từ thư mục dữ liệu hoặc checkpoint để khởi tạo đúng kích thước đầu ra ngay từ đầu
    initial_num_cams = 16
    if preloaded_ckpt is not None and isinstance(preloaded_ckpt, dict):
        if "pos_stats" in preloaded_ckpt and isinstance(preloaded_ckpt["pos_stats"], dict):
            mu_t = preloaded_ckpt["pos_stats"].get("mu")
            if hasattr(mu_t, "shape") and len(mu_t.shape) > 0:
                initial_num_cams = max(initial_num_cams, int(mu_t.shape[0]))
        if "cam_adv_head" in preloaded_ckpt and isinstance(preloaded_ckpt["cam_adv_head"], dict):
            for k_cam, v_cam in preloaded_ckpt["cam_adv_head"].items():
                if k_cam.endswith("net.6.weight") and hasattr(v_cam, "shape"):
                    initial_num_cams = max(initial_num_cams, int(v_cam.shape[0]))
    if args.data_dir and os.path.isdir(args.data_dir):
        try:
            subdirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
            if len(subdirs) >= 2:
                initial_num_cams = max(initial_num_cams, len(subdirs))
        except Exception:
            pass

    pos_stats = PositionStats(num_cams=initial_num_cams, num_patches=448, num_states=4, feat_dim=64).to(device)
    calibrator = GMMCalibrator(max_buffer_size=args.gmm_buffer_size, ema_alpha=args.gmm_ema_alpha)

    # 4. Losses, Metric Heads & Optimizer
    dino_loss_fn = DINOLoss(out_dim=args.out_dim, nepochs=args.epochs).to(device)
    ibot_loss_fn = iBOTPatchLoss(out_dim=args.out_dim).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)
    patch_var_fn = PatchVarianceRegularizer(target_std=1.0).to(device)
    anchor_fn = FeatureAnchorLoss().to(device)
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
    cam_adv_head.bind_optimizer(optimizer)
    scheduler = build_smooth_lr_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=1e-6,
    )

    # Khôi phục trạng thái từ file resume nếu có (chuẩn an toàn tuyệt đối)
    start_epoch = 0
    best_loss = float("inf")
    history = {
        "total": [], "dino": [], "ibot": [], "koleo": [],
        "patch_var": [], "anchor": [], "ric": [], "cam_adv": [],
        "teacher_entropy": [], "student_entropy": [], "kl_div": [],
        "active_prototypes": [], "teacher_temp": [],
    }

    resume_target = args.resume
    if resume_target:
        cli_specified = get_cli_specified_args(sys.argv[1:])
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

        # 7. Khôi phục Loss Centers & CameraAdversarialHead
        if "dino_center" in ckpt and hasattr(dino_loss_fn, "center"):
            dino_loss_fn.center.copy_(ckpt["dino_center"].to(device))
        if "ibot_center" in ckpt and hasattr(ibot_loss_fn, "center"):
            ibot_loss_fn.center.copy_(ckpt["ibot_center"].to(device))
        if "cam_adv_head" in ckpt and ckpt["cam_adv_head"] is not None:
            cam_sd = clean_state_dict(ckpt["cam_adv_head"])
            if "net.6.weight" in cam_sd and cam_sd["net.6.weight"].shape[0] != cam_adv_head.num_cams:
                cam_adv_head.ensure_cam_capacity(int(cam_sd["net.6.weight"].shape[0]) - 1)
            cam_adv_head.load_state_dict(cam_sd)
            print(f"   🛡️ [CameraAdversarialHead] Khôi phục thành công bộ phân biệt đối kháng ({cam_adv_head.num_cams} cameras).")

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
            if dataset.num_cams > cam_adv_head.num_cams:
                cam_adv_head.ensure_cam_capacity(max(0, dataset.num_cams - 1))
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

    # Khởi tạo Fixed Visual Benchmark Batch (Cố định 4 ảnh đại diện xuyên suốt quá trình train để so sánh chuẩn xác)
    fixed_visual_batch = None
    try:
        fixed_indices = []
        if hasattr(dataset, "cam_keys") and len(dataset.cam_keys) >= 2:
            cam1 = dataset.cam_keys[0]
            cam2 = dataset.cam_keys[min(1, len(dataset.cam_keys) - 1)]
            idxs1 = dataset.camera_to_indices.get(dataset.cam_to_id[cam1], [0])
            idxs2 = dataset.camera_to_indices.get(dataset.cam_to_id[cam2], [0])
            if len(idxs1) >= 2 and len(idxs2) >= 2:
                fixed_indices = [idxs1[0], idxs1[-1], idxs2[0], idxs2[-1]]
            else:
                fixed_indices = (idxs1 + idxs2)[:4]
        if len(fixed_indices) < 4:
            fixed_indices = list(range(min(4, len(dataset))))

        fixed_samples = [dataset[i] for i in fixed_indices]
        fixed_x1 = torch.stack([s["x1"] for s in fixed_samples], dim=0).to(device)
        fixed_x2 = torch.stack([s["x2"] for s in fixed_samples], dim=0).to(device)
        fixed_cids = torch.tensor([s["cid"] for s in fixed_samples], dtype=torch.long, device=device)
        fixed_visual_batch = {"x1": fixed_x1, "x2": fixed_x2, "cids": fixed_cids}
        print(f"🎯 [Visual Benchmark] Đã cố định 4 ảnh mẫu benchmark (Camera IDs: {fixed_cids.tolist()}) để theo dõi tiến trình qua từng epoch.")
    except Exception as e:
        print(f"⚠️ [Visual Benchmark] Không thể khởi tạo tập mẫu cố định ({e}), sẽ fallback sang batch cuối.")

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
            epoch_patch_var_losses = []
            epoch_anchor_losses = []
            epoch_ric_losses = []
            epoch_cam_adv_losses = []
            epoch_teacher_h = []
            epoch_student_h = []
            epoch_kl_div = []
            epoch_prototypes = []

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

                # Bước A: Xác định bản đồ tiền cảnh TAM (ưu tiên cờ random_pi nếu chạy ablation)
                if args.random_pi:
                    # Ablation độc lập: pi ngẫu nhiên, bỏ qua cache
                    pi1 = torch.rand((B, 16, 28), device=device)
                    pi2 = torch.rand((B, 16, 28), device=device)
                elif "pi1" in batch and "pi2" in batch and batch["pi1"] is not None:
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

                        pi1 = calibrator.posterior(a1, refine=args.refine_tam).reshape(B, 16, 28)
                        pi2 = calibrator.posterior(a2, refine=args.refine_tam).reshape(B, 16, 28)

                        # Gated Update: Bảo vệ cụm tĩnh không bị xe ở làn đông làm sai lệch
                        pos_stats.update(cids, u1_f, gate_pi=pi1.reshape(B, 448))

                # Bước B: Áp dụng SRS hoán nền song song toàn bộ batch trên GPU (nhanh gấp 80 lần)
                x_student, _ = batch_static_region_swap(
                    x1=x1, x2=x2, pi1=pi1, pi2=pi2,
                    p_srs=args.p_srs, ratio=0.6, feather=4,
                    continuous_bg=args.srs_continuous,
                    color_match=args.cross_cam_srs,
                )

                # Bước B2: Biến dạng quang học vùng phương tiện (Foreground Vehicle Photometric Augmentation)
                # Tác động lên vùng xe M_veh của Student để phá vỡ sự trùng khớp pixel tuyệt đối với Teacher
                x_student = batch_vehicle_photometric_jitter(
                    x_student=x_student,
                    pi1=pi1,
                    p_jitter=0.8,
                    p_solarize=0.2,
                    feather=4,
                )

                # Bước C: Áp dụng AGM sinh mask che phân tầng Super-block Grid 2x2 cho toàn batch (nhanh gấp 500 lần)
                mask_batch = batch_agm_sample(
                    pi=pi1,
                    ratio=args.mask_ratio,
                    phi=args.phi,
                    q_max=args.q_max,
                    mode=args.mask_mode,
                ).to(device)  # (B, 448) bool

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
                    # Teacher nhận ảnh SẠCH GỐC x1_vit (mỏ neo tri thức không che, không hoán nền)
                    with torch.no_grad():
                        t_cls, t_patches, t_dino_logits, t_ibot_logits = teacher_model(x1_vit)

                    # Student nhận ảnh HOÁN NỀN x_student_vit cùng mặt nạ che phân tầng AGM masks
                    # Ràng buộc bất biến Cross-Context: Student phải tái tạo đặc trưng phương tiện khớp với Teacher gốc!
                    s_cls, s_patches, s_dino_logits, s_ibot_logits = student_model(x_student_vit, masks=mask_batch)

                    # Bước E: Tính toán toàn bộ hệ thống Loss
                    loss_dino = dino_loss_fn(s_dino_logits, t_dino_logits, epoch=epoch)

                    # iBOT Patch Loss: Trọng số hóa theo bản đồ tiền cảnh pi1
                    loss_ibot = ibot_loss_fn(s_ibot_logits, t_ibot_logits, mask_batch, pi=pi1.reshape(B, 448), beta=args.ibot_beta)

                    # KoLeo Loss trên CLS token
                    loss_koleo = koleo_loss_fn(s_cls)

                    # Patch-level Variance Regularizer: Ngăn chặn triệt để Dimensional Collapse ở mức patch!
                    if args.use_patch_var:
                        loss_patch_var = patch_var_fn(s_patches, pi=pi1.reshape(B, 448))
                    else:
                        loss_patch_var = torch.tensor(0.0, device=device)

                    # Feature Anchor Loss: Giữ vững không gian biểu diễn DINOv3, chống trôi dạt (representation drift)
                    if args.use_anchor:
                        with torch.no_grad():
                            x1_norm = imagenet_normalize(x1_vit)
                            _, v_patch_spatial = extract_tokens(frozen_backbone, x1_norm, patch_size=patch_size)
                            v_patch_tokens_anchor = v_patch_spatial.reshape(B, -1, embed_dim)
                        loss_anchor = anchor_fn(s_patches, v_patch_tokens_anchor)
                    else:
                        loss_anchor = torch.tensor(0.0, device=device)

                    # Regional Instance Contrastive (RIC): ép mô hình tập trung vào xe qua hard-negatives cùng camera
                    # Bật drop_prob=0.25 trên Student để kết hợp cùng Vehicle Photometric Jitter tạo cặp dương bất đối xứng
                    if args.use_ric:
                        z_veh_s = pool_vehicle_embedding(s_patches, pi1.reshape(B, -1), drop_prob=0.25)
                        z_veh_t = pool_vehicle_embedding(t_patches, pi1.reshape(B, -1), drop_prob=0.0)
                        loss_ric = ric_loss_fn(z_veh_s, z_veh_t, cids)
                    else:
                        loss_ric = torch.tensor(0.0, device=device)

                    # Camera Adversarial Loss (GRL): ép xóa bỏ thông tin camera ID trong CLS token
                    if args.use_cam_adv:
                        current_progress = float(step) / float(max(1, total_train_steps))
                        current_grl_alpha = compute_dann_alpha(current_progress, max_alpha=args.max_grl_alpha, gamma=10.0)
                        loss_cam_adv = cam_adv_head(s_cls, cids, grl_alpha=current_grl_alpha)
                    else:
                        loss_cam_adv = torch.tensor(0.0, device=device)

                    total_loss = (
                        loss_dino
                        + 1.0 * loss_ibot
                        + 0.02 * loss_koleo
                        + args.lambda_patch_var * loss_patch_var
                        + args.lambda_anchor * loss_anchor
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
                epoch_patch_var_losses.append(loss_patch_var.item())
                epoch_anchor_losses.append(loss_anchor.item())
                epoch_ric_losses.append(loss_ric.item())
                epoch_cam_adv_losses.append(loss_cam_adv.item())

                with torch.no_grad():
                    try:
                        b_dist = dino_loss_fn.compute_distribution_metrics(s_dino_logits, t_dino_logits, epoch=epoch)
                        epoch_teacher_h.append(float(b_dist["teacher_entropy"]))
                        epoch_student_h.append(float(b_dist["student_entropy"]))
                        epoch_kl_div.append(float(b_dist["kl_div"]))
                        epoch_prototypes.append(int(b_dist["active_prototypes"]))
                    except Exception:
                        pass

                # Lưu dữ liệu visual thực tế của batch cuối phục vụ fallback
                B_vis = min(4, x1.shape[0])
                with torch.no_grad():
                    x1_vis = x1[:B_vis].to(device)
                    x1_vis_norm = imagenet_normalize(x1_vis)
                    _, v_patch_spatial = extract_tokens(frozen_backbone, x1_vis_norm, patch_size=patch_size)
                    v_patch_tokens = v_patch_spatial.reshape(B_vis, -1, embed_dim).cpu().float()

                    # Trích xuất đặc trưng THỰC SỰ của Student trên ảnh sạch x1 (không che mask)
                    _, s_clean_patches, _, _ = student_model(x1_vis, masks=None)
                    s_patch_tokens_real = s_clean_patches.reshape(B_vis, -1, embed_dim).cpu().float()

                last_visual_data = {
                    "x_orig": x1[:B_vis].detach().cpu(),
                    "pi": pi1[:B_vis].detach().cpu(),
                    "mask": mask_batch[:B_vis].detach().cpu(),
                    "x_srs": x_student[:B_vis].detach().cpu(),  # Giữ nguyên ảnh SRS thực tế của bước train
                    "patch_features": s_patch_tokens_real,      # ĐẶC TRƯNG CHUẨN XÁC TỪ STUDENT
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
            avg_patch_var = float(np.mean(epoch_patch_var_losses)) if epoch_patch_var_losses else 0.0
            avg_anchor = float(np.mean(epoch_anchor_losses)) if epoch_anchor_losses else 0.0
            avg_ric = float(np.mean(epoch_ric_losses)) if epoch_ric_losses else 0.0
            avg_cam_adv = float(np.mean(epoch_cam_adv_losses)) if epoch_cam_adv_losses else 0.0

            history["total"].append(avg_tot)
            history["dino"].append(avg_dino)
            history["ibot"].append(avg_ibot)
            history["koleo"].append(avg_koleo)
            history["patch_var"].append(avg_patch_var)
            history["anchor"].append(avg_anchor)
            history["ric"].append(avg_ric)
            history["cam_adv"].append(avg_cam_adv)

            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"📊 [Epoch {epoch+1}/{args.epochs} Hoàn Tất] LR: {cur_lr:.2e} — Loss TB: {avg_tot:.4f} "
                  f"(DINO: {avg_dino:.4f}, iBOT: {avg_ibot:.4f}, KoLeo: {avg_koleo:.4f}, "
                  f"PatchVar: {avg_patch_var:.4f}, Anchor: {avg_anchor:.4f}, "
                  f"RIC: {avg_ric:.4f}, CamAdv: {avg_cam_adv:.4f})")

            # Giám sát phân bố Teacher-Student (trung bình toàn epoch) và lưu vào history / training_metrics.json
            try:
                cur_t_temp = float(dino_loss_fn.teacher_temp_schedule[epoch]) if hasattr(dino_loss_fn, "teacher_temp_schedule") and epoch < len(dino_loss_fn.teacher_temp_schedule) else 0.07
                avg_th = round(float(np.mean(epoch_teacher_h)), 4) if epoch_teacher_h else 0.0
                avg_sh = round(float(np.mean(epoch_student_h)), 4) if epoch_student_h else 0.0
                avg_kl = round(float(np.mean(epoch_kl_div)), 4) if epoch_kl_div else 0.0
                avg_proto = int(round(float(np.mean(epoch_prototypes)))) if epoch_prototypes else 0

                history.setdefault("teacher_entropy", []).append(avg_th)
                history.setdefault("student_entropy", []).append(avg_sh)
                history.setdefault("kl_div", []).append(avg_kl)
                history.setdefault("active_prototypes", []).append(avg_proto)
                history.setdefault("teacher_temp", []).append(round(cur_t_temp, 4))

                print(f"   📈 [DINO Giám Sát] Teacher Temp: {cur_t_temp:.4f} | Teacher H: {avg_th:.4f} | "
                      f"Student H: {avg_sh:.4f} | KL Div: {avg_kl:.4f} | Prototypes: {avg_proto}")
            except Exception:
                pass

            # Cập nhật hình ảnh trực quan hóa: Ưu tiên dùng Fixed Visual Benchmark để so sánh chuẩn xác qua các epoch
            vis_save_path = os.path.join(args.output_dir, "direction1_new_progress.png")
            if fixed_visual_batch is not None:
                try:
                    with torch.no_grad():
                        fx1 = fixed_visual_batch["x1"]
                        fx2 = fixed_visual_batch["x2"]
                        fcids = fixed_visual_batch["cids"]
                        B_f = fx1.shape[0]

                        # 1. Trích xuất đặc trưng DINO gốc
                        fx1_norm = imagenet_normalize(fx1)
                        _, fv_patch_spatial = extract_tokens(frozen_backbone, fx1_norm, patch_size=patch_size)
                        fv_patch_tokens = fv_patch_spatial.reshape(B_f, -1, embed_dim).cpu().float()

                        # 2. Trích xuất đặc trưng thực thụ của Student (trên ảnh sạch không che)
                        _, fs_patches_clean, _, _ = student_model(fx1, masks=None)
                        fs_patch_tokens = fs_patches_clean.reshape(B_f, -1, embed_dim).cpu().float()

                        # 3. Tính TAM và AGM trên tập mẫu cố định
                        fu1 = frozen_extractor(fx1).float()
                        fa1, _ = pos_stats.atypicality(fcids, fu1)
                        fpi1 = calibrator.posterior(fa1, refine=args.refine_tam).reshape(B_f, 16, 28)

                        fu2 = frozen_extractor(fx2).float()
                        fa2, _ = pos_stats.atypicality(fcids, fu2)
                        fpi2 = calibrator.posterior(fa2, refine=args.refine_tam).reshape(B_f, 16, 28)

                        # 4. Sinh ảnh SRS hoán nền và mặt nạ AGM
                        fx_srs_list = []
                        fmask_list = []
                        for b_i in range(B_f):
                            f_sw, _ = static_region_swap(
                                fx1[b_i], fx2[b_i], fpi1[b_i], fpi2[b_i],
                                ratio=0.7, feather=4, continuous_bg=True, color_match=True
                            )
                            fx_srs_list.append(f_sw)

                            fm_b = agm_sample(
                                pi=fpi1[b_i].reshape(-1),
                                ratio=args.mask_ratio,
                                phi=args.phi,
                                q_max=args.q_max,
                                mode=args.mask_mode,
                            )
                            fmask_list.append(fm_b)

                        fx_srs_tensor = torch.stack(fx_srs_list, dim=0).cpu()
                        fmask_tensor = torch.stack(fmask_list, dim=0).cpu()

                    save_direction_g_visuals(
                        x_orig=fx1.detach().cpu(),
                        pi=fpi1.detach().cpu(),
                        mask=fmask_tensor,
                        x_srs=fx_srs_tensor,
                        patch_features=fs_patch_tokens,  # Student đặc trưng sạch
                        save_path=vis_save_path,
                        epoch=epoch + 1,
                        vanilla_patch_features=fv_patch_tokens,
                    )
                except Exception as e_vis:
                    print(f"⚠️ [Visualizer] Lỗi suy luận benchmark ({e_vis}), fallback sang batch cuối.")
                    if last_visual_data is not None:
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
            elif last_visual_data is not None:
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
