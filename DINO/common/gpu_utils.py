"""
=============================================================================
 Common Utility: Multi-GPU Auto-Detection, Scaling & Smart Checkpointing
 Tự động nhận diện toàn bộ số lượng GPU trên hệ thống (Kaggle 2x T4, Multi-GPU Server)
 và xử lý thông minh, an toàn tuyệt đối khi Lưu & Nạp trọng số mô hình (Zero module. error)
=============================================================================
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


def get_available_devices() -> Tuple[torch.device, int, List[str]]:
    """
    Tự động dò quét phần cứng và trả về thông tin GPU khả dụng.

    Returns:
        primary_device: Thiết bị chính ('cuda:0' hoặc 'cpu').
        num_gpus: Số lượng GPU vật lý tìm thấy (0 nếu chỉ có CPU).
        gpu_names: Danh sách tên phần cứng các GPU.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        primary_device = torch.device("cuda:0")
    else:
        num_gpus = 0
        gpu_names = []
        primary_device = torch.device("cpu")

    return primary_device, num_gpus, gpu_names


def setup_multi_gpu(
    model: nn.Module,
    batch_size_per_gpu: int = 16,
    base_lr: float = 2e-4,
    device_arg: str = "cuda",
    scale_lr: bool = True,
) -> Tuple[nn.Module, torch.device, int, int, float]:
    """
    Tự động cấu hình mô hình để tận dụng toàn bộ số lượng GPU tìm thấy.

    Nguyên lý hoạt động:
      - Nếu tìm thấy N > 1 GPU:
          + Bọc mô hình bằng `nn.DataParallel` để phân phối tensor song song trên tất cả các GPU.
          + Tự động nhân rộng tổng Batch Size: `total_batch_size = batch_size_per_gpu * N`.
          + Tự động áp dụng Linear Scaling Rule cho Learning Rate: `effective_lr = base_lr * N`.
      - Nếu N == 1 GPU: Sử dụng đơn GPU cuda:0.
      - Nếu N == 0: Chuyển sang CPU an toàn.

    Args:
        model: PyTorch nn.Module cần huấn luyện.
        batch_size_per_gpu: Kích thước batch dự kiến trên MỖI GPU.
        base_lr: Tốc độ học cơ sở (dành cho 1 GPU).
        device_arg: Tham số thiết bị truyền vào từ CLI ('cuda' hoặc 'cpu').
        scale_lr: Tự động scale learning rate theo số lượng GPU (Linear Scaling Rule).

    Returns:
        parallel_model: Mô hình đã được chuyển sang thiết bị và bọc DataParallel nếu N > 1.
        device: Thiết bị chính.
        num_gpus: Số lượng GPU thực tế được sử dụng.
        effective_batch_size: Tổng batch size của DataLoader.
        effective_lr: Tốc độ học hiệu dụng đã được scale.
    """
    primary_device, num_gpus, gpu_names = get_available_devices()

    # Nếu người dùng chỉ định rõ 'cpu' từ CLI thì ưu tiên CPU
    if device_arg.lower() == "cpu" or num_gpus == 0:
        device = torch.device("cpu")
        model = model.to(device)
        print("🖥️ [Hardware] Chạy trên CPU (Không phát hiện GPU hoặc được chỉ định thủ công).")
        return model, device, 0, batch_size_per_gpu, base_lr

    # Chuyển mô hình sang primary GPU trước
    model = model.to(primary_device)

    if num_gpus > 1:
        print("\n" + "=" * 78)
        print(f" 🔥 TỰ ĐỘNG PHÁT HIỆN HỆ THỐNG ĐA GPU (MULTI-GPU DETECTED: {num_gpus} GPUs)")
        print("=" * 78)
        for i, name in enumerate(gpu_names):
            mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"   ⚡ GPU [{i}]: {name} | VRAM: {mem_gb:.2f} GB")

        # Bọc DataParallel để phân phối batch đồng thời trên tất cả N GPU
        model = nn.DataParallel(model)

        effective_batch_size = batch_size_per_gpu * num_gpus
        effective_lr = base_lr * (num_gpus if scale_lr else 1.0)

        print(f"\n   ⚙️ [Auto-Config] Cấu hình song song dữ liệu (DataParallel):")
        print(f"      - Batch Size mỗi GPU  : {batch_size_per_gpu}")
        print(f"      - TỔNG BATCH SIZE     : {effective_batch_size} (Gấp {num_gpus} lần)")
        print(f"      - Tốc độ học (LR)     : {effective_lr:.2e} (Scale theo {num_gpus} GPUs)")
        print("=" * 78 + "\n")
    else:
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"⚡ [Hardware] Chạy trên đơn GPU: {gpu_names[0]} (VRAM: {mem_gb:.2f} GB)")
        effective_batch_size = batch_size_per_gpu
        effective_lr = base_lr

    return model, primary_device, num_gpus, effective_batch_size, effective_lr


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Lấy mô hình gốc nguyên bản (bỏ qua mọi lớp bọc DataParallel, DistributedDataParallel hoặc TorchDynamo).
    Đảm bảo việc truy cập thuộc tính tùy biến và lưu Checkpoint hoàn toàn sạch.
    """
    while hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dọn dẹp triệt để các tiền tố phát sinh khi huấn luyện đa GPU hoặc biên dịch:
      - Loại bỏ tiền tố 'module.' (DataParallel / DDP)
      - Loại bỏ tiền tố '_orig_mod.' (torch.compile)
      - Đảm bảo tính tương thích 100% khi nạp vào mô hình đơn GPU hoặc CPU.
    """
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = str(key)
        # Lặp loại bỏ các tiền tố lồng nhau nếu có
        changed = True
        while changed:
            changed = False
            if clean_key.startswith("module."):
                clean_key = clean_key[len("module."):]
                changed = True
            elif clean_key.startswith("_orig_mod."):
                clean_key = clean_key[len("_orig_mod."):]
                changed = True

        cleaned[clean_key] = value

    return cleaned


def smart_load_state_dict(
    model: nn.Module,
    state_dict: Dict[str, Any],
    strict: bool = True,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Cơ chế nạp trọng số thông minh (Smart State Dict Loader):
      - Tự động nhận diện cấu trúc khóa của `model` hiện tại (có hoặc không có `module.`).
      - Tự động khớp và điều chỉnh tiền tố tương ứng giữa `state_dict` và `model`.
      - Báo cáo chi tiết số lượng khóa khớp, thiếu (missing) hoặc thừa (unexpected).
      - Không bao giờ bị văng lỗi ngớ ngẩn do lệch tiền tố `module.`.

    Args:
        model: Mô hình PyTorch mục tiêu.
        state_dict: Dictionary trọng số cần nạp.
        strict: Yêu cầu khớp 100% (nếu False sẽ bỏ qua khóa không khớp).
        verbose: In thông báo chi tiết quá trình nạp.

    Returns:
        missing_keys: Danh sách khóa bị thiếu trong state_dict.
        unexpected_keys: Danh sách khóa dư thừa trong state_dict.
    """
    # 1. Dọn dẹp sạch tiền tố từ file checkpoint
    clean_dict = clean_state_dict(state_dict)

    # 2. Kiểm tra xem model mục tiêu có đang bọc DataParallel không
    model_is_parallel = hasattr(model, "module")
    model_keys = set(model.state_dict().keys())

    # Nếu model mục tiêu có 'module.' ở đầu mà clean_dict không có -> Thêm lại
    target_dict = {}
    first_model_key = next(iter(model_keys)) if model_keys else ""
    needs_module_prefix = first_model_key.startswith("module.")

    for k, v in clean_dict.items():
        new_k = f"module.{k}" if (needs_module_prefix and not k.startswith("module.")) else k
        target_dict[new_k] = v

    # 3. Tiến hành nạp trọng số
    try:
        incompatible = model.load_state_dict(target_dict, strict=strict)
        missing_keys = incompatible.missing_keys
        unexpected_keys = incompatible.unexpected_keys
    except RuntimeError as e:
        if strict:
            if verbose:
                print(f"⚠️ [SmartLoader] Strict loading warning: {e}. Thử fallback sang non-strict...")
            incompatible = model.load_state_dict(target_dict, strict=False)
            missing_keys = incompatible.missing_keys
            unexpected_keys = incompatible.unexpected_keys
        else:
            raise e

    if verbose:
        n_loaded = len(target_dict) - len(unexpected_keys)
        print(f"✅ [SmartLoader] Nạp thành công {n_loaded} tham số vào mô hình.")
        if missing_keys:
            print(f"   ⚠️ Thiếu {len(missing_keys)} tham số (missing keys): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"   ⚠️ Thừa {len(unexpected_keys)} tham số (unexpected keys): {unexpected_keys[:5]}...")

    return missing_keys, unexpected_keys


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra_dict: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
):
    """
    Lưu Checkpoint an toàn tuyệt đối chuẩn Production:
      - Tự động unwrap model để gỡ bỏ lớp bọc Multi-GPU (`DataParallel`).
      - Dọn sạch 100% tiền tố `module.` khỏi `state_dict`.
      - Hỗ trợ lưu đầy đủ toàn bộ trạng thái huấn luyện: Model, Optimizer, LR Scheduler, GradScaler, Epoch, Metrics.
      - Đảm bảo file `.pth` sinh ra có thể được load ở MỌI MÔI TRƯỜNG (Multi-GPU, Single-GPU, CPU).
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # 1. Trích xuất state_dict nguyên bản từ mô hình chưa bị bọc
    raw_model = unwrap_model(model)
    cleaned_state = clean_state_dict(raw_model.state_dict())

    # Chuẩn hóa metrics thành kiểu dữ liệu nguyên bản Python (tránh lỗi NumPy scalar trên PyTorch 2.6+)
    clean_metrics = {}
    if metrics:
        for mk, mv in metrics.items():
            if hasattr(mv, "item"):
                clean_metrics[mk] = mv.item()
            elif isinstance(mv, (float, int, str, bool)):
                clean_metrics[mk] = mv
            else:
                try:
                    clean_metrics[mk] = float(mv)
                except Exception:
                    clean_metrics[mk] = str(mv)

    checkpoint: Dict[str, Any] = {
        "model_state": cleaned_state,
        "epoch": epoch,
        "metrics": clean_metrics,
    }

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None and hasattr(scaler, "state_dict"):
        checkpoint["scaler"] = scaler.state_dict()

    if extra_dict is not None:
        checkpoint.update(extra_dict)

    # 2. Ghi ra đĩa
    torch.save(checkpoint, save_path)
    if verbose:
        print(f"💾 [Checkpoint] Đã lưu mô hình chuẩn sạch (Zero 'module.' prefix) tại: {save_path}")


# Bí danh tương thích ngược
save_clean_checkpoint = save_checkpoint


def load_checkpoint(
    load_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Nạp Checkpoint thông minh và an toàn từ đĩa:
      - Deserialization luôn thực hiện trên CPU (`map_location="cpu"`) để chống tràn bộ nhớ GPU (OOM).
      - Tự động nhận diện các định dạng lưu trữ phổ biến:
          + Dict chứa key 'model_state', 'state_dict', 'student_state'...
          + Hay là state_dict thuần túy.
      - Tự động đồng bộ optimizer state, scheduler state, scaler state sang thiết bị đích nếu có yêu cầu.
      - Trả về dictionary checkpoint gốc để khôi phục 'epoch', 'metrics', 'extra_dict'.
    """
    if not os.path.isfile(load_path):
        if os.path.isdir(load_path):
            candidates = [
                os.path.join(load_path, "best_decomposition_model.pth"),
                os.path.join(load_path, "best_checkpoint.pth"),
                os.path.join(load_path, "last_checkpoint.pth"),
                os.path.join(load_path, "model.pth"),
            ]
            found = False
            for c in candidates:
                if os.path.isfile(c):
                    load_path = c
                    found = True
                    break
            if not found:
                pths = [os.path.join(load_path, f) for f in os.listdir(load_path) if f.endswith(".pth") and os.path.isfile(os.path.join(load_path, f))]
                if pths:
                    load_path = pths[0]
                    found = True
            if not found:
                raise FileNotFoundError(f"Không tìm thấy file checkpoint (.pth) hợp lệ trong thư mục: {load_path}")
        else:
            raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {load_path}")

    if verbose:
        print(f"📂 [Checkpoint] Đang nạp checkpoint từ: {load_path}...")

    # Load sang CPU trước để an toàn bộ nhớ (tương thích PyTorch 2.6+ weights_only)
    try:
        raw_data = torch.load(load_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw_data = torch.load(load_path, map_location="cpu")

    # 1. Trích xuất state_dict của mô hình
    state_to_load = None
    candidate_keys = ["model_state", "state_dict", "student_state", "model", "student"]

    if isinstance(raw_data, dict):
        for ck in candidate_keys:
            if ck in raw_data and isinstance(raw_data[ck], dict):
                state_to_load = raw_data[ck]
                break

    if state_to_load is None:
        # Giả định chính raw_data là state_dict thuần túy
        state_to_load = raw_data

    # 2. Nạp vào mô hình bằng SmartLoader
    smart_load_state_dict(model=model, state_dict=state_to_load, strict=strict, verbose=verbose)

    # 3. Nạp optimizer nếu có và chuyển state sang device
    if optimizer is not None and isinstance(raw_data, dict):
        opt_key = next((k for k in ["optimizer", "optimizer_state", "opt_state"] if k in raw_data), None)
        if opt_key:
            try:
                optimizer.load_state_dict(raw_data[opt_key])
                # Chuyển các tensor trạng thái của optimizer sang device
                dest_device = torch.device(device)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(dest_device)
                if verbose:
                    print("   ✅ [Optimizer] Khôi phục thành công toàn bộ trạng thái optimizer.")
            except Exception as e_opt:
                if verbose:
                    print(f"   ⚠️ [Optimizer Notice] Không thể khôi phục optimizer state: {e_opt}")

    # 4. Nạp scheduler nếu có
    if scheduler is not None and isinstance(raw_data, dict):
        sch_key = next((k for k in ["scheduler", "scheduler_state", "sched_state"] if k in raw_data), None)
        if sch_key:
            try:
                scheduler.load_state_dict(raw_data[sch_key])
                if verbose:
                    print("   ✅ [Scheduler] Khôi phục thành công trạng thái LR scheduler.")
            except Exception as e_sch:
                if verbose:
                    print(f"   ⚠️ [Scheduler Notice] Không thể khôi phục scheduler: {e_sch}")

    # 5. Nạp scaler nếu có
    if scaler is not None and isinstance(raw_data, dict):
        scaler_key = next((k for k in ["scaler", "scaler_state"] if k in raw_data), None)
        if scaler_key and hasattr(scaler, "load_state_dict"):
            try:
                scaler.load_state_dict(raw_data[scaler_key])
                if verbose:
                    print("   ✅ [AMP Scaler] Khôi phục thành công trạng thái GradScaler.")
            except Exception as e_sc:
                if verbose:
                    print(f"   ⚠️ [Scaler Notice] Không thể khôi phục GradScaler: {e_sc}")

    # Đưa model về đúng device
    model.to(device)

    return raw_data if isinstance(raw_data, dict) else {"state_dict": raw_data}
