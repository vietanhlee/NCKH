"""
=============================================================================
 Common Utility: BackboneLoader
 Module tải và chuẩn hóa giao tiếp (Unified Interface) cho Vision Backbones
 Hỗ trợ DINOv3, DINOv2, ConvNeXt và Vision Transformer với Spatial Patch Tokens
=============================================================================
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo UTF-8 an toàn trên Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Bỏ qua các cảnh báo phụ thuộc tùy chọn (xFormers) của DINOv2 khi chạy Native PyTorch
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*dinov2.*")


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Token mặc định (tránh hardcode token hết hạn, ưu tiên biến môi trường hoặc .env)
DEFAULT_HF_TOKEN = ""


def load_env_credentials(verbose: bool = False) -> str:
    """
    Tự động tìm kiếm và nạp các biến môi trường từ file .env.
    Đồng bộ hóa HF_TOKEN và đăng nhập tự động vào Hugging Face Hub nếu có token hợp lệ.
    """
    candidate_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
    ]

    try:
        import dotenv
        for p in candidate_paths:
            if os.path.isfile(p):
                dotenv.load_dotenv(p, override=True)
                if verbose:
                    print(f"[*] [Credentials] Đã nạp cấu hình môi trường từ: {p}")
                break
    except ImportError:
        pass

    # Fallback đọc thủ công nếu thiếu thư viện dotenv
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if not token:
        for p in candidate_paths:
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                k, v = k.strip(), v.strip().strip("'\"")
                                if k in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] and v:
                                    token = v
                                    os.environ[k] = v
                except Exception:
                    pass

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    return token


# Tự động nạp credentials ngay khi module được import
load_env_credentials()


def resolve_dino_weights(
    model_name: str = "dinov3_vits16",
    weights_path: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """
    Tự động phân giải và tìm kiếm file trọng số mô hình DINO theo thứ tự ưu tiên:
      1. Đường dẫn weights_path người dùng chỉ định trực tiếp (nếu file tồn tại).
      2. Tự động quét toàn bộ Local Cache trên hệ thống (Hugging Face Hub snapshots,
         PyTorch Hub checkpoints, thư mục checkpoints nội bộ project, ổ đĩa, v.v.).
      3. Thử nạp từ Hugging Face Hub ở chế độ offline (local_files_only=True).
      4. Tải từ Hugging Face Hub trực tuyến (nếu có token hợp lệ còn hạn).

    Returns:
        Đường dẫn file trọng số (.safetensors, .pth, .bin) hoặc None nếu cần fallback Warm-Start.
    """
    import glob

    # 1. Đường dẫn chỉ định trực tiếp
    if weights_path and os.path.isfile(weights_path):
        return weights_path

    # Hỗ trợ URL Hugging Face spec: hf://repo_id:filename (hoặc định dạng repo_id/filename nếu không phải đường dẫn local)
    is_hf_spec = False
    if weights_path:
        norm_wp = weights_path.replace("\\", "/")
        if norm_wp.startswith("hf://"):
            is_hf_spec = True
        elif (
            not os.path.exists(weights_path)
            and ":" not in norm_wp
            and norm_wp.count("/") == 1
            and not norm_wp.lower().endswith((".safetensors", ".pth", ".bin", ".pt", ".ckpt"))
        ):
            # Dạng org/repo_name hợp lệ (ví dụ: facebook/dinov3-vits16-pretrain-lvd1689m)
            is_hf_spec = True

    if is_hf_spec:
        try:
            from huggingface_hub import hf_hub_download
            hf_spec = weights_path.replace("hf://", "")
            if ":" in hf_spec:
                repo_id, filename = hf_spec.split(":", 1)
            else:
                repo_id = hf_spec
                filename = "model.safetensors"
            print(f"📥 [HuggingFace Hub] Đang tải trọng số từ {repo_id}/{filename}...")
            token_for_dl = hf_token or load_env_credentials() or os.environ.get("HF_TOKEN") or None
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename, token=token_for_dl)
            if downloaded and os.path.isfile(downloaded):
                return downloaded
        except Exception as e_hf_dl:
            print(f"   ℹ️ [HuggingFace Hub] Tải weights từ HuggingFace notice: {e_hf_dl}")

    # 2. Tự động quét Local Cache toàn diện
    clean_name = model_name.lower().strip()
    arch_tag = clean_name.replace('_', '-')
    search_patterns = [
        # Hugging Face Hub cache của user hiện tại
        os.path.expanduser(f"~/.cache/huggingface/hub/models--facebook--{arch_tag}*/snapshots/*/model.safetensors"),
        os.path.expanduser(f"~/.cache/huggingface/hub/models--facebook--{arch_tag}*/**/*.safetensors"),
        os.path.expanduser(f"~/.cache/huggingface/hub/models--facebook--{arch_tag}*/**/*.bin"),
        os.path.expanduser(f"~/.cache/huggingface/hub/models--facebook--{arch_tag}*/**/*.pth"),
        # Quét các profile người dùng khác trên Windows (Admin, levie,...)
        f"C:/Users/*/.cache/huggingface/hub/models--facebook--{arch_tag}*/snapshots/*/model.safetensors",
        f"C:/Users/*/.cache/huggingface/hub/models--facebook--{arch_tag}*/**/*.safetensors",
        f"C:/Users/*/.cache/huggingface/hub/models--facebook--{arch_tag}*/**/*.bin",
        f"C:/Users/*/.cache/huggingface/hub/models--facebook--{arch_tag}*/**/*.pth",
        # PyTorch Hub checkpoints
        os.path.expanduser(f"~/.cache/torch/hub/checkpoints/*{clean_name}*"),
        f"C:/Users/*/.cache/torch/hub/checkpoints/*{clean_name}*",
        # Project checkpoints folder
        os.path.join(os.getcwd(), "checkpoints", f"*{clean_name}*"),
        os.path.join(os.getcwd(), "checkpoints", f"*{arch_tag}*"),
        os.path.join(os.getcwd(), f"*{clean_name}*"),
        os.path.join(os.getcwd(), f"*{arch_tag}*"),
        # Kaggle input
        f"/kaggle/input/**/{clean_name}*",
        f"/kaggle/input/**/{arch_tag}*",
    ]

    for pat in search_patterns:
        try:
            matches = glob.glob(pat, recursive=True)
            for m in matches:
                if os.path.isfile(m) and os.path.getsize(m) > 10 * 1024 * 1024:  # Phải là file model thực sự (>10MB)
                    print(f"   🎯 [Model Loader] Tự động phát hiện trọng số DINOv3 từ local cache: {m}")
                    return m
        except Exception:
            pass

    # 3. Thử Hugging Face Hub chế độ offline (local_files_only=True)
    repo_candidates = [
        f"facebook/{arch_tag}-pretrain-lvd1689m",
        f"facebook/{arch_tag}",
        f"facebook/dinov3-{arch_tag.split('-')[-1]}" if '-' in arch_tag else f"facebook/dinov3"
    ]
    file_candidates = ["model.safetensors", "pytorch_model.bin", f"{clean_name}_pretrain_lvd1689m.pth"]

    try:
        from huggingface_hub import hf_hub_download
        for r_id in repo_candidates:
            for fname in file_candidates:
                try:
                    f = hf_hub_download(repo_id=r_id, filename=fname, local_files_only=True)
                    if f and os.path.isfile(f) and os.path.getsize(f) > 10 * 1024 * 1024:
                        print(f"   🎯 [HuggingFace Hub] Tận dụng file offline từ snapshot cache: {f}")
                        return f
                except Exception:
                    pass
    except Exception:
        pass

    # 4. Thử tải online nếu có HF_TOKEN hợp lệ
    token_to_use = hf_token or os.environ.get("HF_TOKEN") or ""
    if token_to_use and not token_to_use.startswith("#") and len(token_to_use) > 10:
        try:
            from huggingface_hub import hf_hub_download
            for r_id in repo_candidates:
                for fname in file_candidates:
                    try:
                        f = hf_hub_download(repo_id=r_id, filename=fname, token=token_to_use)
                        if f and os.path.isfile(f) and os.path.getsize(f) > 10 * 1024 * 1024:
                            print(f"   ✅ [HuggingFace Hub] Tải thành công weights DINOv3 từ '{r_id}': {f}")
                            return f
                    except Exception as e_dl:
                        err_str = str(e_dl).lower()
                        if "expired" in err_str or "401" in err_str or "invalid" in err_str:
                            print(f"   ℹ️ [HuggingFace Hub] Token HF cung cấp đã hết hạn hoặc không có quyền truy cập repo gated.")
                            return None
                        continue
        except Exception:
            pass

    return None


def clean_corrupted_hub_checkpoints(keyword: str = "dinov2", verbose: bool = True) -> int:
    """
    Quét và tự động dọn dẹp các file checkpoint bị lỗi tải ngắt quãng (Corrupted Zip/miniz Error)
    trong thư mục PyTorch Hub Cache (~/.cache/torch/hub/checkpoints/), ngăn ngừa 100% lỗi miniz / PytorchStreamReader.
    """
    import zipfile
    import glob

    candidate_dirs = [
        os.path.join(torch.hub.get_dir(), "checkpoints"),
        os.path.expanduser("~/.cache/torch/hub/checkpoints"),
        "/root/.cache/torch/hub/checkpoints",
    ]
    cleaned_count = 0
    checked_files = set()

    for c_dir in candidate_dirs:
        if not os.path.isdir(c_dir):
            continue
        patterns = [
            os.path.join(c_dir, f"*{keyword}*"),
            os.path.join(c_dir, "*.pth"),
        ]
        for pat in patterns:
            for p in glob.glob(pat):
                if p in checked_files or not os.path.isfile(p):
                    continue
                checked_files.add(p)
                # Chỉ kiểm tra các file liên quan đến keyword
                if keyword and keyword.lower() not in os.path.basename(p).lower():
                    continue

                is_corrupt = False
                file_size = os.path.getsize(p)
                # DINOv2 ViT-Small chuẩn là ~86MB, ViT-Base là ~340MB. Nếu < 15MB thì chắc chắn file tải dở dang
                if file_size < 15 * 1024 * 1024:
                    is_corrupt = True
                else:
                    # Kiểm tra xem có phải zip hợp lệ không
                    try:
                        with zipfile.ZipFile(p, "r") as zf:
                            if zf.testzip() is not None:
                                is_corrupt = True
                    except Exception:
                        is_corrupt = True

                if is_corrupt:
                    try:
                        os.remove(p)
                        cleaned_count += 1
                        if verbose:
                            print(f"🧹 [Auto-Recovery] Đã xóa file checkpoint cache bị hỏng: {p} ({file_size / (1024*1024):.2f} MB)")
                    except Exception as e_del:
                        if verbose:
                            print(f"⚠️ [Auto-Recovery] Không thể xóa {p}: {e_del}")

    return cleaned_count


def safe_load_dinov2_from_hub(
    hub_name: str = "dinov2_vits14",
    pretrained: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """
    Tải Foundation Model DINOv2 an toàn tuyệt đối từ PyTorch Hub với cơ chế Tự Phục Hồi (Self-Healing).
    Tự động:
      1. Quét và xóa file checkpoint bị hỏng trước khi load.
      2. Nếu gặp lỗi PytorchStreamReader / miniz error (do rớt mạng khi tải), tự động xóa cache và tải lại.
      3. Nếu Meta CDN tiếp tục bị lỗi mạng, tự động fallback sang HuggingFace Mirror / timm.
    """
    # 1. Dọn dẹp cache hỏng trước
    clean_corrupted_hub_checkpoints(keyword="dinov2", verbose=verbose)

    # 2. Thử tải qua PyTorch Hub
    try:
        if verbose:
            print(f"📦 [PyTorch Hub] Đang tải mô hình facebookresearch/dinov2 '{hub_name}'...")
        return torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=pretrained)
    except RuntimeError as e_rt:
        err_msg = str(e_rt).lower()
        if any(w in err_msg for w in ["pytorchstreamreader", "zip archive", "central directory", "miniz", "corrupted"]):
            print(f"\n⚠️ [Self-Healing] Phát hiện checkpoint DINOv2 bị hỏng do rớt mạng khi tải:")
            print(f"   Chi tiết: {e_rt}")
            # Xóa triệt để file hỏng
            clean_corrupted_hub_checkpoints(keyword="", verbose=True)
            print(f"🔄 [Self-Healing] Đang tự động thử tải lại từ Meta PyTorch Hub...")
            try:
                return torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=pretrained, force_reload=True)
            except Exception as e_retry:
                print(f"⚠️ [Self-Healing] Tải lại từ PyTorch Hub không thành công: {e_retry}.")
        else:
            print(f"⚠️ [PyTorch Hub Notice] {e_rt}")
    except Exception as e_other:
        print(f"⚠️ [PyTorch Hub Notice] {e_other}")

    # 3. Fallback sang timm (HuggingFace CDN Mirror cực kỳ ổn định)
    if verbose:
        print(f"💡 [Fallback Loader] Đang nạp DINOv2 qua timm HuggingFace CDN mirror...")
    try:
        import timm
        timm_arch = "vit_small_patch14_dinov2" if "s" in hub_name else "vit_base_patch14_dinov2"
        model = timm.create_model(timm_arch, pretrained=pretrained, num_classes=0)
        # Bổ sung thuộc tính get_intermediate_layers nếu chưa có để tương thích API DINOv2
        if not hasattr(model, "get_intermediate_layers"):
            def _timm_get_intermediate(x, n=1, return_class_token=True):
                feat = model.forward_features(x)
                cls_tok = feat[:, 0]
                patch_toks = feat[:, 1:]
                if return_class_token:
                    return [(patch_toks, cls_tok)]
                return [patch_toks]
            model.get_intermediate_layers = _timm_get_intermediate
        if verbose:
            print(f"✅ [Fallback Loader] Nạp thành công '{timm_arch}' từ timm.")
        return model
    except Exception as e_timm:
        if verbose:
            print(f"❌ [Fallback Loader] Fallback timm thất bại: {e_timm}")
        raise RuntimeError(
            f"Không thể tải DINOv2 '{hub_name}' từ cả PyTorch Hub lẫn timm mirror.\n"
            f"Gợi ý khắc phục thủ công trên terminal máy chủ:\n"
            f"  rm -rf /root/.cache/torch/hub/checkpoints/*dinov2*\n"
        )


def smart_load_backbone_weights(
    backbone: nn.Module,
    weights_path: str,
    verbose: bool = True,
) -> int:
    """
    Nạp trọng số cho Backbone an toàn tuyệt đối từ file checkpoint (.pth, .pt, .bin, .safetensors).
    Tự động:
      1. Đọc an toàn không crash trên mọi phiên bản PyTorch (PyTorch 2.6+ weights_only fallback).
      2. Đọc file .safetensors qua thư viện safetensors nếu có.
      3. Trích xuất state_dict con nếu checkpoint lưu dưới dạng dictionary phức hợp.
      4. Bóc tách tiền tố lồng nhau: 'module.', 'student.', 'teacher.', 'model.', 'base_model.', 'encoder.', 'backbone.', '0.'
      5. So khớp shape nghiêm ngặt: chỉ nạp tensor có kích thước khớp 100% với backbone mục tiêu.
      6. Báo cáo tỷ lệ nạp tham số rõ ràng.
    """
    if not os.path.isfile(weights_path):
        if verbose:
            print(f"⚠️ [BackboneLoader] File trọng số không tồn tại: {weights_path}")
        return 0

    # 1. Đọc file
    state = None
    if str(weights_path).endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state = load_file(weights_path, device="cpu")
        except Exception as e_st:
            if verbose:
                print(f"⚠️ [BackboneLoader] Safetensors notice: {e_st}. Thử nạp qua PyTorch...")

    if state is None:
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(weights_path, map_location="cpu")
        except Exception as e_load:
            if verbose:
                print(f"❌ [BackboneLoader] Lỗi đọc checkpoint: {e_load}")
            return 0

    # 2. Trích xuất state_dict con nếu có
    if isinstance(state, dict):
        for candidate in ["model_state", "model", "student_state", "student", "teacher", "state_dict", "backbone"]:
            if candidate in state and isinstance(state[candidate], dict):
                state = state[candidate]
                break

    if not isinstance(state, dict):
        if verbose:
            print(f"⚠️ [BackboneLoader] Định dạng checkpoint không hợp lệ (không phải dictionary).")
        return 0

    # 3. Bóc tách tiền tố và so khớp tham số với backbone mục tiêu
    target_state = backbone.state_dict()
    target_keys = set(target_state.keys())
    matched_state = {}
    prefixes = [
        "module.", "student.", "teacher.", "model.", "base_model.",
        "encoder.", "backbone.", "student_model.", "teacher_model.", "0."
    ]

    for k, v in state.items():
        clean_k = str(k)
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if clean_k.startswith(p):
                    clean_k = clean_k[len(p):]
                    changed = True

        if clean_k in target_keys:
            if target_state[clean_k].shape == v.shape:
                matched_state[clean_k] = v
            elif verbose:
                print(f"   ⚠️ [BackboneLoader] Lệch shape tensor '{clean_k}': Target {target_state[clean_k].shape} vs Ckpt {v.shape}")
        elif k in target_keys and target_state[k].shape == v.shape:
            matched_state[k] = v

    if matched_state:
        backbone.load_state_dict(matched_state, strict=False)
        if verbose:
            print(f"✅ [BackboneLoader] Nạp thành công {len(matched_state)}/{len(target_keys)} tham số backbone từ: {weights_path}")
    else:
        if verbose:
            print(f"⚠️ [BackboneLoader] Cảnh báo: Không có tham số nào khớp với backbone mục tiêu từ: {weights_path}")

    return len(matched_state)


def get_dino_backbone(
    model_name: str = "dinov3_vits16",
    pretrained: bool = True,
    weights_path: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    hf_token: Optional[str] = None,
    **kwargs,
) -> Tuple[nn.Module, int, int]:
    """
    Tải Foundation Vision Backbone chuẩn Meta AI với cơ chế nạp trọng số linh hoạt.

    Args:
        model_name: Tên backbone ('dinov3_vits16', 'dinov3_vitb16', 'dinov2_vits14', 'dinov2_vitb14', ...).
        pretrained: Nạp trọng số tiền huấn luyện (True/False).
        weights_path: Đường dẫn checkpoint local (.pth, .safetensors) hoặc HuggingFace repo/file.
        device: Thiết bị tính toán ('cuda', 'cpu').
        hf_token: HuggingFace access token (nếu repo cần xác thực).
        **kwargs: Tham số mở rộng (backbone_name, token, ...).

    Returns:
        backbone: Mô hình PyTorch nn.Module.
        embed_dim: Số chiều đặc trưng embedding (ví dụ 384 cho ViT-S, 768 cho ViT-B).
        patch_size: Kích thước patch (16 cho DINOv3, 14 cho DINOv2).
    """
    # Xử lý các alias tương thích ngược
    if "backbone_name" in kwargs and kwargs["backbone_name"] is not None:
        model_name = kwargs["backbone_name"]
    if "token" in kwargs and kwargs["token"] is not None and not hf_token:
        hf_token = kwargs["token"]

    device = torch.device(device)
    patch_size = 16 if ("16" in model_name.lower() or "dinov3" in model_name.lower()) else 14

    # 0. Nạp biến môi trường và giải quyết trọng số qua resolve_dino_weights
    token_to_use = hf_token or load_env_credentials() or os.environ.get("HF_TOKEN") or ""
    if token_to_use:
        os.environ["HF_TOKEN"] = token_to_use
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token_to_use

    if pretrained and (weights_path is None or not os.path.exists(weights_path)):
        resolved_weights = resolve_dino_weights(
            model_name=model_name,
            weights_path=weights_path,
            hf_token=token_to_use,
        )
        if resolved_weights and os.path.isfile(resolved_weights):
            weights_path = resolved_weights

    # 1. Thử tận dụng hàm build_backbone đã viết rất hoàn chỉnh trong train_ssl_dinov3
    dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dino_dir not in sys.path:
        sys.path.insert(0, dino_dir)

    try:
        from train_ssl_dinov3 import build_backbone as base_builder
        backbone, embed_dim = base_builder(
            model_name=model_name,
            pretrained=pretrained,
            weights_path=weights_path,
            hf_token=token_to_use,
            **kwargs,
        )
        backbone = backbone.to(device)
        return backbone, embed_dim, patch_size
    except Exception as e_base:
        if "dinov3" in model_name.lower():
            # Yêu cầu hệ thống: Chỉ dùng chuẩn DINOv3, tuyệt đối KHÔNG fallback sang DINOv2
            raise RuntimeError(
                f"\n❌ [LỖI DINOv3] Không thể nạp mô hình DINOv3 '{model_name}': {e_base}\n"
                f"   Quy tắc hệ thống: Chỉ dùng chuẩn DINOv3, tuyệt đối KHÔNG fallback sang DINOv2.\n"
                f"   👉 Giải pháp:\n"
                f"      - Chỉ định rõ file weights DINOv3 qua đối số: --weights <đường_dẫn_weights>\n"
                f"      - Hoặc tiếp tục huấn luyện từ checkpoint qua đối số: --resume <đường_dẫn_checkpoint.pth>\n"
            ) from e_base
        print(f"⚠️ [BackboneLoader] Base builder notice: {e_base}. Chuyển sang fallback loader.")

    # 2. Fallback trực tiếp qua torch.hub (Chỉ dành cho DINOv2 khi được yêu cầu tường minh)
    clean_name = model_name.lower().strip()
    if "dinov2" in clean_name:
        hub_tag = "dinov2_vits14" if "s" in clean_name else "dinov2_vitb14"
        embed_dim = 384 if "s" in clean_name else 768
        patch_size = 14
        backbone = safe_load_dinov2_from_hub(hub_name=hub_tag, pretrained=pretrained)
    elif "dinov3" in clean_name:
        raise RuntimeError(
            f"❌ [LỖI DINOv3] Không thể khởi tạo DINOv3 '{model_name}'. Tuyệt đối KHÔNG fallback sang DINOv2 theo yêu cầu."
        )
    else:
        # Fallback timm ViT-Small
        import timm
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = getattr(backbone, "num_features", 384)
        patch_size = 16

    if weights_path and os.path.isfile(weights_path):
        smart_load_backbone_weights(backbone=backbone, weights_path=weights_path, verbose=True)

    backbone = backbone.to(device)
    return backbone, embed_dim, patch_size


# Thống kê chuẩn hóa ImageNet — DINOv2/DINOv3 (LVD-1689M) đều được tiền huấn luyện với chuẩn này.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Chuẩn hóa ảnh dải [0, 1] theo thống kê ImageNet trước khi đưa vào ViT DINO.
    Bắt buộc: nếu đưa ảnh [0, 1] thô vào DINOv3, phân phối đầu vào lệch khỏi phân phối
    tiền huấn luyện → đặc trưng patch bị nhiễu, PCA feature map "rỗ" và kém ngữ nghĩa.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _supports_mask_tokens(backbone: nn.Module) -> bool:
    """Kiểm tra backbone có hỗ trợ `forward_features(x, masks=...)` (mask token kiểu iBOT) hay không."""
    cached = getattr(backbone, "_agy_supports_masks", None)
    if cached is not None:
        return cached
    supported = False
    fwd = getattr(backbone, "forward_features", None)
    if fwd is not None:
        try:
            import inspect
            supported = "masks" in inspect.signature(fwd).parameters
        except (TypeError, ValueError):
            supported = False
    try:
        backbone._agy_supports_masks = supported
    except Exception:
        pass
    return supported


def _pixel_mask_fallback(x: torch.Tensor, masks: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Fallback khi backbone không hỗ trợ mask token: che trực tiếp các patch trên ảnh
    (gán giá trị 0 — tức giá trị trung bình sau chuẩn hóa ImageNet).
    masks: (B, N_patches) bool, True = patch bị che.
    """
    B, _, H, W = x.shape
    hp, wp = H // patch_size, W // patch_size
    m = masks.view(B, 1, hp, wp).float()
    m_up = F.interpolate(m, size=(H, W), mode="nearest")
    return x * (1.0 - m_up)


def extract_tokens_with_grad(
    backbone: nn.Module,
    x: torch.Tensor,
    patch_size: int = 16,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trích xuất [CLS] token và Patch Tokens không gian, GIỮ NGUYÊN đồ thị gradient.
    Dùng cho mọi nhánh cần huấn luyện backbone (Student SSL, fine-tune decoder có mở khóa backbone).

    Args:
        backbone: ViT backbone (DINOv3/DINOv2).
        x: Batch tensor ảnh đầu vào ĐÃ chuẩn hóa ImageNet (B, C, H, W).
        patch_size: Kích thước patch tương ứng của mô hình.
        masks: (B, N_patches) bool tùy chọn — patch bị che sẽ được thay bằng mask token học được
               (iBOT/DINOv2/DINOv3). Nếu backbone không hỗ trợ, fallback che ở mức pixel.

    Returns:
        cls_token: (B, embed_dim)
        patch_tokens: (B, H_patches, W_patches, embed_dim)
    """
    B, C, H, W = x.shape
    if H % patch_size != 0 or W % patch_size != 0:
        target_h = max(patch_size, (H // patch_size) * patch_size)
        target_w = max(patch_size, (W // patch_size) * patch_size)
        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        H, W = target_h, target_w

    h_patches = H // patch_size
    w_patches = W // patch_size

    # Nhánh có mask token (iBOT-style masked image modeling)
    if masks is not None and masks.any():
        masks = masks.to(device=x.device, dtype=torch.bool).view(B, h_patches * w_patches)
        if _supports_mask_tokens(backbone):
            feat = backbone.forward_features(x, masks=masks)
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            if isinstance(feat, dict):
                cls_token = feat["x_norm_clstoken"]
                patch_raw = feat["x_norm_patchtokens"]
                patch_raw = patch_raw[:, : h_patches * w_patches, :]
                return cls_token, patch_raw.reshape(B, h_patches, w_patches, patch_raw.shape[-1])
        # Fallback: che mức pixel rồi đi tiếp luồng chuẩn bên dưới
        x = _pixel_mask_fallback(x, masks, patch_size)

    # DINOv2 / DINOv3 API
    if hasattr(backbone, "get_intermediate_layers"):
        # outputs là list of tensors
        outputs = backbone.get_intermediate_layers(x, n=1, return_class_token=True)
        if isinstance(outputs[0], tuple):
            patch_raw, cls_token = outputs[0]
        else:
            patch_raw = outputs[0]
            cls_token = patch_raw[:, 0]
            patch_raw = patch_raw[:, 1:]
    elif hasattr(backbone, "forward_features"):
        feat = backbone.forward_features(x)
        if isinstance(feat, dict):
            cls_token = feat.get("x_norm_clstoken", None)
            patch_raw = feat.get("x_norm_patchtokens", None)
            if patch_raw is None:
                patch_raw = feat.get("x_prenorm", None)
        else:
            cls_token = feat[:, 0]
            patch_raw = feat[:, 1:]
    else:
        # Standard forward
        cls_token = backbone(x)
        # Tạo dummy spatial patch tokens từ CLS nếu model không trả về patch
        patch_raw = cls_token.unsqueeze(1).expand(-1, h_patches * w_patches, -1)

    # Đảm bảo số lượng patch khớp
    n_expected = h_patches * w_patches
    if patch_raw.shape[1] > n_expected:
        patch_raw = patch_raw[:, :n_expected, :]
    elif patch_raw.shape[1] < n_expected:
        # Pad nếu thiếu
        diff = n_expected - patch_raw.shape[1]
        pad = patch_raw[:, -1:, :].expand(-1, diff, -1)
        patch_raw = torch.cat([patch_raw, pad], dim=1)

    embed_dim = patch_raw.shape[-1]
    patch_spatial = patch_raw.reshape(B, h_patches, w_patches, embed_dim)

    return cls_token, patch_spatial


@torch.no_grad()
def extract_tokens(
    backbone: nn.Module,
    x: torch.Tensor,
    patch_size: int = 16,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phiên bản KHÔNG gradient của `extract_tokens_with_grad` (tương thích ngược API cũ).
    Chỉ dùng cho nhánh đóng băng: FrozenExtractor (TAM), Teacher EMA, trích xuất đặc trưng suy luận.
    TUYỆT ĐỐI không dùng cho nhánh Student/backbone cần huấn luyện.
    """
    return extract_tokens_with_grad(backbone, x, patch_size=patch_size, masks=masks)
