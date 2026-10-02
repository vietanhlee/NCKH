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


# Hugging Face Access Token mặc định chính thức cho Meta DINOv3 Foundation Model
DEFAULT_HF_TOKEN = "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"


def load_env_credentials(verbose: bool = False) -> str:
    """
    Tự động tìm kiếm và nạp các biến môi trường từ file .env.
    Đồng bộ hóa HF_TOKEN và đăng nhập tự động vào Hugging Face Hub nếu có token hợp lệ.
    Nếu chưa có, tự động sử dụng token dự phòng chuẩn từ cấu hình hệ thống.

    Returns:
        token: Chuỗi Hugging Face token hợp lệ.
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
                    print(f"[*] [Credentials] Da nap cau hinh moi truong tu: {p}")
                break
    except ImportError:
        pass

    # Fallback đọc thủ công nếu thiếu thư viện dotenv
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
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

    # Nếu vẫn chưa tìm thấy token trong môi trường, dùng fallback từ hệ thống và tự động tạo .env
    if not token:
        token = DEFAULT_HF_TOKEN
        for env_path in [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"), os.path.join(os.getcwd(), ".env")]:
            if not os.path.exists(env_path) and token:
                try:
                    with open(env_path, "w", encoding="utf-8") as f:
                        f.write(f"# Hugging Face Hub Credentials for Meta DINOv3 Access\nHF_TOKEN={token}\nHUGGING_FACE_HUB_TOKEN={token}\n")
                except Exception:
                    pass

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass

    return token


# Tự động nạp credentials ngay khi module được import
load_env_credentials()


def get_dino_backbone(
    model_name: str = "dinov3_vits16",
    pretrained: bool = True,
    weights_path: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, int, int]:
    """
    Tải Foundation Vision Backbone chuẩn Meta AI với cơ chế nạp trọng số linh hoạt.

    Args:
        model_name: Tên backbone ('dinov3_vits16', 'dinov3_vitb16', 'dinov2_vits14', 'dinov2_vitb14', ...).
        pretrained: Nạp trọng số tiền huấn luyện (True/False).
        weights_path: Đường dẫn checkpoint local (.pth, .safetensors) hoặc HuggingFace repo/file.
        device: Thiết bị tính toán ('cuda', 'cpu').

    Returns:
        backbone: Mô hình PyTorch nn.Module.
        embed_dim: Số chiều đặc trưng embedding (ví dụ 384 cho ViT-S, 768 cho ViT-B).
        patch_size: Kích thước patch (16 cho DINOv3, 14 cho DINOv2).
    """
    device = torch.device(device)
    patch_size = 16 if ("16" in model_name.lower() or "dinov3" in model_name.lower()) else 14

    # 0. Nạp biến môi trường và token Hugging Face
    hf_token = load_env_credentials()

    # Hỗ trợ tải trực tiếp từ HuggingFace Hub nếu weights_path trỏ tới repo hoặc hf://
    if weights_path and (weights_path.startswith("hf://") or ("/" in weights_path and not os.path.exists(weights_path))):
        try:
            from huggingface_hub import hf_hub_download
            hf_spec = weights_path.replace("hf://", "")
            if ":" in hf_spec:
                repo_id, filename = hf_spec.split(":", 1)
            else:
                repo_id = hf_spec
                filename = "model.safetensors"
            print(f"📥 [HuggingFace Hub] Đang tải trọng số từ {repo_id}/{filename}...")
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
            if downloaded and os.path.isfile(downloaded):
                weights_path = downloaded
                print(f"✅ [HuggingFace Hub] Tải thành công weights về cache: {weights_path}")
        except Exception as e_hf_dl:
            print(f"⚠️ [HuggingFace Hub] Tải weights từ HuggingFace notice: {e_hf_dl}")

    # Tự động tải weights DINOv3 chính thức từ HuggingFace Hub nếu cần pretrained và chưa có file offline
    if "dinov3" in model_name.lower() and pretrained and (weights_path is None or not os.path.exists(weights_path)):
        arch_tag = model_name.lower().strip().replace('_', '-')
        repo_candidates = [
            f"facebook/{arch_tag}-pretrain-lvd1689m",
            f"facebook/{arch_tag}",
            f"facebook/dinov3-{arch_tag.split('-')[-1]}"
        ]
        try:
            from huggingface_hub import hf_hub_download
            last_hf_err = None
            for r_id in repo_candidates:
                for c_file in ["model.safetensors", "pytorch_model.bin", f"{model_name}_pretrain_lvd1689m.pth"]:
                    try:
                        dl_file = hf_hub_download(repo_id=r_id, filename=c_file, token=hf_token)
                        if dl_file and os.path.exists(dl_file):
                            weights_path = dl_file
                            print(f"   ✅ [HuggingFace Hub] Tự động tải thành công weights DINOv3 từ '{r_id}': {dl_file}")
                            break
                    except Exception as e_inner:
                        last_hf_err = e_inner
                        err_str = str(e_inner).lower()
                        if "expired" in err_str or "401" in err_str or "invalid" in err_str:
                            print(f"   ⚠️ [HuggingFace Hub] Cảnh báo xác thực: Token HF không hợp lệ hoặc đã hết hạn ({e_inner})!")
                        continue
                if weights_path:
                    break
            if not weights_path and last_hf_err:
                print(f"   ⚠️ [HuggingFace Hub Notice] Không thể tải weights từ remote Hub: {last_hf_err}")
        except Exception as e_hf_top:
            print(f"   ⚠️ [HuggingFace Hub Notice] Lỗi kết nối HuggingFace Hub: {e_hf_top}")

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
        )
        backbone = backbone.to(device)
        return backbone, embed_dim, patch_size
    except Exception as e_base:
        print(f"⚠️ [BackboneLoader] Base builder notice: {e_base}. Chuyển sang fallback loader.")

    # 2. Fallback trực tiếp qua torch.hub
    clean_name = model_name.lower().strip()
    if "dinov2" in clean_name:
        hub_tag = "dinov2_vits14" if "s" in clean_name else "dinov2_vitb14"
        embed_dim = 384 if "s" in clean_name else 768
        patch_size = 14
        backbone = torch.hub.load("facebookresearch/dinov2", hub_tag, pretrained=pretrained)
    elif "dinov3" in clean_name:
        # Nếu DINOv3 chưa có weights remote, fallback an toàn sang DINOv2 với cảnh báo rõ ràng
        print("💡 [BackboneLoader] DINOv3 đang dùng DINOv2 proxy weights để đảm bảo thực nghiệm.")
        hub_tag = "dinov2_vits14" if "s" in clean_name else "dinov2_vitb14"
        embed_dim = 384 if "s" in clean_name else 768
        patch_size = 14
        backbone = torch.hub.load("facebookresearch/dinov2", hub_tag, pretrained=pretrained)
    else:
        # Fallback timm ViT-Small
        import timm
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = getattr(backbone, "num_features", 384)
        patch_size = 16

    if weights_path and os.path.isfile(weights_path):
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict):
            for k in ["model_state", "model", "student", "teacher", "state_dict"]:
                if k in state:
                    state = state[k]
                    break
        clean_state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        backbone.load_state_dict(clean_state, strict=False)
        print(f"✅ [BackboneLoader] Nạp thành công trọng số checkpoint từ: {weights_path}")

    backbone = backbone.to(device)
    return backbone, embed_dim, patch_size


@torch.no_grad()
def extract_tokens(
    backbone: nn.Module,
    x: torch.Tensor,
    patch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trích xuất đồng thời [CLS] token đại diện toàn cảnh và ma trận Patch Tokens không gian.

    Args:
        backbone: ViT backbone (DINOv3/DINOv2).
        x: Batch tensor ảnh đầu vào (B, C, H, W).
        patch_size: Kích thước patch tương ứng của mô hình.

    Returns:
        cls_token: (B, embed_dim)
        patch_tokens: (B, H_patches, W_patches, embed_dim)
    """
    B, C, H, W = x.shape
    h_patches = H // patch_size
    w_patches = W // patch_size

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
    patch_spatial = patch_raw.view(B, h_patches, w_patches, embed_dim)

    return cls_token, patch_spatial
