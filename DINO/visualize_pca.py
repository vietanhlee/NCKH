"""
=============================================================================
 DINO Emergent PCA Feature Map Visualizer
 Trực quan hóa đặc trưng tự giám sát không gian nổi trội (Emergent Property)
 Phân tích 3 thành phần chính PCA (PC1, PC2, PC3) trên Patch Tokens thành ảnh RGB
 Khắc phục triệt để và sinh ra đúng 100% định dạng đồ thị báo cáo NCKH chuẩn Meta AI
=============================================================================
"""

import os
import sys
import math
import glob
import argparse
from typing import List, Optional

# Chống xung đột OpenMP và cấu hình UTF-8 cho Windows Terminal
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

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Nạp module tiện ích chung
dino_dir = os.path.dirname(os.path.abspath(__file__))
if dino_dir not in sys.path:
    sys.path.insert(0, dino_dir)
from common.backbone_loader import get_dino_backbone


@torch.no_grad()
def generate_emergent_pca_maps(
    backbone: nn.Module,
    image_paths: List[str],
    save_path: str = "emergent_pca_feature_maps.png",
    device: str = "cpu",
    img_size: int = 224,
    title_prefix: str = "DINOv2",
):
    """
    Trích xuất và vẽ đồ thị Emergent PCA Feature Map chuẩn Meta AI:
    Cột 1: Camera Frame gốc (224x224)
    Cột 2: Emergent PCA Feature Map (RGB)
    """
    device_obj = torch.device(device)
    backbone = backbone.to(device_obj).eval()

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_samples = len(image_paths)
    if num_samples == 0:
        print("❌ Không có ảnh nào để trực quan hóa!")
        return

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 3.4 * num_samples), dpi=250)
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for idx, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
                orig_resized = img_rgb.resize((img_size, img_size))
                inp = eval_transform(img_rgb).unsqueeze(0).to(device_obj)

                # 1. Trích xuất Spatial Patch Tokens
                patch_tokens = None
                if hasattr(backbone, "get_intermediate_layers"):
                    try:
                        out = backbone.get_intermediate_layers(inp, n=1)[0]
                        if isinstance(out, tuple):
                            out = out[0]
                        raw_tokens = out.squeeze(0).cpu().numpy()
                        n_tokens = raw_tokens.shape[0]
                        if math.isqrt(n_tokens) ** 2 == n_tokens:
                            patch_tokens = raw_tokens
                        elif math.isqrt(n_tokens - 1) ** 2 == (n_tokens - 1):
                            patch_tokens = raw_tokens[1:]
                        else:
                            patch_tokens = raw_tokens
                    except Exception as e_tok:
                        print(f"⚠️ get_intermediate_layers warning: {e_tok}")

                if patch_tokens is None and hasattr(backbone, "forward_features"):
                    try:
                        feat = backbone.forward_features(inp)
                        if isinstance(feat, dict):
                            tokens = feat.get("x_norm_patchtokens", feat.get("x_prenorm", None))
                        else:
                            tokens = feat
                        patch_tokens = tokens.squeeze(0).cpu().numpy()
                        if math.isqrt(patch_tokens.shape[0] - 1) ** 2 == (patch_tokens.shape[0] - 1):
                            patch_tokens = patch_tokens[1:]
                    except Exception as e_ff:
                        print(f"⚠️ forward_features warning: {e_ff}")

                # 2. Phân tích PCA 3 thành phần chính
                if patch_tokens is not None and len(patch_tokens) > 0:
                    pca = PCA(n_components=3)
                    # Chuẩn hóa tâm đặc trưng
                    tokens_centered = patch_tokens - np.mean(patch_tokens, axis=0, keepdims=True)
                    pca_features = pca.fit_transform(tokens_centered)  # (N_patches, 3)

                    # Min-Max normalize về đoạn [0, 1] cho hiển thị kênh màu RGB
                    for c in range(3):
                        c_min, c_max = pca_features[:, c].min(), pca_features[:, c].max()
                        pca_features[:, c] = (pca_features[:, c] - c_min) / (c_max - c_min + 1e-8)

                    h_patches = w_patches = int(math.isqrt(pca_features.shape[0]))
                    pca_img = pca_features[: h_patches * w_patches].reshape(h_patches, w_patches, 3)

                    # Vẽ Cột Trái: Camera Frame gốc
                    axes[idx, 0].imshow(orig_resized)
                    axes[idx, 0].set_title(f"Camera Frame: {os.path.basename(path)}", fontsize=10, fontweight="bold")
                    axes[idx, 0].axis("off")

                    # Vẽ Cột Phải: Emergent PCA Feature Map (RGB)
                    axes[idx, 1].imshow(pca_img, interpolation="bilinear")
                    axes[idx, 1].set_title(f"{title_prefix} Emergent PCA Feature Map (RGB)", fontsize=10, fontweight="bold", color="#1f77b4")
                    axes[idx, 1].axis("off")
                else:
                    # Fallback nếu không trích xuất được tokens
                    axes[idx, 0].imshow(orig_resized)
                    axes[idx, 0].set_title(f"Camera Frame: {os.path.basename(path)}", fontsize=10, fontweight="bold")
                    axes[idx, 0].axis("off")
                    axes[idx, 1].imshow(orig_resized)
                    axes[idx, 1].set_title("Feature Map Fallback", fontsize=10, fontweight="bold")
                    axes[idx, 1].axis("off")
        except Exception as e_img:
            print(f"❌ Lỗi xử lý ảnh {path}: {e_img}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=250)
    # Lưu thêm bản PDF chất lượng xuất bản
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    try:
        plt.savefig(pdf_path, bbox_inches="tight")
    except Exception:
        pass
    plt.close()
    print(f"🎉 [Success] Đã lưu Emergent PCA Feature Maps thành công tại:")
    print(f"   👉 PNG: {os.path.abspath(save_path)}")
    print(f"   👉 PDF: {os.path.abspath(pdf_path)}")


def get_safe_device(requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Thử tạo tensor nhỏ trên CUDA để kiểm tra driver
            _ = torch.zeros(1, device="cuda")
            return "cuda"
    except Exception:
        pass
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Trực quan hóa Emergent PCA Feature Map của DINO")
    parser.add_argument("--img_dir", type=str, default="output", help="Thư mục chứa ảnh giao thông, đường dẫn 1 ảnh, hoặc mẫu wildcard (ví dụ: 'output/123_*.jpg')")
    parser.add_argument("--cam_id", "--camera_id", dest="cam_id", type=str, default=None, help="Chỉ định ID camera cụ thể (ví dụ: '123' hoặc danh sách '123,566,101,249')")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn file checkpoint đã huấn luyện (.pth)")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14", help="Tên backbone (dinov2_vits14 / dinov3_vits16)")
    parser.add_argument("--save_path", type=str, default="emergent_pca_feature_maps.png", help="Đường dẫn lưu file ảnh kết quả")
    parser.add_argument("--num_samples", type=int, default=4, help="Số lượng ảnh mẫu muốn hiển thị (mặc định 4 ảnh như báo cáo)")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị ('auto', 'cuda' hoặc 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_safe_device(args.device)

    # 1. Thu thập danh sách ảnh ban đầu
    all_found = []
    if any(ch in args.img_dir for ch in ["*", "?"]):
        # Mẫu wildcard (ví dụ: output/123_*.jpg)
        all_found = sorted(glob.glob(args.img_dir))
    elif os.path.isfile(args.img_dir):
        all_found = [args.img_dir]
    elif os.path.isdir(args.img_dir):
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
        for pat in patterns:
            all_found.extend(glob.glob(os.path.join(args.img_dir, pat)))
            all_found.extend(glob.glob(os.path.join(args.img_dir, "**", pat), recursive=True))
        all_found = sorted(list(set(all_found)))
    else:
        print(f"❌ Không tìm thấy đường dẫn hoặc mẫu: {args.img_dir}")
        sys.exit(1)

    if not all_found:
        print(f"❌ Không tìm thấy file ảnh nào trong {args.img_dir}!")
        sys.exit(1)

    # 2. Lọc theo camera nếu người dùng chỉ định --cam_id
    if args.cam_id:
        target_cams = [c.strip() for c in str(args.cam_id).split(",") if c.strip()]
        sample_paths = []
        for cam in target_cams:
            matched = [
                p for p in all_found
                if os.path.basename(p).startswith(f"{cam}_")
                or f"route_{cam}" in p.replace("\\", "/")
                or f"/{cam}/" in p.replace("\\", "/")
                or os.path.basename(p) == cam
            ]
            if matched:
                # Lấy số lượng ảnh phân bổ đều cho từng camera
                quota = max(1, math.ceil(args.num_samples / len(target_cams)))
                sample_paths.extend(matched[:quota])
                print(f"📷 [Camera {cam}] Khớp {len(matched)} ảnh -> Chọn {min(len(matched), quota)} ảnh mẫu.")
            else:
                print(f"⚠️ [Camera {cam}] Không tìm thấy ảnh nào bắt đầu bằng '{cam}_' trong {args.img_dir}!")
        sample_paths = sample_paths[:args.num_samples]
    else:
        sample_paths = all_found[:args.num_samples]

    if not sample_paths:
        print(f"❌ Không chọn được ảnh nào sau khi lọc! Vui lòng kiểm tra lại Camera ID.")
        sys.exit(1)

    print(f"🔍 Đã chọn {len(sample_paths)} ảnh mẫu. Đang nạp backbone '{args.backbone}'...")
    backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        pretrained=True,
        weights_path=args.weights,
        device=device,
    )

    title = "DINOv3" if "dinov3" in args.backbone.lower() else "DINOv2"
    generate_emergent_pca_maps(
        backbone=backbone,
        image_paths=sample_paths,
        save_path=args.save_path,
        device=device,
        img_size=224,
        title_prefix=title,
    )
