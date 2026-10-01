"""
=============================================================================
 Hướng 2: Zero-Shot Vehicle Segmentation — Execution Pipeline
 Trích xuất Pseudo-Mask tự động và Huấn luyện Student Segmentor
=============================================================================
"""

import argparse
import os
import sys

# Chống xung đột OpenMP trên Windows Anaconda và đảm bảo hiển thị ký tự tiếng Việt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

# Import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.matcher import TrafficPairMatcher
from common.subtraction import BackgroundSubtractor
from common.backbone_loader import get_dino_backbone
from pca_extractor import DINOPCAExtractor
from fusion import MaskFusionEngine
from segmentor import VehicleSegmentor, DiceLoss


def plot_segmentation_artifact(
    origin_np: np.ndarray,
    bg_np: np.ndarray,
    delta_norm: np.ndarray,
    pca_rgb: np.ndarray,
    fused_mask: np.ndarray,
    save_path: str,
    cam_id: str,
):
    """Vẽ biểu đồ phân đoạn chuyên nghiệp 6 ô cho bài báo khoa học."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)

    # 1. Ảnh Background
    axes[0, 0].imshow(bg_np)
    axes[0, 0].set_title(f"Background (Camera {cam_id})", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Ảnh Origin
    axes[0, 1].imshow(origin_np)
    axes[0, 1].set_title("Origin (Vehicles Present)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # 3. Delta Map
    axes[0, 2].imshow(delta_norm, cmap="inferno")
    axes[0, 2].set_title("Δ Background Difference", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # 4. PCA RGB Features
    axes[1, 0].imshow(pca_rgb)
    axes[1, 0].set_title("DINOv3 Emergent PCA (PC1/2/3)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # 5. Fused Binary Mask
    axes[1, 1].imshow(fused_mask, cmap="gray")
    fg_ratio = np.mean(fused_mask > 127) * 100
    axes[1, 1].set_title(f"Zero-Shot Pseudo Mask (FG: {fg_ratio:.1f}%)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # 6. Overlay trên ảnh gốc
    overlay = origin_np.copy()
    mask_bin = fused_mask > 127
    overlay[mask_bin] = (overlay[mask_bin] * 0.4 + np.array([0, 255, 128], dtype=np.float32) * 0.6).astype(np.uint8)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Segmented Vehicle Overlay", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def run_pipeline(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    mask_dir = os.path.join(args.output_dir, "pseudo_masks")
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(" 🚗 ZERO-SHOT VEHICLE SEGMENTATION PIPELINE")
    print("=" * 78)
    print(f" Backbone Architecture : {args.backbone}")
    print(f" Background Directory  : {args.bg_dir}")
    print(f" Origin Directory      : {args.origin_dir}")
    print(f" Output Directory      : {args.output_dir}")
    print(f" Device                : {device}")
    print("=" * 78)

    # 1. Khởi tạo Matcher & Backbone
    matcher = TrafficPairMatcher(bg_dir=args.bg_dir, origin_dir=args.origin_dir, match_strategy=args.match_strategy)
    pairs = matcher.discover_pairs(max_pairs=args.max_samples)
    if not pairs:
        print("❌ Không tìm thấy cặp ảnh nào hợp lệ.")
        return

    print(f"✅ Đã tìm thấy {len(pairs)} cặp ảnh để trích xuất mặt nạ.")

    backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        pretrained=True,
        device=device,
    )

    subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)
    pca_extractor = DINOPCAExtractor(backbone=backbone, patch_size=patch_size, img_size=args.img_size, device=device)
    fusion_engine = MaskFusionEngine()

    # 2. Vòng lặp sinh Pseudo-Masks
    print("\n🔍 Đang sinh Pseudo-Masks qua cơ chế Fusion...")
    generated_data = []

    for i, p in enumerate(tqdm(pairs, desc="Generating Masks")):
        origin_path = p["origin_path"]
        bg_path = p["bg_path"]
        cam_id = p["route_id"]
        fname = p["origin_name"]
        stem = os.path.splitext(fname)[0]

        try:
            origin_pil = Image.open(origin_path).convert("RGB")
            bg_pil = Image.open(bg_path).convert("RGB")
            if origin_pil.size != bg_pil.size:
                bg_pil = bg_pil.resize(origin_pil.size, Image.BICUBIC)

            origin_np = np.array(origin_pil)
            bg_np = np.array(bg_pil)

            # a. Trừ nền
            delta_norm, _ = subtractor.compute_delta(origin_np, bg_np)
            delta_mask = subtractor.extract_binary_mask(delta_norm, method="otsu")

            # b. DINOv3 PCA
            pca_rgb, pc1_mask = pca_extractor.compute_pca_maps(origin_pil)

            # c. Fusion
            fused_mask = fusion_engine.fuse(delta_mask, pc1_mask, origin_np)

            # Lưu mask nhị phân
            out_mask_path = os.path.join(mask_dir, f"{stem}_mask.png")
            Image.fromarray(fused_mask).save(out_mask_path)
            generated_data.append((origin_path, out_mask_path))

            # Lưu visualization
            if i < args.max_vis:
                vis_save = os.path.join(vis_dir, f"{stem}_segmentation.png")
                plot_segmentation_artifact(
                    origin_np=origin_np,
                    bg_np=bg_np,
                    delta_norm=delta_norm,
                    pca_rgb=pca_rgb,
                    fused_mask=fused_mask,
                    save_path=vis_save,
                    cam_id=cam_id,
                )
        except Exception as e:
            print(f"⚠️ Bỏ qua {fname} do lỗi: {e}")
            continue

    print(f"\n✅ Đã trích xuất thành công {len(generated_data)} pseudo-masks lưu tại: {mask_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-Shot Vehicle Segmentation")
    parser.add_argument("--bg_dir", type=str, default="traffic_backgrounds", help="Thư mục background")
    parser.add_argument("--origin_dir", type=str, default="output", help="Thư mục origin")
    parser.add_argument("--output_dir", type=str, default="checkpoints/direction2_zero_shot_seg", help="Thư mục xuất kết quả")
    parser.add_argument("--match_strategy", type=str, default="route_hourly", choices=["route_hourly", "same_name", "camera_id"])
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone DINOv3/v2")
    parser.add_argument("--img_size", type=int, default=224, help="Kích thước xử lý ảnh")
    parser.add_argument("--max_samples", type=int, default=100, help="Số cặp tối đa xử lý")
    parser.add_argument("--max_vis", type=int, default=10, help="Số ảnh visualization xuất ra")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị ('cuda' hoặc 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
