"""
=============================================================================
 Hướng 3: Scene Decomposition — Inference & Inpainting Script
 Tự động tách lớp phương tiện và khôi phục mặt đường sạch xe (Road Inpainting)
 từ ảnh giao thông đơn lẻ (không cần ảnh background khi chạy inference)
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

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from tqdm.auto import tqdm

# Import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.backbone_loader import get_dino_backbone
from models import TrafficDecompositionNet


def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(" 🚗 TRAFFIC SCENE DECOMPOSITION & ROAD INPAINTING INFERENCE")
    print("=" * 78)
    print(f" Checkpoint Path : {args.weights}")
    print(f" Input Target    : {args.input_path}")
    print(f" Output Dir      : {args.output_dir}")
    print(f" Device          : {device}")
    print("=" * 78)

    # 1. Nạp Backbone và Mô hình
    backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        pretrained=False,
        device=device,
    )
    model = TrafficDecompositionNet(
        backbone=backbone,
        embed_dim=embed_dim,
        patch_size=patch_size,
    ).to(device)

    from common.gpu_utils import load_checkpoint

    # Nạp weights thông minh (tự xử lý module. prefix nếu checkpoint từ Multi-GPU)
    load_checkpoint(
        load_path=args.weights,
        model=model,
        device=device,
        strict=True,
        verbose=True,
    )
    model.eval()

    # 2. Thu thập danh sách ảnh
    if os.path.isfile(args.input_path):
        image_paths = [args.input_path]
    elif os.path.isdir(args.input_path):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if os.path.splitext(f)[1].lower() in exts]
        image_paths.sort()
    else:
        raise ValueError(f"Đường dẫn không hợp lệ: {args.input_path}")

    # 3. Chạy Inference
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    for img_p in tqdm(image_paths, desc="Decomposing Scenes"):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        pil_img = Image.open(img_p).convert("RGB")
        orig_w, orig_h = pil_img.size

        x = tf(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(x)

        # Chuyển đổi tensor về hình ảnh
        pred_bg = preds["pred_bg"].squeeze().cpu().permute(1, 2, 0).numpy()
        pred_fg = preds["pred_fg"].squeeze().cpu().permute(1, 2, 0).numpy()
        pred_mask = preds["pred_mask"].squeeze().cpu().numpy()

        pred_bg_img = Image.fromarray((np.clip(pred_bg, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BICUBIC)
        pred_fg_img = Image.fromarray((np.clip(pred_fg, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BICUBIC)
        pred_mask_img = Image.fromarray((np.clip(pred_mask, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)

        # Lưu kết quả 3 lớp
        pred_bg_img.save(os.path.join(args.output_dir, f"{stem}_clean_road.jpg"))
        pred_fg_img.save(os.path.join(args.output_dir, f"{stem}_vehicles_only.jpg"))
        pred_mask_img.save(os.path.join(args.output_dir, f"{stem}_density_mask.png"))

    print(f"\n✅ Đã xuất toàn bộ kết quả phân rã tại thư mục: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Traffic Scene Decomposition")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file .pth checkpoint")
    parser.add_argument("--input_path", type=str, required=True, help="File ảnh đơn lẻ hoặc thư mục ảnh cần phân rã")
    parser.add_argument("--output_dir", type=str, default="checkpoints/direction2_scene_decomp/inferred", help="Thư mục lưu")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
