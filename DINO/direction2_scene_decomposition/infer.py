"""
=============================================================================
 Hướng 2: Scene Decomposition — Inference & Inpainting Script (Production)
 Tự động tách lớp phương tiện và khôi phục mặt đường sạch xe (Road Inpainting)
 Hỗ trợ cả 2 chế độ:
   - Chế độ 1: Frame + Background Prior (--bg_path hoặc --bg_dir) để xóa sạch xe 100%
   - Chế độ 2: Single-frame tự do (khi không có ảnh background)
 Xuất đầy đủ các lớp thành phần và ảnh composite 6 panels trực quan như lúc train.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from tqdm.auto import tqdm

# Import local modules
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_curr_dir)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from common.backbone_loader import get_dino_backbone
from common.matcher import TrafficPairMatcher
from common.gpu_utils import load_checkpoint
from models import TrafficDecompositionNet


def save_composite_figure(
    origin_pil: Image.Image,
    pred_bg_pil: Image.Image,
    pred_fg_pil: Image.Image,
    pred_mask_pil: Image.Image,
    save_path: str,
    prior_bg_pil: Image.Image = None,
    recon_pil: Image.Image = None,
    title_text: str = "Traffic-Decompose Layer Separation",
):
    """Xuất ảnh so sánh trực quan đa bảng (chuẩn như đồ thị trong quá trình training)."""
    has_prior = (prior_bg_pil is not None)
    cols = 3 if has_prior else 2
    fig, axes = plt.subplots(2, cols, figsize=(15 if has_prior else 10, 9), dpi=150)
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    if has_prior:
        # Bảng 1: Ảnh gốc
        axes[0, 0].imshow(origin_pil)
        axes[0, 0].set_title("Input Origin Image", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        # Bảng 2: Ảnh nền Prior
        axes[0, 1].imshow(prior_bg_pil)
        axes[0, 1].set_title("Background Prior (Reference)", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

        # Bảng 3: Ảnh tái tạo
        if recon_pil is not None:
            axes[0, 2].imshow(recon_pil)
            axes[0, 2].set_title("Reconstructed Origin", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")

        # Bảng 4: Nền tái tạo sạch xe
        axes[1, 0].imshow(pred_bg_pil)
        axes[1, 0].set_title("Decomposed Background Layer (Clean)", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")

        # Bảng 5: Tiền cảnh phương tiện
        axes[1, 1].imshow(pred_fg_pil)
        axes[1, 1].set_title("Decomposed Foreground Layer", fontsize=11, fontweight="bold")
        axes[1, 1].axis("off")

        # Bảng 6: Mặt nạ phân bố xe
        axes[1, 2].imshow(pred_mask_pil, cmap="magma")
        axes[1, 2].set_title("Predicted Vehicle Alpha Mask", fontsize=11, fontweight="bold")
        axes[1, 2].axis("off")
    else:
        # Chế độ 4 bảng khi không có prior
        axes[0, 0].imshow(origin_pil)
        axes[0, 0].set_title("Input Origin Image", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        if recon_pil is not None:
            axes[0, 1].imshow(recon_pil)
            axes[0, 1].set_title("Reconstructed Origin", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(pred_bg_pil)
        axes[1, 0].set_title("Decomposed Background (Inpainted)", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(pred_mask_pil, cmap="magma")
        axes[1, 1].set_title("Predicted Vehicle Alpha Mask", fontsize=11, fontweight="bold")
        axes[1, 1].axis("off")

    fig.suptitle(title_text, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device.lower() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(" 🚗 TRAFFIC SCENE DECOMPOSITION & ROAD INPAINTING INFERENCE")
    print("=" * 78)
    print(f" Checkpoint Path : {args.weights}")
    print(f" Input Target    : {args.input_path}")
    print(f" Background Prior: {args.bg_path if args.bg_path else (args.bg_dir if args.bg_dir else 'None (Single-frame mode)')}")
    print(f" Output Dir      : {args.output_dir}")
    print(f" Device          : {device}")
    print("=" * 78)

    # 1. Nạp Backbone và Mô hình
    # ĐẶT PRETRAINED=TRUE: Đảm bảo backbone ViT luôn giữ trọng số ImageNet chuẩn nếu checkpoint chỉ lưu Heads
    backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        pretrained=True,
        device=device,
        hf_token=getattr(args, "hf_token", None),
    )
    model = TrafficDecompositionNet(
        backbone=backbone,
        embed_dim=embed_dim,
        patch_size=patch_size,
    ).to(device)

    # Nạp weights thông minh (tự xử lý module. prefix nếu checkpoint từ Multi-GPU)
    load_checkpoint(
        load_path=args.weights,
        model=model,
        device=device,
        strict=False,
        verbose=True,
    )
    model.eval()

    # 2. Khởi tạo Matcher nếu có chỉ định thư mục Backgrounds
    matcher = None
    if args.bg_dir and os.path.isdir(args.bg_dir):
        origin_ref_dir = args.input_path if os.path.isdir(args.input_path) else os.path.dirname(os.path.abspath(args.input_path))
        matcher = TrafficPairMatcher(
            bg_dir=args.bg_dir,
            origin_dir=origin_ref_dir,
            match_strategy=args.match_strategy,
        )
        matcher.build_background_index()
        print(f"🔗 [Matcher] Đã nạp chỉ mục background từ: {args.bg_dir} (Chiến lược: {args.match_strategy})")

    # 3. Thu thập danh sách ảnh đầu vào
    if os.path.isfile(args.input_path):
        image_paths = [args.input_path]
    elif os.path.isdir(args.input_path):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if os.path.splitext(f)[1].lower() in exts]
        image_paths.sort()
    else:
        raise ValueError(f"Đường dẫn không hợp lệ: {args.input_path}")

    # 4. Pipeline tiền xử lý ảnh
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    print(f"\n🚀 Đang xử lý {len(image_paths)} khung hình...\n")

    for img_p in tqdm(image_paths, desc="Decomposing Scenes"):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        try:
            pil_img = Image.open(img_p).convert("RGB")
        except Exception as e_read:
            print(f"⚠️ [Error] Không thể đọc ảnh {img_p}: {e_read}")
            continue

        orig_w, orig_h = pil_img.size
        x = tf(pil_img).unsqueeze(0).to(device)

        # 5. Xác định ảnh Background Prior (nếu có)
        prior_tensor = None
        prior_pil_img = None
        matched_bg_path = None

        if args.bg_path and os.path.isfile(args.bg_path):
            matched_bg_path = args.bg_path
        elif matcher:
            route_id, ts, hour = matcher.parse_origin_filename(os.path.basename(img_p))
            if route_id is not None:
                bg_cands = matcher.find_best_background(route_id, hour if hour is not None else 12)
                if bg_cands:
                    matched_bg_path = bg_cands[0]

        if matched_bg_path and os.path.isfile(matched_bg_path):
            try:
                prior_pil_img = Image.open(matched_bg_path).convert("RGB")
                prior_tensor = tf(prior_pil_img).unsqueeze(0).to(device)
            except Exception as e_bg:
                print(f"⚠️ [Warning] Không thể nạp ảnh background {matched_bg_path}: {e_bg}")

        # 6. Suy luận qua mô hình
        with torch.no_grad():
            preds = model(x, prior=prior_tensor, use_prior=(prior_tensor is not None))

        # 7. Chuyển đổi tensor về hình ảnh
        pred_bg_arr = preds["pred_bg"].squeeze().cpu().permute(1, 2, 0).numpy()
        pred_fg_arr = preds["pred_fg"].squeeze().cpu().permute(1, 2, 0).numpy()
        pred_mask_arr = preds.get("alpha_mask", preds.get("pred_mask")).squeeze().cpu().numpy()

        pred_bg_img = Image.fromarray((np.clip(pred_bg_arr, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BICUBIC)
        pred_fg_img = Image.fromarray((np.clip(pred_fg_arr, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BICUBIC)
        pred_mask_img = Image.fromarray((np.clip(pred_mask_arr, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)

        recon_pil_img = None
        if "recon_origin" in preds:
            recon_arr = preds["recon_origin"].squeeze().cpu().permute(1, 2, 0).numpy()
            recon_pil_img = Image.fromarray((np.clip(recon_arr, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BICUBIC)

        # 8. Lưu kết quả các lớp đơn lẻ
        pred_bg_img.save(os.path.join(args.output_dir, f"{stem}_clean_road.jpg"))
        pred_fg_img.save(os.path.join(args.output_dir, f"{stem}_vehicles_only.jpg"))
        pred_mask_img.save(os.path.join(args.output_dir, f"{stem}_alpha_mask.png"))

        # Bản đồ độ bất định sigma (nếu có)
        if "sigma" in preds:
            pred_sigma_arr = preds["sigma"].squeeze().cpu().numpy()
            pred_sigma_img = Image.fromarray((np.clip(pred_sigma_arr / 0.50, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
            pred_sigma_img.save(os.path.join(args.output_dir, f"{stem}_uncertainty_sigma.png"))

        # 9. Xuất ảnh composite so sánh trực quan (giống giao diện train)
        if args.save_composite:
            composite_path = os.path.join(args.output_dir, f"{stem}_composite.png")
            prior_resized = prior_pil_img.resize((orig_w, orig_h), Image.BICUBIC) if prior_pil_img else None
            save_composite_figure(
                origin_pil=pil_img,
                pred_bg_pil=pred_bg_img,
                pred_fg_pil=pred_fg_img,
                pred_mask_pil=pred_mask_img,
                save_path=composite_path,
                prior_bg_pil=prior_resized,
                recon_pil=recon_pil_img,
                title_text=f"Traffic-Decompose Inference — {stem}",
            )

    print(f"\n✅ [Hoàn tất] Toàn bộ kết quả phân rã đã được lưu tại: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Traffic Scene Decomposition (Chế độ Single-frame & Frame+Prior)")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file .pth checkpoint")
    parser.add_argument("--input_path", type=str, required=True, help="File ảnh đơn lẻ hoặc thư mục ảnh cần phân rã")
    parser.add_argument("--output_dir", type=str, default="checkpoints/direction2_scene_decomp/inferred", help="Thư mục lưu kết quả")
    parser.add_argument("--bg_path", type=str, default=None, help="Đường dẫn ảnh background tĩnh cụ thể để làm prior")
    parser.add_argument("--bg_dir", type=str, default=None, help="Thư mục chứa các ảnh background (tự động ghép cặp theo camera)")
    parser.add_argument("--match_strategy", type=str, default="route_hourly", help="Chiến lược đối sánh background: route_hourly, camera_id, etc.")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Tên backbone DINOv3")
    parser.add_argument("--img_size", type=int, default=256, help="Kích thước ảnh xử lý")
    parser.add_argument("--save_composite", action="store_true", default=True, help="Lưu thêm ảnh ghép so sánh trực quan đa bảng (composite)")
    parser.add_argument("--no_composite", action="store_false", dest="save_composite", help="Không lưu ảnh composite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị ('cuda' hoặc 'cpu')")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face user access token (nếu dùng Meta DINOv3 đóng)")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_inference(cli_args)

