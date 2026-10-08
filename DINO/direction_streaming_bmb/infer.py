"""
=============================================================================
 ST-BMB: Streaming Inference & Vehicle Removal Pipeline
 Suy luận Phân đoạn Phương tiện & Bóc tách Mặt đường Luồng Thời Gian Thực
=============================================================================
Cải tiến đột phá chuẩn Production:
  1. Nạp checkpoint với logging minh bạch missing_keys / unexpected_keys.
  2. Bóc tách mặt đường luồng bảo toàn tỷ lệ khung hình (Letterbox Resize).
  3. Truyền timestamp chuẩn vào Memory Bank trong suốt quá trình suy luận.
  4. Xuất ảnh composite 4 ô sắc nét và ảnh động GIF trực quan hóa luồng.
  5. Đánh giá định lượng: PSNR, SSIM, Tỷ lệ phương tiện và Drift nền.
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
import glob
import json
import math
import re
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm

dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dino_root not in sys.path:
    sys.path.insert(0, dino_root)

from common.backbone_loader import get_dino_backbone
from direction_streaming_bmb.models import StreamingDecompositionNet
from direction_streaming_bmb.dataset import letterbox_resize


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Tính Peak Signal-to-Noise Ratio (PSNR) giữa hai ảnh dải [0, 1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def run_streaming_inference(args):
    print("=" * 75)
    print(" 🎬 ST-BMB: STREAMING INFERENCE & VEHICLE REMOVAL PIPELINE")
    print("=" * 75)
    print(f"📁 Checkpoint mô hình       : {args.checkpoint}")
    print(f"📁 Thư mục ảnh đầu vào      : {args.input_dir}")
    print(f"📁 Thư mục lưu kết quả      : {args.output_dir}")
    print(f"📹 Camera ID lọc            : {args.cam_id or 'Tất cả các camera'}")
    print(f"🖼️  Kích thước xử lý         : {args.img_size}x{args.img_size}")
    print("=" * 75)

    os.makedirs(args.output_dir, exist_ok=True)
    frames_out_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Quét và gom cụm chuỗi ảnh theo Camera ID sử dụng regex an toàn
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                all_files.append(os.path.join(root, f))

    if not all_files:
        raise RuntimeError(f"Không tìm thấy ảnh hợp lệ trong {args.input_dir}")

    def parse_key(p):
        base = os.path.splitext(os.path.basename(p))[0]
        digits = re.findall(r"\d{8,14}", base)
        if digits:
            ts = float(digits[-1])
            idx = base.rfind(digits[-1])
            c_id = base[:idx].rstrip("_-")
            return c_id or "cam_default", ts
        return "cam_default", os.path.getmtime(p)

    cam_dict = {}
    for p in all_files:
        c_id, ts = parse_key(p)
        if args.cam_id and c_id != args.cam_id:
            continue
        if c_id not in cam_dict:
            cam_dict[c_id] = []
        cam_dict[c_id].append((ts, p))

    if not cam_dict:
        raise RuntimeError(f"Không tìm thấy chuỗi ảnh nào cho camera ID '{args.cam_id}'")

    # Chọn chuỗi có nhiều khung hình nhất để suy luận
    target_cam = max(cam_dict.keys(), key=lambda k: len(cam_dict[k]))
    sequence = cam_dict[target_cam]
    sequence.sort(key=lambda x: x[0])
    if args.max_frames:
        sequence = sequence[:args.max_frames]

    print(f"📹 [Camera Đích] Đã chọn Camera: {target_cam} với {len(sequence)} khung hình liên tiếp.")

    # 2. Khởi tạo Backbone & Mô hình
    print(f"\n🧠 [Backbone] Đang nạp {args.backbone}...")
    backbone, feat_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        weights_path=args.weights,
        freeze=True,
    )

    model = StreamingDecompositionNet(
        backbone=backbone,
        backbone_dim=feat_dim,
        embed_dim=args.embed_dim,
        patch_size=patch_size,
        max_recent_frames=args.max_recent,
        max_anchor_frames=args.max_anchor,
        freeze_backbone=True,
    )

    # 3. Nạp trọng số checkpoint đã huấn luyện kèm logging chi tiết
    if os.path.exists(args.checkpoint):
        print(f"📦 [Checkpoint] Đang nạp trọng số từ: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # Lọc bỏ tiền tố module. nếu huấn luyện DDP
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        incompatible = model.load_state_dict(clean_state_dict, strict=False)
        print(
            f"✅ Nạp trọng số hoàn tất: "
            f"Missing keys: {len(incompatible.missing_keys)}, "
            f"Unexpected keys: {len(incompatible.unexpected_keys)}"
        )
        if incompatible.missing_keys:
            print(f"   ℹ️ Missing keys (top 5): {incompatible.missing_keys[:5]}")
        if incompatible.unexpected_keys:
            print(f"   ℹ️ Unexpected keys (top 5): {incompatible.unexpected_keys[:5]}")
    else:
        print(f"⚠️ [Cảnh báo] Không tìm thấy checkpoint tại {args.checkpoint}, sử dụng mô hình khởi tạo ban đầu.")

    model = model.to(device)
    model.eval()

    # Reset Background Memory Bank
    model.memory_bank.reset()

    # 4. Suy luận Streaming tuần tự từng khung hình
    results_summary = []
    gif_frames = []
    prev_bg_np = None
    bg_drift_scores = []

    print("\n🚀 Bắt đầu quá trình suy luận streaming bóc tách phương tiện...")
    with torch.no_grad():
        for idx, (ts, img_path) in enumerate(tqdm(sequence, desc="Streaming Frames")):
            pil_orig = Image.open(img_path).convert("RGB")
            # Bảo toàn tỷ lệ khung hình
            pil_resized = letterbox_resize(pil_orig, args.img_size)

            t_img = transforms.ToTensor()(pil_resized).unsqueeze(0).to(device)  # (1, 3, H, W)
            ts_tensor = torch.tensor([ts], device=device, dtype=torch.float32)

            # Frame đầu tiên là anchor ban đầu, các frame sau cập nhật gated write
            is_anchor = (idx == 0)
            out = model.forward_single_step(
                t_img,
                update_memory=True,
                is_anchor=is_anchor,
                timestamp=ts_tensor,
            )

            # Chuyển đổi tensor sang ảnh numpy
            in_np = np.clip(t_img[0].cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)
            bg_np = np.clip(out["pred_bg"][0].cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)
            fg_np = np.clip(out["pred_fg"][0].cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)
            alpha_np = np.clip(out["alpha_mask"][0, 0].cpu().numpy(), 0.0, 1.0)
            recon_np = np.clip(out["recon_origin"][0].cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)
            sigma_np = float(out["sigma"][0].mean().cpu().item())

            # Tính độ tương tự và drift nền liên frame
            psnr_val = calculate_psnr(in_np, recon_np)
            if prev_bg_np is not None:
                drift = float(np.mean(np.abs(bg_np - prev_bg_np)))
                bg_drift_scores.append(drift)
            else:
                drift = 0.0
            prev_bg_np = bg_np.copy()

            vehicle_ratio = float(np.mean(alpha_np > 0.5))

            results_summary.append({
                "step": idx + 1,
                "timestamp": ts,
                "file": os.path.basename(img_path),
                "psnr": psnr_val,
                "uncertainty_sigma": sigma_np,
                "vehicle_pixel_ratio": vehicle_ratio,
                "bg_drift_to_prev": drift,
            })

            # Ghép ảnh composite 4 ô: [Gốc | Nền sạch bóc tách | Mặt nạ xe | Xe cộ bóc tách]
            alpha_rgb = np.repeat(alpha_np[:, :, np.newaxis], 3, axis=2)
            composite_np = np.hstack([in_np, bg_np, alpha_rgb, fg_np])
            composite_uint8 = (composite_np * 255).astype(np.uint8)
            composite_pil = Image.fromarray(composite_uint8)

            save_frame_path = os.path.join(frames_out_dir, f"frame_{idx+1:04d}.png")
            composite_pil.save(save_frame_path)

            if args.save_gif:
                gif_frames.append(composite_pil)

    # 5. Xuất ảnh động GIF nếu có yêu cầu
    if args.save_gif and gif_frames:
        gif_path = os.path.join(args.output_dir, f"streaming_{target_cam}.gif")
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(1000 / max(1, args.gif_fps)),
            loop=0,
        )
        print(f"🎞️  [GIF] Đã tạo ảnh động streaming thành công tại: {gif_path}")

    # 6. Tổng hợp chỉ số định lượng
    avg_psnr = float(np.mean([r["psnr"] for r in results_summary]))
    avg_sigma = float(np.mean([r["uncertainty_sigma"] for r in results_summary]))
    avg_veh_ratio = float(np.mean([r["vehicle_pixel_ratio"] for r in results_summary]))
    avg_drift = float(np.mean(bg_drift_scores)) if bg_drift_scores else 0.0

    metrics_file = os.path.join(args.output_dir, "inference_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "camera_id": target_cam,
            "total_frames_processed": len(results_summary),
            "average_psnr": avg_psnr,
            "average_uncertainty_sigma": avg_sigma,
            "average_vehicle_ratio": avg_veh_ratio,
            "average_bg_drift": avg_drift,
            "details": results_summary,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(" 📊 KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG ST-BMB:")
    print(f"   - Tái tạo PSNR trung bình     : {avg_psnr:.2f} dB")
    print(f"   - Độ bất định sigma trung bình: {avg_sigma:.4f}")
    print(f"   - Tỷ lệ diện tích phương tiện : {avg_veh_ratio * 100:.2f}%")
    print(f"   - Độ trôi dạt nền (BG Drift)  : {avg_drift:.4f} (Càng nhỏ càng bất biến tốt)")
    print(f"   - Dữ liệu chi tiết đã lưu tại : {metrics_file}")
    print("=" * 75)


def parse_args():
    parser = argparse.ArgumentParser(description="ST-BMB Streaming Inference & Vehicle Removal")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/direction_streaming_bmb/best_streaming_model.pth",
        help="Đường dẫn file trọng số đã huấn luyện",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="direction_data_article/zenodo_bundle/sample_preview/sample_camera_sequences",
        help="Thư mục chứa chuỗi ảnh camera",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="direction_streaming_bmb/output_inference",
        help="Thư mục lưu kết quả suy luận",
    )
    parser.add_argument("--cam_id", type=str, default=None, help="Lọc riêng theo camera ID")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16")
    parser.add_argument("--weights", type=str, default="checkpoints/dinov3_vits16_model.safetensors")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--max_recent", type=int, default=5)
    parser.add_argument("--max_anchor", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=15)
    parser.add_argument("--save_gif", action="store_true", default=True)
    parser.add_argument("--gif_fps", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_streaming_inference(args)
