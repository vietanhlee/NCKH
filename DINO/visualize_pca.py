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
from common.matcher import TrafficPairMatcher
from common.subtraction import BackgroundSubtractor


@torch.no_grad()
def generate_emergent_pca_maps(
    backbone: nn.Module,
    image_paths: List[str],
    bg_dir: Optional[str] = None,
    save_path: str = "emergent_pca_feature_maps.png",
    device: str = "cpu",
    img_size: int = 518,
    patch_size: int = 14,
    pca_mode: str = "rgb",
    keep_aspect_ratio: bool = False,
    title_prefix: str = "DINOv2",
):
    """
    Trích xuất và vẽ đồ thị Emergent PCA Feature Map độ nét siêu cao (Super-Resolution PCA):
      - Mặc định img_size=518 theo chuẩn Meta DINOv2 (37x37 = 1369 patches thay vì 16x16 = 256 patches).
      - Áp dụng Percentile Normalization (1% - 99%) triệt tiêu outlier, tách biên xe sắc nét.
      - Phóng to Bicubic High-Resolution loại bỏ hoàn toàn răng cưa khối (pixelated) và mờ nhòe.
      - Hỗ trợ giữ nguyên tỷ lệ khung hình camera (keep_aspect_ratio) tránh méo xe.
      - Hỗ trợ 3 chế độ: 'rgb' (chuẩn Meta AI), 'foreground' (làm nổi bật xe), 'overlay' (phủ lên ảnh gốc).
    """
    device_obj = torch.device(device)
    backbone = backbone.to(device_obj).eval()

    num_samples = len(image_paths)
    if num_samples == 0:
        print("❌ Không có ảnh nào để trực quan hóa!")
        return

    # Khởi tạo matcher và subtractor nếu người dùng cung cấp bg_dir
    has_bg = False
    matcher = None
    subtractor = None
    if bg_dir and os.path.isdir(bg_dir):
        origin_sample_dir = os.path.dirname(image_paths[0]) if os.path.isfile(image_paths[0]) else image_paths[0]
        matcher = TrafficPairMatcher(bg_dir=bg_dir, origin_dir=origin_sample_dir, match_strategy="route_hourly")
        matcher.build_background_index()
        subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)
        has_bg = True
        print(f"🌆 [Background Matcher] Đã liên kết thư mục Background: {bg_dir} -> Chế độ 4 cột đối chiếu đầy đủ.")

    n_cols = 4 if has_bg else 2
    fig_w = 16 if has_bg else 8
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(fig_w, 3.6 * num_samples), dpi=250)
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for idx, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
                orig_w, orig_h = img_rgb.size

                # 1. Tính toán kích thước lưới patch (hỗ trợ cả vuông và giữ tỷ lệ khung hình thật)
                if keep_aspect_ratio:
                    scale = img_size / max(orig_w, orig_h)
                    target_w = max(patch_size, int(round(orig_w * scale / patch_size)) * patch_size)
                    target_h = max(patch_size, int(round(orig_h * scale / patch_size)) * patch_size)
                else:
                    target_w = target_h = max(patch_size, (img_size // patch_size) * patch_size)

                h_patches = target_h // patch_size
                w_patches = target_w // patch_size
                n_expected_patches = h_patches * w_patches

                eval_transform = transforms.Compose([
                    transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                orig_resized = img_rgb.resize((target_w, target_h), Image.BICUBIC)
                inp = eval_transform(img_rgb).unsqueeze(0).to(device_obj)

                # 2. Trích xuất Spatial Patch Tokens
                patch_tokens = None
                if hasattr(backbone, "get_intermediate_layers"):
                    try:
                        out = backbone.get_intermediate_layers(inp, n=1)[0]
                        if isinstance(out, tuple):
                            out = out[0]
                        raw_tokens = out.squeeze(0).cpu().numpy()
                        if raw_tokens.shape[0] == n_expected_patches + 1:
                            patch_tokens = raw_tokens[1:]
                        elif raw_tokens.shape[0] >= n_expected_patches:
                            patch_tokens = raw_tokens[:n_expected_patches]
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
                        raw_tokens = tokens.squeeze(0).cpu().numpy()
                        if raw_tokens.shape[0] == n_expected_patches + 1:
                            patch_tokens = raw_tokens[1:]
                        elif raw_tokens.shape[0] >= n_expected_patches:
                            patch_tokens = raw_tokens[:n_expected_patches]
                        else:
                            patch_tokens = raw_tokens
                    except Exception as e_ff:
                        print(f"⚠️ forward_features warning: {e_ff}")

                # 3. Phân tích PCA 3 thành phần chính siêu nét
                pca_render = None
                if patch_tokens is not None and len(patch_tokens) >= n_expected_patches:
                    pca = PCA(n_components=3)
                    tokens_centered = patch_tokens[:n_expected_patches] - np.mean(patch_tokens[:n_expected_patches], axis=0, keepdims=True)
                    pca_features = pca.fit_transform(tokens_centered)  # (n_expected_patches, 3)

                    # Chuẩn hóa phân vị (Percentile Normalization 1% - 99%)
                    # Triệt tiêu ngoại lai (outliers), tăng cường tương phản và độ tách biên chi tiết
                    for c in range(3):
                        p_low = np.percentile(pca_features[:, c], 1.0)
                        p_high = np.percentile(pca_features[:, c], 99.0)
                        if p_high > p_low:
                            pca_features[:, c] = np.clip((pca_features[:, c] - p_low) / (p_high - p_low), 0.0, 1.0)
                        else:
                            pca_features[:, c] = 0.5

                    pca_grid = pca_features[: h_patches * w_patches].reshape(h_patches, w_patches, 3)

                    # Xử lý theo chế độ hiển thị
                    if pca_mode == "foreground":
                        # PC1 thường phân tách vật thể chuyển động (xe) và nền đường
                        pc1 = pca_grid[:, :, 0]
                        if np.mean(pc1 > 0.5) > 0.5:
                            pc1 = 1.0 - pc1
                        fg_mask = (pc1 > np.percentile(pc1, 35)).astype(np.float32)
                        pca_processed = pca_grid.copy()
                        for c in range(3):
                            pca_processed[:, :, c] = pca_processed[:, :, c] * (0.25 + 0.75 * fg_mask)
                    else:
                        pca_processed = pca_grid

                    # Nội suy Bicubic cao cấp đưa lưới patch lên kích thước pixel đầy đủ (target_w, target_h)
                    pca_uint8 = (pca_processed * 255.0).clip(0, 255).astype(np.uint8)
                    pca_pil = Image.fromarray(pca_uint8)
                    pca_highres = np.array(pca_pil.resize((target_w, target_h), Image.BICUBIC)) / 255.0

                    if pca_mode == "overlay":
                        orig_norm = np.array(orig_resized) / 255.0
                        pca_render = np.clip(0.40 * orig_norm + 0.60 * pca_highres, 0.0, 1.0)
                    else:
                        pca_render = pca_highres

                # 4. Hiển thị ra các cột tương ứng
                patch_info = f"({h_patches}x{w_patches} patches)"
                if has_bg:
                    # Chế độ 4 cột: [1. Background | 2. Origin | 3. Delta Map Δ | 4. DINO PCA Map]
                    route_id, _, hour = matcher.parse_origin_filename(path)
                    if not route_id:
                        m_alt = re.search(r"(\d+)", os.path.basename(path))
                        if m_alt:
                            route_id = str(int(m_alt.group(1)))

                    bg_res = matcher.find_best_background(route_id, hour) if route_id else None
                    bg_path, bg_hour = bg_res if bg_res is not None else (None, -1)

                    # Quét dự phòng trực tiếp trong bg_dir nếu chưa tìm thấy qua index
                    if (not bg_path or not os.path.isfile(bg_path)) and route_id and bg_dir:
                        possible_bgs = (
                            glob.glob(os.path.join(bg_dir, f"*{route_id}*.*"))
                            + glob.glob(os.path.join(bg_dir, f"route_{route_id}", "*.*"))
                            + glob.glob(os.path.join(bg_dir, f"**/*{route_id}*.*"), recursive=True)
                        )
                        for pb in possible_bgs:
                            if os.path.splitext(pb)[1].lower() in TrafficPairMatcher.SUPPORTED_EXTS:
                                bg_path = pb
                                bg_hour = -1
                                break

                    if bg_path and os.path.isfile(bg_path):
                        bg_img = Image.open(bg_path).convert("RGB").resize((target_w, target_h), Image.BICUBIC)
                        delta_norm, _ = subtractor.compute_delta(orig_resized, bg_img)
                        bg_label = f"1. Background (Route {route_id} - {bg_hour}h)" if bg_hour >= 0 else f"1. Background (Route {route_id})"
                    else:
                        bg_img = Image.new("RGB", (target_w, target_h), (128, 128, 128))
                        delta_norm = np.zeros((target_h, target_w), dtype=np.float32)
                        bg_label = f"1. Background (Route {route_id} N/A)"

                    # Cột 1: Background
                    axes[idx, 0].imshow(bg_img)
                    axes[idx, 0].set_title(bg_label, fontsize=10, fontweight="bold")
                    axes[idx, 0].axis("off")

                    # Cột 2: Origin
                    axes[idx, 1].imshow(orig_resized)
                    axes[idx, 1].set_title(f"2. Origin: {os.path.basename(path)}", fontsize=10, fontweight="bold")
                    axes[idx, 1].axis("off")

                    # Cột 3: Delta Map
                    axes[idx, 2].imshow(delta_norm, cmap="inferno")
                    axes[idx, 2].set_title("3. Delta Map Δ (Trừ nền)", fontsize=10, fontweight="bold")
                    axes[idx, 2].axis("off")

                    # Cột 4: DINO PCA Map
                    if pca_render is not None:
                        axes[idx, 3].imshow(pca_render)
                        axes[idx, 3].set_title(f"4. {title_prefix} Emergent PCA {patch_info}", fontsize=10, fontweight="bold", color="#1f77b4")
                    else:
                        axes[idx, 3].imshow(orig_resized)
                        axes[idx, 3].set_title("Feature Map Fallback", fontsize=10, fontweight="bold")
                    axes[idx, 3].axis("off")
                else:
                    # Chế độ 2 cột: [Camera Frame | DINO Emergent PCA]
                    axes[idx, 0].imshow(orig_resized)
                    axes[idx, 0].set_title(f"Camera Frame: {os.path.basename(path)}", fontsize=10, fontweight="bold")
                    axes[idx, 0].axis("off")

                    if pca_render is not None:
                        axes[idx, 1].imshow(pca_render)
                        axes[idx, 1].set_title(f"{title_prefix} Emergent PCA {patch_info}", fontsize=10, fontweight="bold", color="#1f77b4")
                    else:
                        axes[idx, 1].imshow(orig_resized)
                        axes[idx, 1].set_title("Feature Map Fallback", fontsize=10, fontweight="bold")
                    axes[idx, 1].axis("off")
        except Exception as e_img:
            print(f"❌ Lỗi xử lý ảnh {path}: {e_img}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=250)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    try:
        plt.savefig(pdf_path, bbox_inches="tight")
    except Exception:
        pass
    plt.close()
    mode_text = "4 cột đối chiếu [Background, Origin, Delta, PCA]" if has_bg else "2 cột [Frame, PCA]"
    print(f"🎉 [Success] Đã lưu Emergent PCA Feature Maps ({mode_text}) thành công tại:")
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
    parser.add_argument("--bg_dir", type=str, default=None, help="Thư mục chứa ảnh background tĩnh (để xuất đủ 4 cột đối chiếu [Background | Origin | Delta Map Δ | DINO PCA Map])")
    parser.add_argument("--cam_id", "--camera_id", "--routes", "--route", "--route_id", dest="cam_id", type=str, default=None, help="Chỉ định ID camera / tuyến đường cụ thể (ví dụ: '1' hoặc danh sách '1,2,3')")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn file checkpoint đã huấn luyện (.pth)")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14", help="Tên backbone (dinov2_vits14 / dinov3_vits16)")
    parser.add_argument("--save_path", type=str, default="emergent_pca_feature_maps.png", help="Đường dẫn lưu file ảnh kết quả")
    parser.add_argument("--num_samples", type=int, default=4, help="Số lượng ảnh mẫu muốn hiển thị (mặc định 4 ảnh như báo cáo)")
    parser.add_argument("--img_size", type=int, default=518, help="Kích thước ảnh đầu vào khi suy luận (mặc định 518 cho độ nét cao chuẩn Meta DINOv2: 37x37 patches; có thể dùng 448, 672, 700)")
    parser.add_argument("--pca_mode", type=str, choices=["rgb", "foreground", "overlay"], default="rgb", help="Chế độ trực quan hóa PCA: 'rgb' (chuẩn Meta 3 thành phần chính), 'foreground' (làm nổi bật phương tiện, giảm nền), 'overlay' (phủ bán trong suốt lên ảnh gốc)")
    parser.add_argument("--keep_aspect_ratio", action="store_true", help="Giữ nguyên tỷ lệ khung hình thật của camera (ví dụ 16:9), tránh bị ép méo thành hình vuông")
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

    # 3. Tự động tìm thư mục background tĩnh nếu người dùng chưa truyền cờ --bg_dir
    bg_dir = args.bg_dir
    if not bg_dir:
        candidates = []
        if os.path.isdir(args.img_dir):
            parent_dir = os.path.dirname(os.path.abspath(args.img_dir))
            candidates.append(os.path.join(parent_dir, "traffic_backgrounds"))
            candidates.append(os.path.join(parent_dir, "backgrounds"))
            candidates.append(os.path.join(args.img_dir, "traffic_backgrounds"))
            candidates.append(os.path.join(args.img_dir, "..", "traffic_backgrounds"))
        candidates.extend([
            "traffic_backgrounds",
            "../traffic_backgrounds",
            "D:/DATN_transport-network-model/thu_thap_du_lieu/traffic_backgrounds",
        ])
        for cand in candidates:
            if os.path.isdir(cand):
                bg_dir = cand
                print(f"💡 [Tự động phát hiện] Tìm thấy thư mục ảnh nền tĩnh tại: {cand}")
                break

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
        bg_dir=bg_dir,
        save_path=args.save_path,
        device=device,
        img_size=args.img_size,
        patch_size=patch_size,
        pca_mode=args.pca_mode,
        keep_aspect_ratio=args.keep_aspect_ratio,
        title_prefix=title,
    )
