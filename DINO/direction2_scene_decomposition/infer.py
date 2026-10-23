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

from typing import Dict, List, Optional, Tuple, Union
import cv2
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


def split_touching_instances(
    bin_clean: np.ndarray,
    watershed_thresh: float = 0.55,
    min_seed_area: int = 4,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Tách các phương tiện bị dính chùm vào nhau (Touching / Overlapping Vehicles)
    bằng biến đổi khoảng cách thích ứng theo từng cụm (Component-wise Distance Transform)
    kết hợp phân vùng lưu vực Watershed:
      - Tự động phát hiện các đỉnh cực đại khoảng cách cục bộ (Local Distance Peaks)
      - Phân lập các xe máy/ô tô đứng sát nhau thành các nhãn cá thể (Instance Labels) độc lập
      - Tạo rãnh phân tách 1-pixel tinh tế giữa các xe tiếp xúc để vẽ viền và tô màu riêng biệt

    Trả về:
      - instance_labels (H, W): Bản đồ nhãn cá thể (0 là nền, 1..N là từng xe)
      - total_instances: Tổng số xe phát hiện được sau khi tách
      - bin_clean_separated: Mặt nạ nhị phân đã được cắt rãnh 1px giữa các xe chạm nhau
    """
    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(bin_clean)
    if num_cc <= 1:
        return np.zeros_like(bin_clean, dtype=np.int32), 0, bin_clean.copy()

    dist = cv2.distanceTransform(bin_clean, cv2.DIST_L2, 5)
    markers = np.zeros(bin_clean.shape, dtype=np.int32)
    current_marker = 1
    split_occurred = False

    for lbl in range(1, num_cc):
        comp_mask = (cc_labels == lbl)
        local_dist = dist * comp_mask
        max_d = float(local_dist.max())

        # Nếu phương tiện nhỏ (ví dụ xe máy ở xa), bán kính max < 5px thì giữ nguyên là 1 instance
        if max_d < 5.0:
            markers[comp_mask] = current_marker
            current_marker += 1
            continue

        # Tìm các hạt giống (seeds) bằng cực đại cục bộ kết hợp ngưỡng khoảng cách
        # Giúp bắt trọn cả xe máy nhỏ (dist ~ 3-4px) nằm sát cạnh ô tô lớn (dist ~ 20-30px)
        local_max = cv2.dilate(local_dist, np.ones((5, 5), np.uint8))
        is_peak = (local_dist == local_max) & (local_dist >= 2.0)
        local_seeds_peak = is_peak.astype(np.uint8) * 255
        local_seeds_thresh = ((local_dist > (watershed_thresh * max_d)) * 255).astype(np.uint8)
        local_seeds = cv2.bitwise_or(local_seeds_peak, local_seeds_thresh)
        n_seeds, seed_lbls = cv2.connectedComponents(local_seeds)

        if n_seeds <= 2:
            # Chỉ có 1 đỉnh duy nhất (xe độc lập không bị dính)
            markers[comp_mask] = current_marker
            current_marker += 1
        else:
            # Có nhiều hơn 1 đỉnh: Cụm này gồm 2 hoặc nhiều xe dính vào nhau!
            split_occurred = True
            for s_id in range(1, n_seeds):
                s_mask = (seed_lbls == s_id)
                if np.sum(s_mask) >= min_seed_area:
                    markers[s_mask] = current_marker
                    current_marker += 1

    total_instances = current_marker - 1

    # Nếu không có cụm nào bị dính chùm, giữ nguyên nhãn ban đầu
    if not split_occurred:
        return cc_labels, num_cc - 1, bin_clean.copy()

    # Áp dụng Watershed để mở rộng hạt giống ra toàn bộ diện tích của từng xe
    bg_marker = current_marker
    markers[bin_clean == 0] = bg_marker
    norm_dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cost_map = cv2.cvtColor(255 - norm_dist, cv2.COLOR_GRAY2BGR)
    cv2.watershed(cost_map, markers)

    instance_labels = np.zeros_like(bin_clean, dtype=np.int32)
    for m in range(1, current_marker):
        instance_labels[(markers == m) & (bin_clean > 0)] = m

    # Cắt rãnh phân tách 1px tại biên giới giữa các xe chạm nhau (markers == -1)
    bin_clean_separated = bin_clean.copy()
    boundary_slit = (markers == -1) & (bin_clean > 0)
    bin_clean_separated[boundary_slit] = 0

    return instance_labels, total_instances, bin_clean_separated


def create_segmentation_overlay(
    origin_pil: Image.Image,
    mask_arr: np.ndarray,
    threshold: float = 0.25,
    mode: str = "multi",  # 'multi' (mỗi xe một màu rực rỡ như YOLO/Mask R-CNN) hoặc 'single'
    single_color: Tuple[int, int, int] = (0, 230, 118),
    draw_contours: bool = True,
    contour_color: Tuple[int, int, int] = (255, 255, 255),  # Viền trắng nét thanh dạ quang
    contour_thickness: int = 1,
    min_area: int = 36,
    close_kernel: int = 3,       # Hệ số gộp/lan truyền mask (0 hoặc 1 để tắt hoàn toàn gộp)
    open_kernel: int = 3,        # Bẻ gãy các cầu nối mảnh giữa các xe (0 hoặc 1 để tắt)
    split_instances: bool = True,# Tách các xe bị chạm dính nhau bằng Watershed
    watershed_thresh: float = 0.55, # Ngưỡng phát hiện tâm xe (0.35 - 0.75; tăng để tách gắt)
    banner_ratio: float = 0.05,  # Tỷ lệ chiều cao dải timestamp trên cùng (0.0 = tắt)
) -> Tuple[Image.Image, Image.Image]:
    """
    Tạo ảnh phân đoạn phương tiện vẽ trực tiếp lên ảnh gốc chuẩn Mask R-CNN / YOLO-Seg:
      - Ngưỡng mềm (Soft Ramp) kết hợp Gaussian filter khử cạnh sắc cứng
      - Lọc hình thái học linh hoạt (Open bẻ cầu nối, Close điều khiển hệ số lan truyền/gộp)
      - Lấp đầy lỗ thủng bên trong thân xe bằng RETR_CCOMP (không làm dính khe hở giữa các xe)
      - Lọc bỏ dải timestamp camera và các vạch sơn kẻ đường mảnh
      - Tách cá thể phương tiện (Instance Splitting) bằng Watershed & Distance Transform
      - Phân cụm đối tượng đa sắc (Multi-color instance-like) và vẽ đường bao ôm khít từng xe
    """
    orig_np = np.array(origin_pil)  # uint8 (H, W, 3)
    orig_h, orig_w = orig_np.shape[:2]

    # Đồng bộ kích thước mask với ảnh gốc
    if mask_arr.shape != (orig_h, orig_w):
        mask_pil = Image.fromarray((np.clip(mask_arr, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
        norm_mask = np.array(mask_pil).astype(np.float32) / 255.0
    else:
        norm_mask = np.clip(mask_arr, 0.0, 1.0).astype(np.float32)

    # Khử vùng banner timestamp trên cùng nếu được chỉ định
    if banner_ratio > 0.0:
        banner_h = max(1, int(orig_h * banner_ratio))
        norm_mask[:banner_h, :] = 0.0

    # 1. Làm mượt bằng Gaussian Blur để triệt tiêu răng cưa viền cắt
    mask_blur = cv2.GaussianBlur(norm_mask, (5, 5), 1.2)

    # 2. Xử lý nhị phân
    bin_mask = ((mask_blur > threshold) * 255).astype(np.uint8)

    # 3. Lọc hình thái học:
    # Kernel Mở (Open) để bẻ gãy các cầu nối mảnh giữa các xe và xóa nhiễu đốm
    if open_kernel > 1:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel_open)

    # Kernel Đóng (Close) - Hệ số gộp / lan truyền mask gần nhau
    # Đặt nhỏ (3 hoặc 0/1) để tránh hàn dính các xe đi sát nhau
    # Đặt lớn (7, 9) nếu muốn các mask gần nhau tự động lan truyền gộp lại thành 1 khối
    if close_kernel > 1:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel_close)

    # 4. Lấp đầy lỗ thủng bên trong thân xe (kính lái, mui phản quang) bằng RETR_CCOMP
    # Không dùng RETR_EXTERNAL FILLED để tránh hàn kín khe hở giữa 2 xe cạnh nhau
    contours_hier, hierarchy = cv2.findContours(bin_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    bin_clean = bin_mask.copy()
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            # h[3] != -1: đây là lỗ rỗng nằm lọt thỏm hoàn toàn bên trong 1 contour xe khác
            if h[3] != -1:
                hole_area = cv2.contourArea(contours_hier[i])
                if hole_area < 4000:
                    cv2.drawContours(bin_clean, contours_hier, i, 255, thickness=cv2.FILLED)

    # 5. Loại bỏ các vạch kẻ đường mảnh (elongation > 6.5 và short_side < 6.0) hoặc diện tích quá bé
    contours_ext, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bin_filtered = np.zeros_like(bin_clean)
    for cnt in contours_ext:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        w_r, h_r = rect[1]
        short_side = max(min(w_r, h_r), 1.0)
        long_side = max(w_r, h_r)
        elongation = long_side / short_side
        if elongation > 6.5 and short_side < 6.0:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = max(cv2.contourArea(hull), 1.0)
        solidity = area / hull_area
        if solidity < 0.20 and area < min_area * 2.5:
            continue
        cv2.drawContours(bin_filtered, [cnt], -1, 255, thickness=cv2.FILLED)
    bin_clean = bin_filtered

    # 6. Tách cá thể phương tiện (Instance Separation)
    if split_instances:
        instance_labels, num_instances, bin_clean = split_touching_instances(
            bin_clean=bin_clean,
            watershed_thresh=watershed_thresh,
            min_seed_area=4,
        )
    else:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_clean)
        instance_labels = labels
        num_instances = num_labels - 1

    # Bảng màu rực rỡ phong cách Instance Segmentation (YOLO-Seg / SAM)
    palette = [
        (0, 230, 118),   # Xanh ngọc lục bảo (Emerald Green)
        (255, 112, 67),  # Cam san hô (Coral Orange)
        (41, 182, 246),  # Xanh dương sáng (Light Blue)
        (255, 64, 129),  # Hồng dạ quang (Rose Pink)
        (171, 71, 188),  # Tím tử đinh hương (Purple)
        (255, 238, 88),  # Vàng chanh (Bright Yellow)
        (38, 198, 218),  # Xanh ngọc lam (Cyan)
        (255, 87, 34),   # Đỏ cam (Deep Orange)
        (102, 187, 106), # Xanh lá tươi (Lime)
    ]

    overlay_canvas = orig_np.astype(np.float32).copy()

    if mode == "multi" and num_instances > 0:
        for inst_id in range(1, num_instances + 1):
            comp_mask = (instance_labels == inst_id).astype(np.float32)
            if np.sum(comp_mask) < min_area:
                continue
            soft_ramp = np.clip((mask_blur - (threshold * 0.6)) / (threshold * 1.2 + 1e-4), 0.35, 1.0)
            alpha_layer = (comp_mask * soft_ramp * 0.46)[:, :, None]

            color_np = np.array(palette[(inst_id - 1) % len(palette)], dtype=np.float32)
            overlay_canvas = overlay_canvas * (1.0 - alpha_layer) + color_np * alpha_layer
    else:
        soft_ramp = np.clip((mask_blur - (threshold * 0.6)) / (threshold * 1.2 + 1e-4), 0.30, 1.0)
        bin_f = (bin_clean > 0).astype(np.float32)
        alpha_layer = (bin_f * soft_ramp * 0.48)[:, :, None]
        color_np = np.array(single_color, dtype=np.float32)
        overlay_canvas = overlay_canvas * (1.0 - alpha_layer) + color_np * alpha_layer

    overlay_final = np.clip(overlay_canvas, 0, 255).astype(np.uint8)

    # 7. Vẽ đường viền biên dạng (Contours) ôm khít từng xe độc lập
    if draw_contours:
        if split_instances and num_instances > 0:
            for inst_id in range(1, num_instances + 1):
                inst_bin = ((instance_labels == inst_id) * 255).astype(np.uint8)
                cnts, _ = cv2.findContours(inst_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    if cv2.contourArea(cnt) >= min_area:
                        cv2.drawContours(overlay_final, [cnt], -1, contour_color, contour_thickness, cv2.LINE_AA)
        else:
            contours, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
            cv2.drawContours(overlay_final, valid_contours, -1, contour_color, contour_thickness, cv2.LINE_AA)

    overlay_img = Image.fromarray(overlay_final)
    binary_img = Image.fromarray(bin_clean)
    return overlay_img, binary_img


def save_composite_figure(
    origin_pil: Image.Image,
    pred_bg_pil: Image.Image,
    pred_fg_pil: Image.Image,
    pred_mask_pil: Image.Image,
    seg_overlay_pil: Image.Image,
    binary_mask_pil: Image.Image,
    save_path: str,
    prior_bg_pil: Image.Image = None,
    recon_pil: Image.Image = None,
    title_text: str = "Traffic-Decompose Layer Separation & Vehicle Segmentation",
):
    """Xuất ảnh so sánh trực quan đa bảng (tích hợp đầy đủ các lớp bóc tách và phân đoạn xe)."""
    has_prior = (prior_bg_pil is not None)
    cols = 4 if has_prior else 3
    fig, axes = plt.subplots(2, cols, figsize=(19 if has_prior else 14, 9), dpi=150)
    plt.subplots_adjust(wspace=0.18, hspace=0.25)

    if has_prior:
        # Hàng 1: Các góc nhìn bối cảnh và lớp phủ phân đoạn
        axes[0, 0].imshow(origin_pil)
        axes[0, 0].set_title("(a) Input Origin Image", fontsize=10, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(prior_bg_pil)
        axes[0, 1].set_title("(b) Background Prior (Reference)", fontsize=10, fontweight="bold")
        axes[0, 1].axis("off")

        if recon_pil is not None:
            axes[0, 2].imshow(recon_pil)
            axes[0, 2].set_title("(c) Reconstructed Origin", fontsize=10, fontweight="bold")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(seg_overlay_pil)
        axes[0, 3].set_title("(d) Vehicle Segmentation Overlay", fontsize=10, fontweight="bold", color="darkgreen")
        axes[0, 3].axis("off")

        # Hàng 2: Các lớp thành phần phân rã và mặt nạ
        axes[1, 0].imshow(pred_bg_pil)
        axes[1, 0].set_title("(e) Decomposed Clean Road", fontsize=10, fontweight="bold")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(pred_fg_pil)
        axes[1, 1].set_title("(f) Decomposed Vehicles Layer", fontsize=10, fontweight="bold")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(pred_mask_pil, cmap="magma")
        axes[1, 2].set_title("(g) Continuous Alpha Heatmap", fontsize=10, fontweight="bold")
        axes[1, 2].axis("off")

        axes[1, 3].imshow(binary_mask_pil, cmap="gray")
        axes[1, 3].set_title("(h) Binary Vehicle Mask", fontsize=10, fontweight="bold", color="navy")
        axes[1, 3].axis("off")
    else:
        # Chế độ 6 bảng khi chạy Single-frame
        axes[0, 0].imshow(origin_pil)
        axes[0, 0].set_title("(a) Input Origin Image", fontsize=10, fontweight="bold")
        axes[0, 0].axis("off")

        if recon_pil is not None:
            axes[0, 1].imshow(recon_pil)
            axes[0, 1].set_title("(b) Reconstructed Origin", fontsize=10, fontweight="bold")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(seg_overlay_pil)
        axes[0, 2].set_title("(c) Vehicle Segmentation Overlay", fontsize=10, fontweight="bold", color="darkgreen")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(pred_bg_pil)
        axes[1, 0].set_title("(d) Decomposed Clean Road", fontsize=10, fontweight="bold")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(pred_fg_pil)
        axes[1, 1].set_title("(e) Decomposed Vehicles Layer", fontsize=10, fontweight="bold")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(binary_mask_pil, cmap="gray")
        axes[1, 2].set_title("(f) Binary Vehicle Mask", fontsize=10, fontweight="bold", color="navy")
        axes[1, 2].axis("off")

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
    # Tự động phát hiện nếu checkpoint chứa weights của backbone thì không cần bắt buộc tải lại từ Hub
    has_custom = (args.weights is not None and os.path.isfile(args.weights))
    backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.backbone,
        pretrained=not has_custom,
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

        # Tạo ảnh Segmentation Overlay (phủ màu trực tiếp lên ảnh gốc) & Mặt nạ nhị phân đã lọc sạch
        seg_overlay_img, bin_mask_img = create_segmentation_overlay(
            origin_pil=pil_img,
            mask_arr=pred_mask_arr,
            threshold=args.seg_thresh,
            mode=args.seg_mode,
            close_kernel=args.close_kernel,
            open_kernel=args.open_kernel,
            split_instances=args.split_instances,
            watershed_thresh=args.watershed_thresh,
            banner_ratio=args.banner_ratio,
            min_area=args.min_area,
            contour_thickness=args.contour_thickness,
        )

        pred_bg_img = Image.fromarray((np.clip(pred_bg_arr, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BICUBIC)
        pred_mask_img = Image.fromarray((np.clip(pred_mask_arr, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)

        # Lớp phương tiện tách biệt (Decomposed Vehicles Layer): Chỉ giữ lại các xe trên nền tối sạch
        fg_full_np = np.array(Image.fromarray((np.clip(pred_fg_arr, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BICUBIC)).astype(np.float32) / 255.0
        alpha_full_np = np.array(pred_mask_img).astype(np.float32) / 255.0
        bin_clean_np = (np.array(bin_mask_img).astype(np.float32) / 255.0)
        # Kết hợp mặt nạ mềm và mặt nạ nhị phân đã khử nhiễu để tách xe sắc nét
        fg_matte = np.clip(np.maximum(bin_clean_np, alpha_full_np * 1.2) * ((alpha_full_np > (args.seg_thresh * 0.65)).astype(np.float32)), 0.0, 1.0)
        fg_matte = cv2.GaussianBlur(fg_matte, (5, 5), 1.0)[:, :, None]
        isolated_fg_np = fg_full_np * fg_matte + 0.05 * (1.0 - fg_matte)
        pred_fg_img = Image.fromarray((np.clip(isolated_fg_np, 0, 1) * 255).astype(np.uint8))

        recon_pil_img = None
        if "recon_origin" in preds:
            recon_arr = preds["recon_origin"].squeeze().cpu().permute(1, 2, 0).numpy()
            recon_pil_img = Image.fromarray((np.clip(recon_arr, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BICUBIC)

        # 8. Lưu kết quả các lớp đơn lẻ
        pred_bg_img.save(os.path.join(args.output_dir, f"{stem}_clean_road.jpg"))
        pred_fg_img.save(os.path.join(args.output_dir, f"{stem}_vehicles_only.jpg"))
        pred_mask_img.save(os.path.join(args.output_dir, f"{stem}_alpha_mask.png"))
        seg_overlay_img.save(os.path.join(args.output_dir, f"{stem}_segmentation_overlay.jpg"))
        bin_mask_img.save(os.path.join(args.output_dir, f"{stem}_binary_segmentation.png"))

        # Bản đồ độ bất định sigma (nếu có)
        if "sigma" in preds:
            pred_sigma_arr = preds["sigma"].squeeze().cpu().numpy()
            pred_sigma_img = Image.fromarray((np.clip(pred_sigma_arr / 0.50, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
            pred_sigma_img.save(os.path.join(args.output_dir, f"{stem}_uncertainty_sigma.png"))

        # 9. Xuất ảnh composite so sánh trực quan đa bảng (tích hợp đầy đủ Segmentation)
        if args.save_composite:
            composite_path = os.path.join(args.output_dir, f"{stem}_composite.png")
            prior_resized = prior_pil_img.resize((orig_w, orig_h), Image.BICUBIC) if prior_pil_img else None
            save_composite_figure(
                origin_pil=pil_img,
                pred_bg_pil=pred_bg_img,
                pred_fg_pil=pred_fg_img,
                pred_mask_pil=pred_mask_img,
                seg_overlay_pil=seg_overlay_img,
                binary_mask_pil=bin_mask_img,
                save_path=composite_path,
                prior_bg_pil=prior_resized,
                recon_pil=recon_pil_img,
                title_text=f"Traffic-Decompose Inference — {stem}",
            )

    print(f"\n✅ [Hoàn tất] Toàn bộ kết quả phân rã & phân đoạn xe đã được lưu tại: {args.output_dir}")


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
    parser.add_argument("--seg_thresh", type=float, default=0.25, help="Ngưỡng nhị phân hóa mặt nạ phương tiện (mặc định: 0.25; tăng lên 0.3-0.35 nếu xe bị dính bóng râm/shadow)")
    parser.add_argument("--close_kernel", type=int, default=3, help="Hệ số gộp/lan truyền mask gần nhau (Morphological Close Kernel, mặc định: 3; đặt 0 hoặc 1 để tắt hoàn toàn việc gộp; tăng lên 7, 9 nếu muốn gộp các mask gần nhau thành 1 khối)")
    parser.add_argument("--open_kernel", type=int, default=3, help="Kích thước kernel phép Mở (Morphological Open) bẻ gãy cầu nối mảnh giữa các xe (mặc định: 3; đặt 0 hoặc 1 để tắt)")
    parser.add_argument("--split_instances", action="store_true", default=True, help="Tự động phân tách các xe bị chạm dính nhau (Instance Splitting via Watershed & Distance Transform)")
    parser.add_argument("--no_split_instances", action="store_false", dest="split_instances", help="Tắt chế độ tự động tách cá thể (giữ nguyên cụm dính)")
    parser.add_argument("--watershed_thresh", type=float, default=0.55, help="Ngưỡng tìm tâm hạt giống phương tiện trên bản đồ khoảng cách (mặc định: 0.55; khoảng giá trị 0.35 - 0.75; tăng lên 0.65-0.75 để tách gắt các xe đi sát nhau)")
    parser.add_argument("--min_area", type=int, default=36, help="Diện tích tối thiểu (pixel) của mask phương tiện để lọc nhiễu (mặc định: 36)")
    parser.add_argument("--banner_ratio", type=float, default=0.05, help="Tỷ lệ chiều cao dải timestamp trên cùng để khử nhiễu (mặc định: 0.05; đặt 0.0 để tắt hoàn toàn)")
    parser.add_argument("--contour_thickness", type=int, default=1, help="Độ dày đường viền vẽ quanh từng xe (mặc định: 1)")
    parser.add_argument("--seg_mode", type=str, default="multi", choices=["multi", "single"], help="Chế độ màu phân đoạn: 'multi' (đa sắc từng xe kiểu YOLO/Mask R-CNN) hoặc 'single' (đơn sắc)")
    parser.add_argument("--save_composite", action="store_true", default=True, help="Lưu thêm ảnh ghép so sánh trực quan đa bảng (composite)")
    parser.add_argument("--no_composite", action="store_false", dest="save_composite", help="Không lưu ảnh composite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị ('cuda' hoặc 'cpu')")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face user access token (nếu dùng Meta DINOv3 đóng)")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_inference(cli_args)

