"""
=============================================================================
 Hướng 1 Mới (Direction 1 New): Khởi động thống kê TAM & Tiền xử lý Cache
 Gồm warmup_tam_offline và resolve_or_precompute_tam_cache
=============================================================================
"""

import os
import sys
import time
from collections import defaultdict
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

_dino_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from common.backbone_loader import extract_tokens, imagenet_normalize
from common.gpu_utils import unwrap_model
from direction1_new.core.features import FrozenExtractor
from direction1_new.core.tam import PositionStats, GMMCalibrator


def resolve_or_precompute_tam_cache(
    data_dir: Optional[str],
    requested_cache_dir: Optional[str] = None,
    output_dir: str = "checkpoints/direction1_new",
    model_name: str = "dinov3_vits16",
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    hf_token: str = "",
    tam_warmup_frames: Optional[int] = None,
    gmm_buffer_size: int = 300000,
    gmm_ema_alpha: float = 0.15,
) -> Optional[str]:
    """
    Tự động tìm kiếm hoặc kích hoạt tiền xử lý bộ nhớ đệm TAM pi (tam_cache):
    1. Quét các vị trí tiềm năng (requested_cache_dir, tam_cache, data_dir/../tam_cache, output_dir/tam_cache).
    2. Nếu tìm thấy thư mục chứa file .npy hợp lệ: tự động nạp trực tiếp.
    3. Nếu chưa có: tự động chạy precompute_tam_cache để trích xuất trước toàn bộ pi rồi nạp vào.
    """
    if not data_dir or not os.path.isdir(data_dir):
        return None

    data_parent = os.path.dirname(os.path.abspath(data_dir))
    candidate_dirs = []
    if requested_cache_dir:
        candidate_dirs.append(requested_cache_dir)
    candidate_dirs.extend([
        os.path.join(data_parent, "tam_cache"),
        os.path.join(output_dir, "tam_cache"),
        os.path.abspath("tam_cache"),
        "/workspace/tam_cache",
    ])

    # 1. Tìm xem có thư mục nào đã chứa cache hoàn chỉnh không
    for c_dir in candidate_dirs:
        if os.path.isdir(c_dir):
            meta_file = os.path.join(c_dir, "cache_meta.json")
            has_cache = False
            if os.path.isfile(meta_file):
                has_cache = True
            else:
                for root, _, files in os.walk(c_dir):
                    if any(f.endswith(".npy") for f in files):
                        has_cache = True
                        break
            if has_cache:
                print(f"⚡ [TAM Cache Auto-Detect] Đã tìm thấy bộ nhớ đệm TAM tại: '{c_dir}'. Tự động nạp trực tiếp!")
                return c_dir

    # 2. Nếu chưa có bất kỳ cache nào, tự động kích hoạt tiền xử lý precompute
    chosen_cache_dir = requested_cache_dir or os.path.join(data_parent, "tam_cache")
    print("\n" + "=" * 80)
    print(" 🔍 [TAM Cache Auto-Detect] Chưa tìm thấy bộ nhớ đệm TAM pi.")
    print(f" 🚀 Tự động kích hoạt tiền xử lý (precompute_tam_cache) vào: '{chosen_cache_dir}'")
    print("=" * 80)

    try:
        from direction1_new.precompute_tam import precompute_tam_cache
        precompute_tam_cache(
            data_dir=data_dir,
            cache_dir=chosen_cache_dir,
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            device_str=device,
            hf_token=hf_token,
            tam_warmup_frames=tam_warmup_frames,
            gmm_buffer_size=gmm_buffer_size,
            gmm_ema_alpha=gmm_ema_alpha,
        )
        return chosen_cache_dir
    except Exception as e:
        print(f"⚠️ [TAM Cache Precompute] Không thể tự động tính trước cache ({e}). Chuyển sang trích xuất online...")
        return None


def warmup_tam_offline(
    frozen_extractor: Union[nn.Module, FrozenExtractor],
    pos_stats: PositionStats,
    calibrator: GMMCalibrator,
    dataset: Dataset,
    device: torch.device,
    num_frames_per_cam: int = 200,
    kmeans_iters: int = 20,
    gmm_iters: int = 30,
    batch_size: int = 8,
):
    """
    Khởi động TAM (Temporal Atypicality Map) trước khi bước vào huấn luyện chính thức:
    1. Trích xuất tokens đa camera để fit ma trận chiếu PCA tối ưu (embed_dim -> 64) cho FrozenExtractor.
    2. Chạy offline K-means cosine (K=4) cho từng camera để thiết lập tâm cụm mu, trọng số w và độ phân tán s thực tế.
    3. Thu thập các giá trị atypicality ban đầu theo hạn ngạch cân bằng (Equal Quota) qua 100% camera,
       sau đó hiệu chuẩn phân phối GMM (mu_bg, mu_fg, std_bg, std_fg).
    """
    print("\n" + "=" * 80)
    print(" 🛠️  [TAM WARMUP] BẮT ĐẦU KHỞI ĐỘNG THỐNG KÊ KHÔNG-THỜI GIAN TAM OFFLINE")
    print("=" * 80)
    start_t = time.time()
    raw_fe = unwrap_model(frozen_extractor)

    # Chuẩn bị danh sách ảnh theo từng camera với phân tầng thời gian trải rộng
    cam_images = defaultdict(list)
    if hasattr(dataset, "cam_to_files") and hasattr(dataset, "cam_to_id"):
        for cam_key, files in dataset.cam_to_files.items():
            cid = dataset.cam_to_id[cam_key]
            n_f = len(files)
            if n_f <= num_frames_per_cam:
                subset = list(files)
            else:
                indices = np.linspace(0, n_f - 1, num_frames_per_cam, dtype=int)
                subset = [files[i] for i in indices]
            cam_images[cid] = subset
    else:
        for idx in range(min(len(dataset), num_frames_per_cam * 16)):
            item = dataset[idx]
            cid = int(item["cid"])
            if len(cam_images[cid]) < num_frames_per_cam:
                cam_images[cid].append(item["x1"])

    num_active_cams = len(cam_images)
    print(f" 📹 Số camera phát hiện cho Warmup: {num_active_cams} camera (tối đa {num_frames_per_cam} frame/cam)")
    pos_stats.ensure_cam_capacity(max(cam_images.keys(), default=0))

    # GIAI ĐOẠN 1: FIT PCA CHO FROZENEXTRACTOR
    if not getattr(raw_fe.is_pca_fitted, "item", lambda: False)():
        print(" 🔍 [Giai Đoạn 1/3] Trích xuất patch tokens để Fit PCA ma trận chiếu 64 chiều...")
        all_sample_tokens = []
        raw_fe.backbone.eval()

        for cid, img_list in cam_images.items():
            sample_imgs = img_list[:10]
            tensors = []
            for item in sample_imgs:
                if isinstance(item, str):
                    tensors.append(dataset._load_img(item))
                else:
                    tensors.append(item)
            if not tensors:
                continue

            batch_x = torch.stack(tensors, dim=0).to(device)
            target_H = 16 * raw_fe.patch_size
            target_W = 28 * raw_fe.patch_size
            if batch_x.shape[-2] != target_H or batch_x.shape[-1] != target_W:
                batch_x = F.interpolate(batch_x, size=(target_H, target_W), mode="bilinear", align_corners=False)

            with torch.no_grad():
                x_norm = imagenet_normalize(batch_x)
                _, patch_spatial = extract_tokens(raw_fe.backbone, x_norm, patch_size=raw_fe.patch_size)
                B_cur, Hp, Wp, D_cur = patch_spatial.shape
                tokens_flat = patch_spatial.reshape(B_cur * Hp * Wp, D_cur)
                if tokens_flat.shape[0] > 1000:
                    perm = torch.randperm(tokens_flat.shape[0], device=device)[:1000]
                    tokens_flat = tokens_flat[perm]
                all_sample_tokens.append(tokens_flat.cpu())

        if all_sample_tokens:
            collected_tokens = torch.cat(all_sample_tokens, dim=0)
            if collected_tokens.shape[0] > 30000:
                perm_all = torch.randperm(collected_tokens.shape[0])[:30000]
                collected_tokens = collected_tokens[perm_all]
            print(f"    -> Đã thu thập {collected_tokens.shape[0]} tokens đặc trưng. Đang tính phân rã PCA...")
            raw_fe.fit_pca(collected_tokens.to(device))
        else:
            print("    ⚠️ Không thu thập được tokens, sử dụng ma trận trực giao ban đầu.")
    else:
        print(" ⏩ [Giai Đoạn 1/3] Ma trận chiếu PCA đã được fit từ trước -> Bỏ qua.")

    # GIAI ĐOẠN 2: KHỞI TẠO K-MEANS CHO TỪNG CAMERA VỚI EQUAL QUOTA BUFFER SAMPLING
    print(f" 🎯 [Giai Đoạn 2/3] Chạy K-means cosine (K={pos_stats.num_states}, iters={kmeans_iters}) cho từng camera...")
    quota_per_cam = max(100, int(np.ceil(calibrator.max_buffer_size / max(1, num_active_cams))))
    print(f"    -> Cơ chế Equal Quota Sampling: Tối đa {quota_per_cam} mẫu/cam (đảm bảo cân bằng qua toàn bộ {num_active_cams} camera)")

    cams_initialized = 0
    cams_skipped = 0
    for cid, img_list in cam_images.items():
        if len(img_list) == 0:
            continue
        if cid < pos_stats.num_cams and bool(pos_stats.is_initialized[cid].item()):
            cams_skipped += 1
            continue

        cam_tensors = []
        for item in img_list:
            if isinstance(item, str):
                cam_tensors.append(dataset._load_img(item))
            else:
                cam_tensors.append(item)

        all_u = []
        with torch.no_grad():
            for i in range(0, len(cam_tensors), batch_size):
                sub_batch = torch.stack(cam_tensors[i:i+batch_size], dim=0).to(device)
                u_sub = raw_fe(sub_batch)  # (B_sub, 448, 64)
                all_u.append(u_sub)
        U_cam = torch.cat(all_u, dim=0)  # (N_cam, 448, 64)

        pos_stats.init_kmeans(cid, U_cam, n_iters=kmeans_iters)
        cams_initialized += 1

        with torch.no_grad():
            cids_tensor = torch.full((U_cam.shape[0],), cid, dtype=torch.long, device=device)
            a_cam, valid_cam = pos_stats.atypicality(cids_tensor, U_cam)
            calibrator.push(a_cam, valid_cam, max_samples=quota_per_cam)

    print(f"    -> Đã khởi tạo K-means mới cho {cams_initialized} camera (giữ nguyên {cams_skipped} camera đã có sẵn).")

    # GIAI ĐOẠN 3: HIỆU CHUẨN SỚM PHÂN PHỐI GMM
    current_samples = getattr(calibrator, "size", len(calibrator.buffer))
    print(f" 📊 [Giai Đoạn 3/3] Hiệu chuẩn GMM 2 thành phần (EM iters={gmm_iters}) từ buffer ({current_samples} mẫu đại diện)...")
    calibrator.refit(max_iters=gmm_iters)
    if calibrator.is_fitted:
        print(f"    ✅ GMM Hiệu chuẩn thành công: mu_bg={calibrator.mu_bg:.3f}, mu_fg={calibrator.mu_fg:.3f}, "
              f"std_bg={calibrator.std_bg:.3f}, std_fg={calibrator.std_fg:.3f}, pi_weight={calibrator.pi_weight:.3f}")
    else:
        print("    ℹ️ Cần thêm mẫu atypicality trong quá trình train để hoàn tất GMM.")

    elapsed = time.time() - start_t
    print(f" ✅ [TAM WARMUP HOÀN TẤT] Thời gian thực thi: {elapsed:.2f}s")
    print("=" * 80 + "\n")
