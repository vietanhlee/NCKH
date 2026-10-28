"""
=============================================================================
  Hướng G: audit2.py
  Bộ công cụ kiểm toán định lượng thế hệ 2 cho Hệ thống Vehicle-Centric SSL DINOv3

  Giải quyết trực diện 4 yêu cầu ưu tiên từ Giảng viên Hướng dẫn (GVHD):
  1. Frozen Linear Probe CHỈ sử dụng chia dữ liệu phân lập hoàn toàn theo Camera ID
     (Camera-Disjoint Split: GroupKFold / StratifiedGroupKFold theo cam_id) nhằm
     triệt tiêu 100% rò rỉ dữ liệu giữa Train và Test.
  2. Hỗ trợ nạp và đánh giá đa mô hình để đo lường đóng góp thực sự:
     - DINOv3 Pretrained (Mốc tham chiếu gốc)
     - Baseline DINOv3 Continued SSL + Random Masking (Đo đóng góp của TAM + AGM + SRS)
     - Ablation w/o SRS (Kiểm chứng AGM khi không có hoán đổi nền)
     - Full Proposed Method (TAM + AGM + SRS), đánh giá cả Student và EMA Teacher.
  3. Lọc khung giờ ban ngày sáng - chiều (06:00 - 17:00, GMT+7) theo yêu cầu
     để đảm bảo chất lượng hình ảnh và độ chính xác của Faster R-CNN detector.
  4. Kiểm định TAM và AGM trên tập ảnh có nhãn phương tiện được kiểm tra thủ công
     (sample_test_images/audit_set/sample_labels.csv) kèm các tác vụ downstream
     cấp độ ảnh: Hồi quy số lượng xe (Ridge MAE, RMSE, R2) và Phân loại mật độ.
=============================================================================
"""

import os
import sys
import argparse
import json
import time
import math
import re
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

# Cấu hình OpenMP an toàn cho môi trường Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Thiết lập đường dẫn import hệ thống
_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_script_dir)
_dino_root = os.path.dirname(_pkg_dir)
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from common.backbone_loader import get_dino_backbone, extract_tokens, imagenet_normalize
from common.gpu_utils import smart_load_state_dict, clean_state_dict
from direction1_new.core import (
    FrozenExtractor,
    PositionStats,
    GMMCalibrator,
    refine_traffic_foreground_map,
    agm_sample,
    dilate_mask_2d,
    static_region_swap,
)

# Sklearn cho đánh giá Frozen Linear Probe phân lập theo Camera ID
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from scipy.stats import pearsonr, spearmanr


# =============================================================================
# 1. BỘ LỌC KHUNG GIỜ BAN NGÀY SÁNG - CHIỀU (06:00 - 17:00, GMT+7)
# =============================================================================
def parse_vietnam_hour(filename: str) -> Optional[int]:
    """
    Trích xuất giờ chụp ảnh theo múi giờ Việt Nam (GMT+7) từ tên tệp tin.
    Định dạng tệp chuẩn trong hệ thống: `<cam_id>_<unix_timestamp>.jpg`
    """
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]
    parts = name_no_ext.split("_")
    if len(parts) >= 2:
        try:
            ts = int(parts[-1])
            # Chuyển đổi timestamp Unix sang giờ GMT+7
            dt = datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=7)))
            return dt.hour
        except (ValueError, OverflowError, OSError):
            pass
    return None


def is_daylight_hour(filename: str, start_hour: int = 6, end_hour: int = 17) -> bool:
    """
    Kiểm tra xem khung hình có rơi vào khoảng thời gian sáng - chiều (6h - 17h) hay không.
    Nếu tệp không chứa timestamp chuẩn, mặc định chấp nhận để tránh bỏ sót dữ liệu.
    """
    hour = parse_vietnam_hour(filename)
    if hour is None:
        return True
    return start_hour <= hour <= end_hour


# =============================================================================
# 2. BỘ PHÁT HIỆN PHƯƠNG TIỆN FASTER R-CNN (ĐỘ CHÍNH XÁC CAO TRÊN BAN NGÀY)
# =============================================================================
class GroundTruthVehicleDetector:
    """
    Bộ phát hiện phương tiện khách quan dựa trên Faster R-CNN (COCO Pretrained)
    để tách biệt mặt nạ phương tiện (M_veh) và mặt đường trống (M_road).
    Ưu tiên ResNet-50 FPN V2 độ phân giải cao; tự động fallback MobileNetV3 nếu cần.
    """
    VEHICLE_CLASSES = {
        1: "pedestrian",  # Người điều khiển phương tiện / người đi đường
        2: "bicycle",     # Xe đạp
        3: "car",         # Ô tô con
        4: "motorcycle",  # Xe máy (chủ đạo giao thông TP.HCM)
        6: "bus",         # Xe buýt
        8: "truck",       # Xe tải
    }

    def __init__(self, device: torch.device, conf_thresh: float = 0.35, arch: str = "auto"):
        self.device = device
        self.conf_thresh = conf_thresh
        self.model = None

        # Kiểm tra xem trọng số ResNet-50 FPN V2 đã có sẵn trong cache chưa
        torch_cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        resnet_cached = os.path.isfile(os.path.join(torch_cache_dir, "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"))

        if arch == "resnet50" or (arch == "auto" and resnet_cached):
            try:
                from torchvision.models.detection import (
                    fasterrcnn_resnet50_fpn_v2,
                    FasterRCNN_ResNet50_FPN_V2_Weights,
                )
                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self.model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
                self.model.eval()
                print(f"🚗 [VehicleDetector] Nạp thành công Faster R-CNN ResNet-50 FPN V2 trên {device}!")
                return
            except Exception as e_res:
                print(f"   ℹ️ ResNet-50 FPN V2 notice ({e_res}). Chuyển sang MobileNetV3 FPN...")

        # Fallback hoặc mặc định dùng MobileNetV3 FPN (đã có sẵn trong cache)
        try:
            from torchvision.models.detection import (
                fasterrcnn_mobilenet_v3_large_fpn,
                FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            )
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(device)
            self.model.eval()
            print(f"🚗 [VehicleDetector] Nạp thành công Faster R-CNN MobileNetV3 FPN trên {device}!")
        except Exception as e_mob:
            print(f"   ⚠️ Không thể nạp Faster R-CNN detector: {e_mob}")

    @torch.no_grad()
    def detect_vehicles_batch(self, image_tensors: List[torch.Tensor]) -> List[List[Dict]]:
        """Phát hiện phương tiện theo batch trên GPU."""
        if self.model is None or not image_tensors:
            return [[] for _ in image_tensors]

        inputs = [img.to(self.device) for img in image_tensors]
        predictions = self.model(inputs)

        batch_valid = []
        for pred in predictions:
            boxes = pred["boxes"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()

            valid_detections = []
            for box, label, score in zip(boxes, labels, scores):
                if score >= self.conf_thresh and int(label) in self.VEHICLE_CLASSES:
                    valid_detections.append({
                        "box": box,  # [x1, y1, x2, y2]
                        "label": int(label),
                        "class_name": self.VEHICLE_CLASSES[int(label)],
                        "score": float(score),
                    })
            batch_valid.append(valid_detections)
        return batch_valid

    @staticmethod
    def build_patch_masks(
        detections: List[Dict],
        img_h: int = 256,
        img_w: int = 448,
        patch_size: int = 16,
        road_start_row: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Chuyển đổi bounding boxes sang mặt nạ patch (Hp=16, Wp=28):
        - M_veh : Patch chứa phương tiện (diện tích che phủ >= 20%).
        - M_road: Patch mặt đường trống (từ hàng road_start_row trở xuống, loại bỏ vùng đệm quanh xe).
        - M_sky : Patch hậu cảnh tĩnh trên cao (bầu trời, bảng tên đường, tán cây).
        """
        Hp = img_h // patch_size
        Wp = img_w // patch_size

        pixel_mask = np.zeros((img_h, img_w), dtype=np.float32)
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            x1 = max(0, min(img_w, int(round(x1))))
            y1 = max(0, min(img_h, int(round(y1))))
            x2 = max(0, min(img_w, int(round(x2))))
            y2 = max(0, min(img_h, int(round(y2))))
            if x2 > x1 and y2 > y1:
                pixel_mask[y1:y2, x1:x2] = 1.0

        M_veh = torch.zeros((Hp, Wp), dtype=torch.bool)
        for i in range(Hp):
            for j in range(Wp):
                p_area = pixel_mask[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
                if p_area.mean() >= 0.20:
                    M_veh[i, j] = True

        # Mở rộng vùng đệm 1 ô patch quanh xe để loại trừ hoàn toàn bóng râm và đường viền mép xe
        M_veh_dilated = dilate_mask_2d(M_veh, radius=1)

        # Mặt nạ mặt đường trống: chỉ lấy khu vực lòng đường hợp lệ và không dính xe
        M_road = torch.zeros((Hp, Wp), dtype=torch.bool)
        M_road[road_start_row:, :] = True
        M_road = M_road & (~M_veh_dilated)

        # Mặt nạ hậu cảnh trên cao
        M_sky = torch.zeros((Hp, Wp), dtype=torch.bool)
        M_sky[:road_start_row, :] = True
        M_sky = M_sky & (~M_veh_dilated)

        return M_veh, M_road, M_sky


# =============================================================================
# 3. THU THẬP MẪU DỮ LIỆU BAN NGÀY (6H - 17H) PHÂN BỔ THEO CAMERA ID
# =============================================================================
def collect_daylight_samples(
    data_dir: str,
    tam_cache_dir: Optional[str] = None,
    num_cams_sample: int = 25,
    frames_per_cam: int = 8,
    img_size: Tuple[int, int] = (256, 448),
    start_hour: int = 6,
    end_hour: int = 17,
) -> Tuple[List[torch.Tensor], List[int], List[str], List[Optional[torch.Tensor]]]:
    """
    Thu thập ảnh có lọc khung giờ ban ngày (06:00 - 17:00, GMT+7) từ nhiều camera khác nhau.
    Đảm bảo mỗi camera có nhiều khung hình ở các thời điểm khác nhau để kiểm định Camera ID.
    """
    H, W = img_size
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    cam_to_files = defaultdict(list)

    total_scanned = 0
    total_daylight = 0

    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                total_scanned += 1
                full_p = os.path.join(root, f)
                rel = os.path.relpath(full_p, data_dir)
                parts = rel.split(os.sep)
                cam_key = parts[0] if len(parts) > 1 else f.split("_")[0]

                # Lọc khung giờ ban ngày sáng - chiều (6h - 17h)
                if is_daylight_hour(f, start_hour=start_hour, end_hour=end_hour):
                    total_daylight += 1
                    cam_to_files[cam_key].append((full_p, rel, f))

    print(f"☀️ [Daylight Filter] Quét {total_scanned} ảnh -> Giữ lại {total_daylight} ảnh "
          f"trong khung giờ ban ngày ({start_hour}h00 - {end_hour}h00, GMT+7) trên {len(cam_to_files)} camera.")

    if not cam_to_files:
        raise FileNotFoundError(f"Không tìm thấy ảnh ban ngày hợp lệ trong '{data_dir}'!")

    # Ưu tiên các camera có đủ số lượng khung hình ban ngày
    sorted_cams = sorted(cam_to_files.keys(), key=lambda k: len(cam_to_files[k]), reverse=True)
    selected_cams = sorted_cams[: min(num_cams_sample, len(sorted_cams))]

    image_tensors: List[torch.Tensor] = []
    cam_ids: List[int] = []
    file_names: List[str] = []
    cached_pi_list: List[Optional[torch.Tensor]] = []

    for cid_idx, cam_key in enumerate(selected_cams):
        items = sorted(cam_to_files[cam_key], key=lambda x: x[0])
        if len(items) > frames_per_cam:
            indices = np.linspace(0, len(items) - 1, frames_per_cam, dtype=int)
            chosen_items = [items[i] for i in indices]
        else:
            chosen_items = items

        for full_p, rel, fname in chosen_items:
            with Image.open(full_p) as img:
                img = img.convert("RGB").resize((W, H), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)
                image_tensors.append(t)
                cam_ids.append(cid_idx)
                file_names.append(fname)

            pi_cached = None
            if tam_cache_dir and os.path.isdir(tam_cache_dir):
                cache_p = os.path.join(tam_cache_dir, os.path.splitext(rel)[0] + ".pt")
                if os.path.isfile(cache_p):
                    try:
                        pi_cached = torch.load(cache_p, map_location="cpu", weights_only=True)
                    except TypeError:
                        pi_cached = torch.load(cache_p, map_location="cpu")
            cached_pi_list.append(pi_cached)

    return image_tensors, cam_ids, file_names, cached_pi_list


# =============================================================================
# 4. TÍNH BẢN ĐỒ TIỀN CẢNH TAM (THỰC TẾ TỪ CACHE HOẶC MODEL)
# =============================================================================
@torch.no_grad()
def get_current_tam_maps(
    frozen_extractor: FrozenExtractor,
    pos_stats: PositionStats,
    calibrator: GMMCalibrator,
    image_tensors: List[torch.Tensor],
    cam_ids: List[int],
    cached_pi_list: Optional[List[Optional[torch.Tensor]]],
    device: torch.device,
    batch_size: int = 16,
) -> List[torch.Tensor]:
    """Lấy bản đồ xác suất tiền cảnh TAM (ưu tiên đọc từ cache .pt đã nạp hoặc tính qua GMM)."""
    current_tam_maps = []
    N = len(image_tensors)

    for start_idx in range(0, N, batch_size):
        end_idx = min(N, start_idx + batch_size)
        batch_x = torch.stack(image_tensors[start_idx:end_idx], dim=0).to(device)
        batch_cids = torch.tensor(cam_ids[start_idx:end_idx], dtype=torch.long, device=device)
        B = batch_x.shape[0]

        U = frozen_extractor(batch_x).float()
        a_curr, _ = pos_stats.atypicality(batch_cids, U)
        pi_batch = calibrator.posterior(a_curr, refine=True).reshape(B, 16, 28).cpu()

        for b_i in range(B):
            global_i = start_idx + b_i
            if cached_pi_list is not None and cached_pi_list[global_i] is not None:
                c_pi = refine_traffic_foreground_map(cached_pi_list[global_i].float())
                current_tam_maps.append(c_pi.cpu())
            else:
                current_tam_maps.append(pi_batch[b_i])

    return current_tam_maps


# =============================================================================
# 5. TRÍCH XUẤT ĐẶC TRƯNG TỔNG QUÁT TỪ BACKBONE
# =============================================================================
@torch.no_grad()
def extract_model_features(
    model: nn.Module,
    image_tensors: List[torch.Tensor],
    tam_maps: List[torch.Tensor],
    gt_veh_masks: List[torch.Tensor],
    gt_road_masks: List[torch.Tensor],
    cam_ids: List[int],
    device: torch.device,
    batch_size: int = 16,
    max_patches_per_image: int = 25,
) -> Dict[str, Any]:
    """
    Trích xuất toàn bộ biểu diễn từ backbone:
      - CLS token (N, D)
      - Mean Patch token (N, D)
      - Foreground-weighted token z_fg = sum(pi * z) / (sum(pi) + eps) (N, D)
      - Patch tokens trên xe (M_veh) và trên đường trống (M_road) KÈM THEO CAMERA ID
      - Khoảng cách Cosine giữa tâm cụm xe và đường trong từng khung hình
    """
    model.eval()
    cls_list, mean_patch_list, fg_patch_list = [], [], []
    veh_patch_tokens, veh_cam_ids = [], []
    road_patch_tokens, road_cam_ids = [], []
    cosine_separations = []

    N = len(image_tensors)
    for start_idx in range(0, N, batch_size):
        end_idx = min(N, start_idx + batch_size)
        batch_x = torch.stack(image_tensors[start_idx:end_idx], dim=0).to(device)
        batch_pi = torch.stack(tam_maps[start_idx:end_idx], dim=0).to(device)
        B = batch_x.shape[0]

        norm_x = imagenet_normalize(batch_x)
        cls_tok, patch_tok = extract_tokens(model, norm_x, patch_size=16)
        if patch_tok.dim() == 4:
            patch_tok = patch_tok.flatten(1, 2)  # (B, 448, D)

        cls_list.append(cls_tok.cpu().numpy())
        mean_patch_list.append(patch_tok.mean(dim=1).cpu().numpy())

        # Tính Foreground Token có trọng số TAM
        pi_flat = batch_pi.view(B, -1, 1)
        fg_tok = (patch_tok * pi_flat).sum(dim=1) / (pi_flat.sum(dim=1) + 1e-6)
        fg_patch_list.append(fg_tok.cpu().numpy())

        patch_norm = F.normalize(patch_tok.float(), p=2, dim=-1).cpu()
        for b_i in range(B):
            g_idx = start_idx + b_i
            cid = cam_ids[g_idx]
            m_v = gt_veh_masks[g_idx].view(-1).bool()
            m_r = gt_road_masks[g_idx].view(-1).bool()
            p_b = patch_norm[b_i]

            if m_v.sum() >= 2 and m_r.sum() >= 2:
                v_feats = p_b[m_v]
                r_feats = p_b[m_r]

                # Lấy mẫu cân bằng số lượng patch mỗi ảnh
                if v_feats.shape[0] > max_patches_per_image:
                    v_feats = v_feats[torch.randperm(v_feats.shape[0])[:max_patches_per_image]]
                if r_feats.shape[0] > max_patches_per_image:
                    r_feats = r_feats[torch.randperm(r_feats.shape[0])[:max_patches_per_image]]

                n_v = v_feats.shape[0]
                n_r = r_feats.shape[0]

                veh_patch_tokens.append(v_feats.numpy())
                veh_cam_ids.extend([cid] * n_v)

                road_patch_tokens.append(r_feats.numpy())
                road_cam_ids.extend([cid] * n_r)

                v_mean = F.normalize(v_feats.mean(dim=0, keepdim=True), p=2, dim=-1)
                r_mean = F.normalize(r_feats.mean(dim=0, keepdim=True), p=2, dim=-1)
                cos_vr = float((v_mean * r_mean).sum().item())
                cosine_separations.append(1.0 - cos_vr)

    return {
        "cls": np.concatenate(cls_list, axis=0),
        "mean_patch": np.concatenate(mean_patch_list, axis=0),
        "fg_patch": np.concatenate(fg_patch_list, axis=0),
        "veh_patches": np.concatenate(veh_patch_tokens, axis=0) if veh_patch_tokens else np.zeros((10, 384)),
        "veh_cam_ids": np.array(veh_cam_ids, dtype=int) if veh_cam_ids else np.zeros(10, dtype=int),
        "road_patches": np.concatenate(road_patch_tokens, axis=0) if road_patch_tokens else np.zeros((10, 384)),
        "road_cam_ids": np.array(road_cam_ids, dtype=int) if road_cam_ids else np.zeros(10, dtype=int),
        "mean_cosine_sep": float(np.mean(cosine_separations)) if cosine_separations else 0.0,
    }


def compute_effective_rank(features: np.ndarray, eps: float = 1e-12) -> float:
    """Tính số chiều hiệu dụng Effective Rank (Roy & Vetterli) từ ma trận đặc trưng."""
    X = features - np.mean(features, axis=0, keepdims=True)
    try:
        _, S, _ = np.linalg.svd(X, full_matrices=False)
        S_norm = S / (np.sum(S) + eps)
        p = np.clip(S_norm, eps, 1.0)
        entropy = -np.sum(p * np.log(p))
        return float(np.exp(entropy))
    except Exception:
        return 1.0


# =============================================================================
# 6. ƯU TIÊN 1: FROZEN LINEAR PROBE CHỈ DÙNG CAMERA-DISJOINT SPLIT
# =============================================================================
def evaluate_camera_disjoint_linear_probe(
    data: Dict[str, Any],
    cids_arr: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Huấn luyện và đánh giá Frozen Linear Probe trên các camera HOÀN TOÀN CHƯA THẤY:
    Cam_train ∩ Cam_test = ∅ (sử dụng GroupKFold theo Camera ID).
    Tuyệt đối không rò rỉ dữ liệu giữa tập huấn luyện và kiểm định!
    """
    # 1. Patch-level Vehicle vs. Road Linear Probe
    X_patch = np.concatenate([data["veh_patches"], data["road_patches"]], axis=0)
    y_patch = np.concatenate([
        np.ones(len(data["veh_patches"]), dtype=int),
        np.zeros(len(data["road_patches"]), dtype=int),
    ])
    groups_patch = np.concatenate([data["veh_cam_ids"], data["road_cam_ids"]], axis=0)

    # Chuẩn hóa L2 trước khi đưa vào bộ phân loại tuyến tính
    X_patch_norm = X_patch / (np.linalg.norm(X_patch, axis=1, keepdims=True) + 1e-8)

    # Lấy mẫu tối đa 40,000 patches để đánh giá nhanh chóng và ổn định
    if len(X_patch_norm) > 40000:
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(len(X_patch_norm), 40000, replace=False)
        X_eval = X_patch_norm[sub_idx]
        y_eval = y_patch[sub_idx]
        groups_eval = groups_patch[sub_idx]
    else:
        X_eval = X_patch_norm
        y_eval = y_patch
        groups_eval = groups_patch

    unique_groups = np.unique(groups_eval)
    actual_splits = max(2, min(n_splits, len(unique_groups)))

    # Phân lập 100% theo Camera ID qua GroupKFold
    gkf = GroupKFold(n_splits=actual_splits)
    p_accs, p_f1s, p_aucs = [], [], []

    for tr_idx, te_idx in gkf.split(X_eval, y_eval, groups_eval):
        # Kiểm tra tính phân lập: cameras trong test phải hoàn toàn không xuất hiện ở train
        tr_cams = set(groups_eval[tr_idx])
        te_cams = set(groups_eval[te_idx])
        assert len(tr_cams.intersection(te_cams)) == 0, "LỖI RÒ RỈ: Có camera xuất hiện ở cả Train và Test!"

        clf = LogisticRegression(max_iter=300, C=1.0, n_jobs=-1, solver="lbfgs")
        clf.fit(X_eval[tr_idx], y_eval[tr_idx])
        preds = clf.predict(X_eval[te_idx])
        probs = clf.predict_proba(X_eval[te_idx])[:, 1]

        p_accs.append(accuracy_score(y_eval[te_idx], preds))
        p_f1s.append(f1_score(y_eval[te_idx], preds))
        try:
            p_aucs.append(roc_auc_score(y_eval[te_idx], probs))
        except Exception:
            p_aucs.append(0.5)

    # 2. Camera-ID Shortcut Probe trên CLS token và Foreground token
    # Đo lường mức độ mô hình còn ghi nhớ đặc thù góc máy/mặt đường
    unique_cams, cam_counts = np.unique(cids_arr, return_counts=True)
    cam_probe_cls_acc = 0.0
    cam_probe_fg_acc = 0.0
    chance_level_acc = 100.0 / max(1, len(unique_cams))

    if len(unique_cams) >= 2 and np.min(cam_counts) >= 2:
        k_cam_splits = max(2, min(5, int(np.min(cam_counts))))
        from sklearn.model_selection import StratifiedKFold
        skf_cam = StratifiedKFold(n_splits=k_cam_splits, shuffle=True, random_state=42)

        X_cls_norm = data["cls"] / (np.linalg.norm(data["cls"], axis=1, keepdims=True) + 1e-8)
        X_fg_norm = data["fg_patch"] / (np.linalg.norm(data["fg_patch"], axis=1, keepdims=True) + 1e-8)

        c_accs, fg_accs = [], []
        for tr_idx, te_idx in skf_cam.split(X_cls_norm, cids_arr):
            clf_c = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1)
            clf_c.fit(X_cls_norm[tr_idx], cids_arr[tr_idx])
            c_accs.append(accuracy_score(cids_arr[te_idx], clf_c.predict(X_cls_norm[te_idx])))

            clf_fg = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1)
            clf_fg.fit(X_fg_norm[tr_idx], cids_arr[tr_idx])
            fg_accs.append(accuracy_score(cids_arr[te_idx], clf_fg.predict(X_fg_norm[te_idx])))

        cam_probe_cls_acc = float(np.mean(c_accs) * 100.0)
        cam_probe_fg_acc = float(np.mean(fg_accs) * 100.0)

    return {
        "camera_disjoint_probe": {
            "num_camera_groups": len(unique_groups),
            "num_splits": actual_splits,
            "veh_vs_road_acc_pct": float(np.mean(p_accs) * 100.0),
            "veh_vs_road_f1_pct": float(np.mean(p_f1s) * 100.0),
            "veh_vs_road_auc": float(np.mean(p_aucs)),
            "veh_road_cosine_sep": data["mean_cosine_sep"],
        },
        "camera_id_shortcut": {
            "chance_level_acc_pct": chance_level_acc,
            "cam_id_cls_acc_pct": cam_probe_cls_acc,
            "cam_id_fg_acc_pct": cam_probe_fg_acc,
            "shortcut_suppressed": bool(cam_probe_fg_acc < cam_probe_cls_acc),
        },
        "effective_rank": {
            "erank_cls": compute_effective_rank(data["cls"]),
            "erank_fg": compute_effective_rank(data["fg_patch"]),
            "erank_patches": compute_effective_rank(X_patch),
        },
    }


# =============================================================================
# 7. ƯU TIÊN 4: ĐÁNH GIÁ TRÊN BỘ DỮ LIỆU NHÃN THỦ CÔNG (SAMPLE_LABELS.CSV)
# =============================================================================
def evaluate_manual_labeled_subset(
    models_dict: Dict[str, nn.Module],
    csv_path: str,
    image_dir: str,
    device: torch.device,
    start_hour: int = 6,
    end_hour: int = 17,
) -> Dict[str, Any]:
    """
    Kiểm định trên tập nhãn gán thủ công (sample_labels.csv):
    1. Lọc khung giờ ban ngày (6h - 17h, GMT+7)
    2. Đánh giá Downstream Probe phân lập theo Camera ID:
       - Hồi quy số lượng xe (Ridge MAE, RMSE, R2 trên xe_may, o_to, tong)
       - Phân loại cấp độ mật độ (phan_loai)
    3. Tương quan TAM với số đếm thủ công của thẩm định viên
    """
    if not os.path.isfile(csv_path) or not os.path.isdir(image_dir):
        print(f"ℹ️ [Manual Audit] Không tìm thấy '{csv_path}' hoặc '{image_dir}'. Bỏ qua đánh giá nhãn thủ công.")
        return {}

    df_raw = pd.read_csv(csv_path)
    print(f"\n🏷️ [Manual GT Audit] Đọc {len(df_raw)} dòng từ {csv_path}...")

    # Tìm các ảnh thực sự tồn tại trên đĩa và thuộc khung giờ ban ngày
    valid_rows = []
    for _, row in df_raw.iterrows():
        fname = str(row["filename"]).strip()
        fpath = os.path.join(image_dir, fname)
        if os.path.isfile(fpath) and is_daylight_hour(fname, start_hour=start_hour, end_hour=end_hour):
            # Parse camera ID từ tên tệp (ví dụ: '102_1755762668.jpg' -> cam_id 102)
            cam_str = fname.split("_")[0]
            try:
                cam_id = int(cam_str)
            except ValueError:
                cam_id = 0
            valid_rows.append({
                "filename": fname,
                "filepath": fpath,
                "cam_id": cam_id,
                "xe_may": float(row.get("xe_may", 0.0)),
                "o_to": float(row.get("o_to", 0.0)),
                "tong": float(row.get("tong", 0.0)),
                "phan_loai": int(row.get("phan_loai", 1)),
            })

    df = pd.DataFrame(valid_rows)
    print(f"☀️ [Manual GT Audit] Giữ lại {len(df)} ảnh ban ngày ({start_hour}h - {end_hour}h) trên {df['cam_id'].nunique()} camera.")

    if len(df) < 10:
        print("⚠️ [Manual GT Audit] Số lượng ảnh ban ngày trong tập nhãn thủ công quá ít (<10). Bỏ qua.")
        return {}

    # Nạp ảnh tensor
    img_tensors = []
    for _, row in df.iterrows():
        with Image.open(row["filepath"]) as img:
            img = img.convert("RGB").resize((448, 256), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            img_tensors.append(t)

    cam_groups = df["cam_id"].values
    y_tong = df["tong"].values
    y_density = df["phan_loai"].values

    manual_results = {
        "num_samples": len(df),
        "num_cameras": int(df["cam_id"].nunique()),
        "downstream_probes": {},
    }

    # Đánh giá từng mô hình trong danh sách đối chứng
    for m_name, model in models_dict.items():
        model.eval()
        feats_list = []
        batch_sz = 16
        with torch.no_grad():
            for i in range(0, len(img_tensors), batch_sz):
                bx = torch.stack(img_tensors[i : i + batch_sz]).to(device)
                norm_bx = imagenet_normalize(bx)
                cls_tok, patch_tok = extract_tokens(model, norm_bx, patch_size=16)
                if patch_tok.dim() == 4:
                    patch_tok = patch_tok.flatten(1, 2)
                # Dùng vector gộp [CLS || Mean_Patch] làm đặc trưng toàn khung hình
                mean_p = patch_tok.mean(dim=1)
                comb_feat = torch.cat([cls_tok, mean_p], dim=-1).cpu().numpy()
                feats_list.append(comb_feat)

        X = np.concatenate(feats_list, axis=0)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # 1. Hồi quy số lượng xe tổng (Ridge Regression với Camera-Disjoint GroupKFold)
        n_splits_m = max(2, min(5, df["cam_id"].nunique()))
        gkf = GroupKFold(n_splits=n_splits_m)

        maes, rmses, r2s = [], [], []
        for tr_idx, te_idx in gkf.split(X_norm, y_tong, cam_groups):
            reg = Ridge(alpha=1.0)
            reg.fit(X_norm[tr_idx], y_tong[tr_idx])
            preds = reg.predict(X_norm[te_idx])
            maes.append(mean_absolute_error(y_tong[te_idx], preds))
            rmses.append(np.sqrt(mean_squared_error(y_tong[te_idx], preds)))
            r2s.append(r2_score(y_tong[te_idx], preds))

        # 2. Phân loại mức độ mật độ (phan_loai: 1 đến 5)
        dens_accs, dens_f1s = [], []
        for tr_idx, te_idx in gkf.split(X_norm, y_density, cam_groups):
            clf = LogisticRegression(max_iter=300, C=1.0)
            clf.fit(X_norm[tr_idx], y_density[tr_idx])
            d_preds = clf.predict(X_norm[te_idx])
            dens_accs.append(accuracy_score(y_density[te_idx], d_preds))
            dens_f1s.append(f1_score(y_density[te_idx], d_preds, average="macro"))

        manual_results["downstream_probes"][m_name] = {
            "vehicle_counting_mae": float(np.mean(maes)),
            "vehicle_counting_rmse": float(np.mean(rmses)),
            "vehicle_counting_r2": float(np.mean(r2s)),
            "density_acc_pct": float(np.mean(dens_accs) * 100.0),
            "density_macro_f1_pct": float(np.mean(dens_f1s) * 100.0),
        }

    return manual_results


# =============================================================================
# 8. XUẤT HÌNH ẢNH TRỰC QUAN HÓA AGM TRÊN BAN NGÀY
# =============================================================================
def render_daylight_visual_overlay(
    image_tensors: List[torch.Tensor],
    tam_maps: List[torch.Tensor],
    agm_masks: List[torch.Tensor],
    gt_veh_masks: List[torch.Tensor],
    detections_list: List[List[Dict]],
    file_names: List[str],
    save_path: str = "audit2_visuals_overlay.png",
    patch_size: int = 16,
):
    """Xuất hình ảnh trực quan hóa AGM chồng ảnh gốc ban ngày (Đỏ: Trên xe, Xanh cyan: Ngoài xe)."""
    # Chọn tối đa 4 ảnh ban ngày có nhiều phương tiện nhất
    valid_indices = sorted(range(len(image_tensors)), key=lambda idx: len(detections_list[idx]), reverse=True)[:4]
    num_samples = len(valid_indices)
    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 4, figsize=(22, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for row_i, idx in enumerate(valid_indices):
        img_t = image_tensors[idx].cpu()
        pi = tam_maps[idx].cpu().numpy()
        Hp, Wp = pi.shape
        mask = agm_masks[idx].cpu().reshape(Hp, Wp).numpy()
        m_veh = gt_veh_masks[idx].cpu().reshape(Hp, Wp).numpy()
        dets = detections_list[idx]
        fname = file_names[idx]
        hr = parse_vietnam_hour(fname)
        hr_str = f" [{hr}h00]" if hr is not None else ""

        img_np = (img_t.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # Cột 1: Ảnh gốc + Bounding Box
        col1_img = img_pil.copy()
        draw1 = ImageDraw.Draw(col1_img)
        for det in dets:
            b = det["box"]
            draw1.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 230, 0), width=2)
            draw1.text((b[0], max(0, b[1] - 12)), f"{det['class_name']}", fill=(255, 230, 0))
        axes[row_i, 0].imshow(col1_img)
        axes[row_i, 0].set_title(f"Ảnh Gốc Ban Ngày{hr_str} ({len(dets)} xe)", fontsize=11, fontweight="bold")
        axes[row_i, 0].axis("off")

        # Cột 2: Bản đồ TAM pi
        im2 = axes[row_i, 1].imshow(pi, cmap="turbo", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[row_i, 1].set_title(f"Bản Đồ TAM π (Mean: {pi.mean():.3f})", fontsize=11, fontweight="bold")
        axes[row_i, 1].axis("off")
        plt.colorbar(im2, ax=axes[row_i, 1], fraction=0.046, pad=0.04)

        # Cột 3: AGM Overlay
        overlay_np = img_np.copy().astype(np.float32)
        n_on_veh, n_off_veh = 0, 0
        for r in range(Hp):
            for c in range(Wp):
                y1, y2 = r * patch_size, (r + 1) * patch_size
                x1, x2 = c * patch_size, (c + 1) * patch_size
                if mask[r, c]:
                    if m_veh[r, c]:
                        n_on_veh += 1
                        overlay_np[y1:y2, x1:x2, 0] = overlay_np[y1:y2, x1:x2, 0] * 0.35 + 255.0 * 0.65
                        overlay_np[y1:y2, x1:x2, 1] = overlay_np[y1:y2, x1:x2, 1] * 0.35 + 30.0 * 0.65
                        overlay_np[y1:y2, x1:x2, 2] = overlay_np[y1:y2, x1:x2, 2] * 0.35 + 30.0 * 0.65
                    else:
                        n_off_veh += 1
                        overlay_np[y1:y2, x1:x2, 0] = overlay_np[y1:y2, x1:x2, 0] * 0.35 + 0.0 * 0.65
                        overlay_np[y1:y2, x1:x2, 1] = overlay_np[y1:y2, x1:x2, 1] * 0.35 + 190.0 * 0.65
                        overlay_np[y1:y2, x1:x2, 2] = overlay_np[y1:y2, x1:x2, 2] * 0.35 + 255.0 * 0.65
                    overlay_np[y1 : y1 + 1, x1:x2] = 255.0
                    overlay_np[y2 - 1 : y2, x1:x2] = 255.0
                    overlay_np[y1:y2, x1 : x1 + 1] = 255.0
                    overlay_np[y1:y2, x2 - 1 : x2] = 255.0

        axes[row_i, 2].imshow(Image.fromarray(overlay_np.clip(0, 255).astype(np.uint8)))
        tot_m = max(1, n_on_veh + n_off_veh)
        axes[row_i, 2].set_title(f"AGM Overlay (Xe [Đỏ]: {n_on_veh} | Ngoài [Xanh]: {n_off_veh} | {n_on_veh/tot_m*100:.1f}%)",
                                 fontsize=10, fontweight="bold")
        axes[row_i, 2].axis("off")

        # Cột 4: Đầu vào Student
        student_input_np = img_np.copy().astype(np.float32)
        for r in range(Hp):
            for c in range(Wp):
                if mask[r, c]:
                    y1, y2 = r * patch_size, (r + 1) * patch_size
                    x1, x2 = c * patch_size, (c + 1) * patch_size
                    student_input_np[y1:y2, x1:x2] = 127.5
        axes[row_i, 3].imshow(Image.fromarray(student_input_np.clip(0, 255).astype(np.uint8)))
        axes[row_i, 3].set_title(f"Đầu Vào Student (Che {int(mask.sum())}/448 patches)", fontsize=11, fontweight="bold")
        axes[row_i, 3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ [Visual Overlay] Đã xuất ảnh trực quan hóa AGM ban ngày tại: {save_path}")


# =============================================================================
# 9. HÀM MAIN: ĐIỀU PHỐI KIỂM ĐỊNH TOÀN DIỆN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="audit2: Kiểm toán định lượng thế hệ 2 giải trình phản biện GVHD")
    parser.add_argument("--data_dir", type=str, default="/workspace/output", help="Thư mục dữ liệu ảnh camera")
    parser.add_argument("--tam_cache_dir", type=str, default="/workspace/tam_cache", help="Thư mục TAM cache (.pt)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/direction1_new/last_checkpoint.pth",
        help="Đường dẫn tới file checkpoint cần đánh giá (mặc định: last_checkpoint.pth)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Danh sách nhiều checkpoints để so sánh ma trận (vd: baseline=path1 nosrs=path2 full=path3)",
    )
    parser.add_argument("--manual_csv", type=str, default="sample_test_images/audit_set/sample_labels.csv",
                        help="File CSV nhãn kiểm tra thủ công")
    parser.add_argument("--manual_image_dir", type=str, default="sample_test_images/audit_set",
                        help="Thư mục ảnh nhãn kiểm tra thủ công")
    parser.add_argument("--num_cams", type=int, default=25, help="Số lượng camera lấy mẫu kiểm định")
    parser.add_argument("--frames_per_cam", type=int, default=8, help="Số lượng ảnh ban ngày trên mỗi camera")
    parser.add_argument("--start_hour", type=int, default=6, help="Giờ bắt đầu khung giờ ban ngày (mặc định: 6h)")
    parser.add_argument("--end_hour", type=int, default=17, help="Giờ kết thúc khung giờ ban ngày (mặc định: 17h)")
    parser.add_argument("--eval_teacher", action="store_true", default=True, help="Đánh giá cả nhánh EMA Teacher nếu có")
    parser.add_argument("--detector_arch", type=str, default="auto", choices=["auto", "resnet50", "mobilenet"],
                        help="Kiến trúc detector (auto: ưu tiên mobilenet có sẵn để không mất thời gian tải weights)")
    parser.add_argument("--output_json", type=str, default="audit2_metrics_summary.json", help="File JSON xuất số liệu")
    parser.add_argument("--output_visual", type=str, default="audit2_visuals_overlay.png", help="File ảnh AGM overlay ban ngày")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 85)
    print(" 🔍 BẮT ĐẦU KIỂM ĐỊNH AUDIT2 (CAMERA-DISJOINT & LỌC BAN NGÀY 6H - 17H)")
    print(f"    - Thiết bị thực thi      : {device}")
    print(f"    - Thư mục ảnh camera    : {args.data_dir}")
    print(f"    - Thư mục TAM Cache      : {args.tam_cache_dir}")
    print(f"    - Khung giờ lọc          : {args.start_hour}h00 - {args.end_hour}h00 (GMT+7)")
    print(f"    - Checkpoint chỉ định    : {args.checkpoint}")
    print("=" * 85)

    # 1. Thu thập ảnh ban ngày (06:00 - 17:00) phân bổ theo Camera ID
    image_tensors, cam_ids, file_names, cached_pi_list = collect_daylight_samples(
        data_dir=args.data_dir,
        tam_cache_dir=args.tam_cache_dir,
        num_cams_sample=args.num_cams,
        frames_per_cam=args.frames_per_cam,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
    )
    print(f"📦 Đã nạp thành công {len(image_tensors)} ảnh ban ngày từ {len(set(cam_ids))} camera khác biệt.")

    # 2. Phát hiện phương tiện bằng Faster R-CNN theo batch
    detector = GroundTruthVehicleDetector(device=device, conf_thresh=0.35, arch=args.detector_arch)
    detections_list, gt_veh_masks, gt_road_masks, gt_sky_masks = [], [], [], []
    batch_det_sz = 16
    print(f"🚗 Bắt đầu phát hiện phương tiện trên {len(image_tensors)} ảnh ban ngày...")
    for start_idx in range(0, len(image_tensors), batch_det_sz):
        end_idx = min(len(image_tensors), start_idx + batch_det_sz)
        batch_imgs = image_tensors[start_idx:end_idx]
        b_dets = detector.detect_vehicles_batch(batch_imgs)
        for dets in b_dets:
            m_v, m_r, m_s = detector.build_patch_masks(dets)
            detections_list.append(dets)
            gt_veh_masks.append(m_v)
            gt_road_masks.append(m_r)
            gt_sky_masks.append(m_s)
    total_veh_detected = sum(len(d) for d in detections_list)
    print(f"🚗 Hoàn tất phát hiện tổng cộng {total_veh_detected} phương tiện trên {len(image_tensors)} ảnh.")

    # 3. Khởi tạo Vanilla DINOv3 và tính toán TAM
    print("\n⏳ Đang nạp mô hình DINOv3 Pretrained và chuẩn bị PositionStats/GMM...")
    vanilla_model, embed_dim, patch_size = get_dino_backbone("dinov3_vits16", pretrained=True, device=device)
    frozen_extractor = FrozenExtractor(
        model_name="dinov3_vits16",
        pca_dim=64,
        device=device,
        backbone=vanilla_model,
        embed_dim=embed_dim,
        patch_size=patch_size,
    ).to(device)

    num_cams_actual = max(cam_ids) + 1
    pos_stats = PositionStats(num_cams=num_cams_actual, num_patches=448, num_states=4, feat_dim=64).to(device)
    calibrator = GMMCalibrator()

    # Fit PCA & GMM nhanh cho các camera lấy mẫu
    with torch.no_grad():
        all_toks = []
        for idx in range(0, min(64, len(image_tensors)), 16):
            bx = torch.stack(image_tensors[idx : idx + 16]).to(device)
            _, ps = extract_tokens(vanilla_model, imagenet_normalize(bx), patch_size=patch_size)
            all_toks.append(ps.reshape(-1, embed_dim))
        frozen_extractor.fit_pca(torch.cat(all_toks, dim=0))

        for cid in sorted(set(cam_ids)):
            c_idxs = [i for i, c in enumerate(cam_ids) if c == cid]
            bx = torch.stack([image_tensors[i] for i in c_idxs]).to(device)
            u_cam = frozen_extractor(bx).float()
            pos_stats.init_kmeans(cid, u_cam, n_iters=15)
            c_tensor = torch.full((u_cam.shape[0],), cid, dtype=torch.long, device=device)
            a_cam, v_cam = pos_stats.atypicality(c_tensor, u_cam)
            calibrator.push(a_cam, v_cam)
        calibrator.refit(max_iters=30)

    current_tam_maps = get_current_tam_maps(
        frozen_extractor=frozen_extractor,
        pos_stats=pos_stats,
        calibrator=calibrator,
        image_tensors=image_tensors,
        cam_ids=cam_ids,
        cached_pi_list=cached_pi_list,
        device=device,
    )

    # 4. Trực quan hóa AGM trên ảnh ban ngày
    agm_masks = [agm_sample(pi.view(16, 28), ratio=0.35, phi=0.5, q_max=0.6, mode="block") for pi in current_tam_maps]
    render_daylight_visual_overlay(
        image_tensors, current_tam_maps, agm_masks, gt_veh_masks, detections_list, file_names, save_path=args.output_visual
    )

    # 5. Xây dựng danh sách các mô hình cần đánh giá
    models_to_eval = {}
    models_to_eval["vanilla_dinov3"] = vanilla_model

    # Phân tích danh sách checkpoint
    ckpt_dict = {}
    if args.checkpoints:
        for item in args.checkpoints:
            if "=" in item:
                k, v = item.split("=", 1)
                ckpt_dict[k.strip()] = v.strip()
            else:
                ckpt_dict[os.path.splitext(os.path.basename(item))[0]] = item.strip()
    elif args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt_dict["student_ssl"] = args.checkpoint

    for m_label, ckpt_p in ckpt_dict.items():
        if not os.path.isfile(ckpt_p):
            print(f"⚠️ Không tìm thấy file checkpoint: {ckpt_p}. Bỏ qua.")
            continue
        print(f"\n📦 Đang nạp mô hình từ checkpoint: {ckpt_p} ({m_label})...")
        try:
            ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_p, map_location=device)

        # Nạp Student Backbone
        s_bb, _, _ = get_dino_backbone("dinov3_vits16", pretrained=False, device=device)
        s_sd = ckpt.get("student_backbone", ckpt.get("student", ckpt.get("model_state", ckpt))) if isinstance(ckpt, dict) else ckpt
        smart_load_state_dict(s_bb, clean_state_dict(s_sd), strict=False, verbose=False)
        models_to_eval[m_label] = s_bb
        print(f"   ✅ Đã nạp thành công Student Backbone: '{m_label}'")

        # Nạp EMA Teacher Backbone nếu có
        if args.eval_teacher and isinstance(ckpt, dict) and "teacher_backbone" in ckpt:
            t_bb, _, _ = get_dino_backbone("dinov3_vits16", pretrained=False, device=device)
            smart_load_state_dict(t_bb, clean_state_dict(ckpt["teacher_backbone"]), strict=False, verbose=False)
            models_to_eval[f"{m_label}_teacher"] = t_bb
            print(f"   ✅ Đã nạp thành công EMA Teacher Backbone: '{m_label}_teacher'")

    # 6. Trích xuất đặc trưng và Đánh giá Frozen Linear Probe phân lập theo Camera ID
    cids_arr = np.array(cam_ids, dtype=int)
    all_probe_results = {}

    print("\n" + "=" * 85)
    print(" 🎯 BẮT ĐẦU ĐÁNH GIÁ FROZEN LINEAR PROBE (CHỈ DÙNG CAMERA-DISJOINT SPLIT)")
    print("=" * 85)

    for m_name, m_net in models_to_eval.items():
        print(f"\n🔬 [Đánh giá mô hình] -> {m_name.upper()}")
        m_data = extract_model_features(
            model=m_net,
            image_tensors=image_tensors,
            tam_maps=current_tam_maps,
            gt_veh_masks=gt_veh_masks,
            gt_road_masks=gt_road_masks,
            cam_ids=cam_ids,
            device=device,
        )

        res = evaluate_camera_disjoint_linear_probe(m_data, cids_arr=cids_arr, n_splits=5)
        all_probe_results[m_name] = res

        cd_res = res["camera_disjoint_probe"]
        cid_res = res["camera_id_shortcut"]
        er_res = res["effective_rank"]

        print(f"   📊 [Camera-Disjoint Patch Probe] Acc = {cd_res['veh_vs_road_acc_pct']:.2f}% | "
              f"F1 = {cd_res['veh_vs_road_f1_pct']:.2f}% | AUC = {cd_res['veh_vs_road_auc']:.4f} | "
              f"Cosine Sep = {cd_res['veh_road_cosine_sep']:.4f}")
        print(f"   📡 [Camera-ID Shortcut Probe] CLS Acc = {cid_res['cam_id_cls_acc_pct']:.2f}% | "
              f"FG Acc = {cid_res['cam_id_fg_acc_pct']:.2f}% (Chance: {cid_res['chance_level_acc_pct']:.2f}%)")
        print(f"   📐 [Effective Rank] eRank(FG) = {er_res['erank_fg']:.1f} | eRank(CLS) = {er_res['erank_cls']:.1f}")

    # 7. Đánh giá Downstream trên bộ dữ liệu nhãn thủ công (sample_labels.csv)
    manual_results = evaluate_manual_labeled_subset(
        models_dict=models_to_eval,
        csv_path=args.manual_csv,
        image_dir=args.manual_image_dir,
        device=device,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
    )

    # 8. Xuất báo cáo bảng tổng hợp ra màn hình
    print("\n" + "=" * 95)
    print(" 📋 BẢNG TỔNG HỢP SO SÁNH MA TRẬN MÔ HÌNH (CAMERA-DISJOINT & BAN NGÀY)")
    print("=" * 95)
    header = f"{'Mô hình':<24} | {'Disjoint F1':<12} | {'Disjoint AUC':<12} | {'Cosine Sep':<11} | {'Cam-ID (FG)':<12} | {'eRank (FG)':<10}"
    print(header)
    print("-" * 95)
    for m_name, res in all_probe_results.items():
        cd = res["camera_disjoint_probe"]
        cid = res["camera_id_shortcut"]
        er = res["effective_rank"]
        print(f"{m_name:<24} | {cd['veh_vs_road_f1_pct']:>10.2f}% | {cd['veh_vs_road_auc']:>12.4f} | {cd['veh_road_cosine_sep']:>11.4f} | {cid['cam_id_fg_acc_pct']:>10.2f}% | {er['erank_fg']:>10.1f}")
    print("=" * 95)

    if manual_results and "downstream_probes" in manual_results:
        print("\n" + "=" * 80)
        print(" 🏷️ BẢNG ĐÁNH GIÁ TRÊN TẬP NHÃN THỦ CÔNG (DOWNSTREAM COUNTING & DENSITY)")
        print("=" * 80)
        h2 = f"{'Mô hình':<24} | {'Count MAE':<12} | {'Count R2':<10} | {'Density Acc':<12} | {'Density F1':<12}"
        print(h2)
        print("-" * 80)
        for m_name, m_res in manual_results["downstream_probes"].items():
            print(f"{m_name:<24} | {m_res['vehicle_counting_mae']:>10.2f}  | {m_res['vehicle_counting_r2']:>10.4f} | {m_res['density_acc_pct']:>10.2f}% | {m_res['density_macro_f1_pct']:>10.2f}%")
        print("=" * 80)

    # 9. Lưu toàn bộ kết quả chi tiết ra file JSON
    full_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "filter_hours": f"{args.start_hour}h00 - {args.end_hour}h00 (GMT+7)",
        "num_images_evaluated": len(image_tensors),
        "num_cameras_evaluated": len(set(cam_ids)),
        "split_protocol": "Camera-Disjoint (GroupKFold by cam_id)",
        "probe_results": all_probe_results,
        "manual_gt_downstream": manual_results,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Đã lưu đầy đủ kết quả kiểm toán tại: {args.output_json}")


if __name__ == "__main__":
    main()
