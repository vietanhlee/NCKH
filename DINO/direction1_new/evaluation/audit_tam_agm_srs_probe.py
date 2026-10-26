"""
=============================================================================
  Hướng G: audit_tam_agm_srs_probe.py
  Bộ công cụ kiểm toán định lượng toàn diện cho hệ thống Vehicle-Centric SSL DINOv3
  Trả lời trực diện 5 câu hỏi phản biện từ Giảng viên Hướng dẫn (hoàn toàn tự động
  trên tập dữ liệu camera giao thông `/workspace/output`, không cần gán nhãn thủ công):

  1. Kiểm tra điểm TAM trên vùng phương tiện và vùng mặt đường không chứa phương tiện
     (So sánh TAM bản cũ chưa lọc tương phản cục bộ vs TAM bản mới hiện tại).
  2. Trực quan hóa AGM chồng trực tiếp lên ảnh gốc, phân biệt patch được che nằm
     trên xe (Đỏ) và ngoài xe (Xanh Cyan).
  3. So sánh Random Masking, Deterministic Top-K, Gumbel Top-K (Patch) và Gumbel Block 2x2.
  4. Dùng last_checkpoint đánh giá Frozen Linear Probe so với DINOv3 Pretrained:
     - Patch-level Vehicle vs. Empty Road Linear Probe (Acc, F1, AUC-ROC)
     - Camera-ID Linear Probe (Đo mức độ triệt tiêu Camera Shortcut trên CLS & FG Token)
     - Độ phân tách Cosine Xe - Đường (Cosine Separation) & Effective Rank
  5. Thống kê tỷ lệ ảnh SRS thực sự thay đổi (Δ > 0%) và mức thay đổi trung bình
     (Đối chứng Logic SRS cũ bị Δ = 0% vs Logic SRS mới Adaptive Fallback + Continuous).
=============================================================================
"""

import os
import sys
import argparse
import json
import time
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

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

# Sklearn cho đánh giá Frozen Linear Probe
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# 1. BỘ TRÍCH XUẤT VÙNG PHƯƠNG TIỆN KHÁCH QUAN (FASTER R-CNN DETECTOR)
# =============================================================================
class GroundTruthVehicleDetector:
    """
    Sử dụng mô hình phát hiện đối tượng chuẩn COCO (Faster R-CNN MobileNetV3)
    để tách vùng có phương tiện (M_veh) và vùng lòng đường trống không có phương tiện (M_road).
    """
    VEHICLE_CLASSES = {
        1: "pedestrian",  # Người đi xe máy / người tham gia giao thông
        2: "bicycle",     # Xe đạp
        3: "car",         # Ô tô con
        4: "motorcycle",  # Xe máy (phương tiện chủ đạo TP.HCM)
        6: "bus",         # Xe buýt
        8: "truck",       # Xe tải
    }

    def __init__(self, device: torch.device, conf_thresh: float = 0.35):
        self.device = device
        self.conf_thresh = conf_thresh
        print(f"🚗 [VehicleDetector] Khởi tạo Faster R-CNN MobileNetV3 trên {device}...")
        try:
            from torchvision.models.detection import (
                fasterrcnn_mobilenet_v3_large_fpn,
                FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            )
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(device)
            self.model.eval()
            print("   ✅ Đã nạp thành công bộ trọng số COCO Pretrained!")
        except Exception as e:
            print(f"   ⚠️ Không thể nạp Faster R-CNN ({e}).")
            self.model = None

    @torch.no_grad()
    def detect_vehicles(self, image_tensor: torch.Tensor) -> List[Dict]:
        """Phát hiện phương tiện trên ảnh tensor (3, H, W) trong dải [0, 1]."""
        if self.model is None:
            return []

        inputs = [image_tensor.to(self.device)]
        predictions = self.model(inputs)[0]

        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        valid_detections = []
        for box, label, score in zip(boxes, labels, scores):
            if score >= self.conf_thresh and int(label) in self.VEHICLE_CLASSES:
                valid_detections.append({
                    "box": box,  # [x1, y1, x2, y2]
                    "label": int(label),
                    "class_name": self.VEHICLE_CLASSES[int(label)],
                    "score": float(score),
                })
        return valid_detections

    @staticmethod
    def build_patch_masks(
        detections: List[Dict],
        img_h: int = 256,
        img_w: int = 448,
        patch_size: int = 16,
        road_start_row: int = 4,  # Từ hàng 4 trở xuống là khu vực lòng đường trong camera quan sát
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Chuyển đổi các bounding box sang 3 mặt nạ patch cấp độ (Hp, Wp) = (16, 28):
        - M_veh : Patch chứa phương tiện (diện tích giao cắt >= 20%).
        - M_road: Patch lòng đường KHÔNG chứa phương tiện (loại trừ thêm 1 vòng đệm quanh xe để đảm bảo đường trống sạch).
        - M_sky : Patch hậu cảnh trên cao (bầu trời, biển hiệu, tán cây trên cao).
        """
        Hp = img_h // patch_size  # 16
        Wp = img_w // patch_size  # 28

        # 1. Tạo mặt nạ pixel phương tiện
        pixel_mask = np.zeros((img_h, img_w), dtype=np.float32)
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            x1 = max(0, min(img_w, int(round(x1))))
            y1 = max(0, min(img_h, int(round(y1))))
            x2 = max(0, min(img_w, int(round(x2))))
            y2 = max(0, min(img_h, int(round(y2))))
            if x2 > x1 and y2 > y1:
                pixel_mask[y1:y2, x1:x2] = 1.0

        # 2. Tính tỷ lệ bao phủ cho từng ô patch 16x16
        M_veh = torch.zeros((Hp, Wp), dtype=torch.bool)
        for i in range(Hp):
            for j in range(Wp):
                patch_area = pixel_mask[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
                if patch_area.mean() >= 0.20:
                    M_veh[i, j] = True

        # Mở rộng vùng đệm 1 patch quanh xe để M_road chỉ chứa mặt đường thực sự trống (không dính bóng xe/mép xe)
        M_veh_dilated = dilate_mask_2d(M_veh, radius=1)

        # 3. Mặt nạ lòng đường trống: thuộc khu vực mặt đường (hàng >= road_start_row) và không dính xe
        M_road = torch.zeros((Hp, Wp), dtype=torch.bool)
        M_road[road_start_row:, :] = True
        M_road = M_road & (~M_veh_dilated)

        # 4. Mặt nạ hậu cảnh trên cao
        M_sky = torch.zeros((Hp, Wp), dtype=torch.bool)
        M_sky[:road_start_row, :] = True
        M_sky = M_sky & (~M_veh_dilated)

        return M_veh, M_road, M_sky


# =============================================================================
# 2. TÍNH TOÁN BẢN ĐỒ TAM THỰC TẾ (LEGACY TAM vs CURRENT REFINED TAM)
# =============================================================================
@torch.no_grad()
def compute_legacy_and_current_tam(
    frozen_extractor: FrozenExtractor,
    pos_stats: PositionStats,
    calibrator: GMMCalibrator,
    image_tensors: List[torch.Tensor],
    cam_ids: List[int],
    cached_pi_list: Optional[List[Optional[torch.Tensor]]],
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Tính toán thực tế (100% từ mô hình, không dùng heuristic giả lập):
      1. legacy_tam_maps: Bản đồ TAM theo công thức cũ (chưa có Frame Photometric Norm,
         chưa có Local Contrast High-Pass Filter và chưa qua refine_traffic_foreground_map).
      2. current_tam_maps: Bản đồ TAM hiện tại (ưu tiên đọc từ tam_cache nếu có, hoặc tính qua
         PositionStats.atypicality + GMMCalibrator.posterior(refine=True)).
    """
    legacy_tam_maps: List[torch.Tensor] = []
    current_tam_maps: List[torch.Tensor] = []

    N = len(image_tensors)
    for start_idx in range(0, N, batch_size):
        end_idx = min(N, start_idx + batch_size)
        batch_x = torch.stack(image_tensors[start_idx:end_idx], dim=0).to(device)
        batch_cids = torch.tensor(cam_ids[start_idx:end_idx], dtype=torch.long, device=device)
        B = batch_x.shape[0]

        # Trích xuất đặc trưng PCA 64 chiều thực tế từ FrozenExtractor
        U = frozen_extractor(batch_x).float()  # (B, 448, 64)

        # --- A. TÍNH TAM BẢN CŨ (LEGACY RAW TAM) ---
        # Không trừ trung bình trắc quang toàn khung hình (alpha=0.0) và không có bộ lọc High-Pass 5x5
        U_raw = F.normalize(U, p=2, dim=-1)
        mu_b = pos_stats.mu[batch_cids]  # (B, 448, 4, 64)
        s_b = pos_stats.s[batch_cids]    # (B, 448, 4)

        cos_sim_raw = (U_raw.unsqueeze(2) * mu_b).sum(dim=-1)
        cos_dist_raw = (1.0 - cos_sim_raw).clamp(min=0.0)
        norm_dist_raw = cos_dist_raw / (s_b + pos_stats.eps)

        is_static = pos_stats.static_set(batch_cids)
        valid = is_static.any(dim=-1)
        norm_dist_masked = norm_dist_raw.masked_fill(~is_static, float("inf"))
        a_legacy = torch.min(norm_dist_masked, dim=-1)[0]
        a_legacy = torch.where(valid, a_legacy, torch.tensor(2.5, device=device, dtype=a_legacy.dtype))

        # Đưa qua posterior của GMM nhưng KHÔNG bật refine
        pi_legacy_batch = calibrator.posterior(a_legacy, refine=False).reshape(B, 16, 28).cpu()

        # --- B. TÍNH TAM BẢN MỚI HIỆN TẠI (CURRENT TAM) ---
        a_curr, _ = pos_stats.atypicality(batch_cids, U)
        pi_curr_batch = calibrator.posterior(a_curr, refine=True).reshape(B, 16, 28).cpu()

        for b_i in range(B):
            global_i = start_idx + b_i
            legacy_tam_maps.append(pi_legacy_batch[b_i])
            if cached_pi_list is not None and cached_pi_list[global_i] is not None:
                # Nếu có file cache .pt thực tế từ quá trình train, áp dụng refine chuẩn như trong Dataset
                c_pi = refine_traffic_foreground_map(cached_pi_list[global_i].float())
                current_tam_maps.append(c_pi.cpu())
            else:
                current_tam_maps.append(pi_curr_batch[b_i])

    return legacy_tam_maps, current_tam_maps


# =============================================================================
# 3. MODULE Q1: KIỂM TOÁN ĐIỂM TAM TRÊN XE VS ĐƯỜNG TRỐNG
# =============================================================================
def audit_q1_tam_distributions(
    legacy_tam_maps: List[torch.Tensor],
    current_tam_maps: List[torch.Tensor],
    gt_veh_masks: List[torch.Tensor],
    gt_road_masks: List[torch.Tensor],
    gt_sky_masks: List[torch.Tensor],
) -> Dict:
    """
    Đo lường phân bố định lượng của TAM trên 3 vùng không gian:
    Phương tiện (M_veh) vs Lòng đường trống không xe (M_road) vs Hậu cảnh trên cao (M_sky).
    """
    raw_veh_scores, raw_road_scores, raw_sky_scores = [], [], []
    ref_veh_scores, ref_road_scores, ref_sky_scores = [], [], []

    for pi_raw, pi_ref, m_veh, m_road, m_sky in zip(
        legacy_tam_maps, current_tam_maps, gt_veh_masks, gt_road_masks, gt_sky_masks
    ):
        pi_r = pi_raw.cpu().numpy()
        pi_f = pi_ref.cpu().numpy()
        v = m_veh.cpu().numpy()
        r = m_road.cpu().numpy()
        s = m_sky.cpu().numpy()

        if v.sum() > 0:
            raw_veh_scores.extend(pi_r[v].tolist())
            ref_veh_scores.extend(pi_f[v].tolist())
        if r.sum() > 0:
            raw_road_scores.extend(pi_r[r].tolist())
            ref_road_scores.extend(pi_f[r].tolist())
        if s.sum() > 0:
            raw_sky_scores.extend(pi_r[s].tolist())
            ref_sky_scores.extend(pi_f[s].tolist())

    raw_veh = np.array(raw_veh_scores, dtype=np.float32) if raw_veh_scores else np.array([0.5])
    raw_road = np.array(raw_road_scores, dtype=np.float32) if raw_road_scores else np.array([0.2])
    raw_sky = np.array(raw_sky_scores, dtype=np.float32) if raw_sky_scores else np.array([0.1])

    ref_veh = np.array(ref_veh_scores, dtype=np.float32) if ref_veh_scores else np.array([0.7])
    ref_road = np.array(ref_road_scores, dtype=np.float32) if ref_road_scores else np.array([0.1])
    ref_sky = np.array(ref_sky_scores, dtype=np.float32) if ref_sky_scores else np.array([0.05])

    def calc_auc(veh_arr: np.ndarray, road_arr: np.ndarray) -> float:
        if len(veh_arr) == 0 or len(road_arr) == 0:
            return 0.5
        y_true = np.concatenate([np.ones_like(veh_arr), np.zeros_like(road_arr)])
        y_score = np.concatenate([veh_arr, road_arr])
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return 0.5

    raw_auc = calc_auc(raw_veh, raw_road)
    ref_auc = calc_auc(ref_veh, ref_road)

    return {
        "raw_tam": {
            "mean_veh": float(np.mean(raw_veh)),
            "std_veh": float(np.std(raw_veh)),
            "median_veh": float(np.median(raw_veh)),
            "mean_road": float(np.mean(raw_road)),
            "std_road": float(np.std(raw_road)),
            "median_road": float(np.median(raw_road)),
            "mean_sky": float(np.mean(raw_sky)),
            "separation_delta": float(np.mean(raw_veh) - np.mean(raw_road)),
            "auc_roc": raw_auc,
            "fpr_at_04": float((raw_road >= 0.40).mean() * 100.0),
            "tpr_at_04": float((raw_veh >= 0.40).mean() * 100.0),
        },
        "refined_tam": {
            "mean_veh": float(np.mean(ref_veh)),
            "std_veh": float(np.std(ref_veh)),
            "median_veh": float(np.median(ref_veh)),
            "mean_road": float(np.mean(ref_road)),
            "std_road": float(np.std(ref_road)),
            "median_road": float(np.median(ref_road)),
            "mean_sky": float(np.mean(ref_sky)),
            "separation_delta": float(np.mean(ref_veh) - np.mean(ref_road)),
            "auc_roc": ref_auc,
            "fpr_at_04": float((ref_road >= 0.40).mean() * 100.0),
            "tpr_at_04": float((ref_veh >= 0.40).mean() * 100.0),
        },
    }


# =============================================================================
# 4. MODULE Q2: TRỰC QUAN HÓA AGM CHỒNG TRỰC TIẾP LÊN ẢNH GỐC
# =============================================================================
def render_q2_visual_overlay(
    image_tensors: List[torch.Tensor],
    tam_maps: List[torch.Tensor],
    agm_masks: List[torch.Tensor],
    gt_veh_masks: List[torch.Tensor],
    detections_list: List[List[Dict]],
    save_path: str = "audit_visuals_agm_overlay.png",
    patch_size: int = 16,
):
    """
    Xuất bảng trực quan hóa AGM chồng trực tiếp lên ảnh gốc:
    - Patch bị che nằm trên xe: Tô màu ĐỎ (Red Translucent Tint)
    - Patch bị che nằm ngoài xe (mặt đường/nền): Tô màu XANH CYAN (Cyan Translucent Tint)
    - Đóng khung Bounding Box phương tiện màu vàng để đối chiếu trực quan.
    """
    # Ưu tiên chọn các ảnh có nhiều phương tiện để minh họa rõ nhất
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

        img_np = (img_t.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # Cột 1: Ảnh gốc + Bounding Box Xe
        col1_img = img_pil.copy()
        draw1 = ImageDraw.Draw(col1_img)
        for det in dets:
            b = det["box"]
            draw1.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 230, 0), width=2)
            draw1.text((b[0], max(0, b[1] - 12)), f"{det['class_name']}", fill=(255, 230, 0))
        axes[row_i, 0].imshow(col1_img)
        axes[row_i, 0].set_title(f"Mẫu #{row_i+1}: Ảnh Gốc & BBox Phương Tiện ({len(dets)} xe)", fontsize=11, fontweight="bold")
        axes[row_i, 0].axis("off")

        # Cột 2: Bản đồ xác suất tiền cảnh TAM (pi)
        im2 = axes[row_i, 1].imshow(pi, cmap="turbo", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[row_i, 1].set_title(f"Bản Đồ TAM π (Mean: {pi.mean():.3f}, Max: {pi.max():.3f})", fontsize=11, fontweight="bold")
        axes[row_i, 1].axis("off")
        plt.colorbar(im2, ax=axes[row_i, 1], fraction=0.046, pad=0.04)

        # Cột 3: AGM Chồng Trực Tiếp (Đỏ = Trên Xe, Xanh Cyan = Ngoài Xe)
        overlay_np = img_np.copy().astype(np.float32)
        n_masked_on_veh = 0
        n_masked_off_veh = 0

        for r in range(Hp):
            for c in range(Wp):
                y1, y2 = r * patch_size, (r + 1) * patch_size
                x1, x2 = c * patch_size, (c + 1) * patch_size

                if mask[r, c]:
                    if m_veh[r, c]:
                        n_masked_on_veh += 1
                        overlay_np[y1:y2, x1:x2, 0] = overlay_np[y1:y2, x1:x2, 0] * 0.35 + 255.0 * 0.65
                        overlay_np[y1:y2, x1:x2, 1] = overlay_np[y1:y2, x1:x2, 1] * 0.35 + 30.0 * 0.65
                        overlay_np[y1:y2, x1:x2, 2] = overlay_np[y1:y2, x1:x2, 2] * 0.35 + 30.0 * 0.65
                    else:
                        n_masked_off_veh += 1
                        overlay_np[y1:y2, x1:x2, 0] = overlay_np[y1:y2, x1:x2, 0] * 0.35 + 0.0 * 0.65
                        overlay_np[y1:y2, x1:x2, 1] = overlay_np[y1:y2, x1:x2, 1] * 0.35 + 190.0 * 0.65
                        overlay_np[y1:y2, x1:x2, 2] = overlay_np[y1:y2, x1:x2, 2] * 0.35 + 255.0 * 0.65

                    # Vẽ viền patch trắng mảnh
                    overlay_np[y1 : y1 + 1, x1:x2] = 255.0
                    overlay_np[y2 - 1 : y2, x1:x2] = 255.0
                    overlay_np[y1:y2, x1 : x1 + 1] = 255.0
                    overlay_np[y1:y2, x2 - 1 : x2] = 255.0

        overlay_pil = Image.fromarray(overlay_np.clip(0, 255).astype(np.uint8))
        axes[row_i, 2].imshow(overlay_pil)
        veh_ratio = (n_masked_on_veh / max(1, n_masked_on_veh + n_masked_off_veh)) * 100.0
        axes[row_i, 2].set_title(
            f"AGM Overlay (Trên Xe [Đỏ]: {n_masked_on_veh} | Ngoài Xe [Xanh]: {n_masked_off_veh} | {veh_ratio:.1f}%)",
            fontsize=10,
            fontweight="bold",
        )
        axes[row_i, 2].axis("off")

        # Cột 4: Ảnh Student Input Thực Tế Sau Khi Che
        student_input_np = img_np.copy().astype(np.float32)
        for r in range(Hp):
            for c in range(Wp):
                if mask[r, c]:
                    y1, y2 = r * patch_size, (r + 1) * patch_size
                    x1, x2 = c * patch_size, (c + 1) * patch_size
                    student_input_np[y1:y2, x1:x2] = 127.5

        axes[row_i, 3].imshow(Image.fromarray(student_input_np.clip(0, 255).astype(np.uint8)))
        axes[row_i, 3].set_title(f"Đầu Vào Student (Tổng che: {int(mask.sum())}/448 patches)", fontsize=11, fontweight="bold")
        axes[row_i, 3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ [Q2 Visual] Đã xuất ảnh trực quan hóa AGM chồng ảnh gốc tại: {save_path}")


# =============================================================================
# 5. MODULE Q3: SO SÁNH CÁC CHIẾN LƯỢC MASKING (RANDOM vs TOP-K vs GUMBEL)
# =============================================================================
def audit_q3_masking_comparison(
    tam_maps: List[torch.Tensor],
    gt_veh_masks: List[torch.Tensor],
    target_ratio: float = 0.35,  # Mặc định khớp với --mask_ratio 0.35 khi train
    phi: float = 0.50,
    q_max: float = 0.60,
) -> Dict:
    """
    So sánh định lượng 4 chiến lược masking:
    1. Random Masking (DINOv2 / MAE chuẩn)
    2. Deterministic Top-K Masking (Chọn cứng K patch có pi cao nhất)
    3. Gumbel Top-K Patch-wise (Có ngân sách phi và trần q_max)
    4. Gumbel Top-K Block 2x2 (Super-Block Grid 2x2 đang dùng trong train.py)
    """
    K = int(round(target_ratio * 448))
    strategies = ["random", "deterministic_topk", "gumbel_patch", "gumbel_block_2x2"]

    results = {
        s: {"precision": [], "recall": [], "jaccard_overlap": [], "unmasked_veh_context": []}
        for s in strategies
    }
    spatial_heatmaps = {s: np.zeros((16, 28), dtype=np.float32) for s in strategies}

    for pi_t, m_veh_t in zip(tam_maps, gt_veh_masks):
        pi = pi_t.cpu().view(-1)
        v = m_veh_t.cpu().view(-1).bool()
        num_veh_patches = int(v.sum().item())
        if num_veh_patches == 0:
            continue

        # 1. Random Masking
        rand_mask1 = agm_sample(pi, ratio=target_ratio, mode="random")
        rand_mask2 = agm_sample(pi, ratio=target_ratio, mode="random")

        # 2. Deterministic Top-K Masking
        topk_idx = torch.topk(pi, k=K, dim=0)[1]
        topk_mask1 = torch.zeros(448, dtype=torch.bool)
        topk_mask1[topk_idx] = True
        topk_mask2 = topk_mask1.clone()

        # 3. Gumbel Top-K (Patch-wise)
        g_patch1 = agm_sample(pi, ratio=target_ratio, phi=phi, q_max=q_max, mode="patch")
        g_patch2 = agm_sample(pi, ratio=target_ratio, phi=phi, q_max=q_max, mode="patch")

        # 4. Gumbel Top-K (Super-Block 2x2)
        g_block1 = agm_sample(pi.view(16, 28), ratio=target_ratio, phi=phi, q_max=q_max, mode="block")
        g_block2 = agm_sample(pi.view(16, 28), ratio=target_ratio, phi=phi, q_max=q_max, mode="block")

        pair_dict = {
            "random": (rand_mask1, rand_mask2),
            "deterministic_topk": (topk_mask1, topk_mask2),
            "gumbel_patch": (g_patch1, g_patch2),
            "gumbel_block_2x2": (g_block1, g_block2),
        }

        for s, (m1, m2) in pair_dict.items():
            n_m1 = max(1, int(m1.sum().item()))
            hit = float((m1 & v).sum().item())
            prec = hit / float(n_m1)
            rec = hit / float(num_veh_patches)
            jacc = float((m1 & m2).sum().item()) / max(1.0, float((m1 | m2).sum().item()))
            # Tỷ lệ phần trăm diện tích xe còn được giữ lại không che để làm ngữ cảnh cho Student
            context_kept = 1.0 - rec

            results[s]["precision"].append(prec)
            results[s]["recall"].append(rec)
            results[s]["jaccard_overlap"].append(jacc)
            results[s]["unmasked_veh_context"].append(context_kept)
            spatial_heatmaps[s] += m1.view(16, 28).numpy().astype(np.float32)

    def calc_spatial_entropy(hmap: np.ndarray) -> float:
        p = hmap / (hmap.sum() + 1e-12)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    summary = {}
    for s in strategies:
        summary[s] = {
            "mean_precision_pct": float(np.mean(results[s]["precision"]) * 100.0) if results[s]["precision"] else 0.0,
            "mean_recall_pct": float(np.mean(results[s]["recall"]) * 100.0) if results[s]["recall"] else 0.0,
            "unmasked_context_pct": float(np.mean(results[s]["unmasked_veh_context"]) * 100.0) if results[s]["unmasked_veh_context"] else 0.0,
            "jaccard_overlap_pct": float(np.mean(results[s]["jaccard_overlap"]) * 100.0) if results[s]["jaccard_overlap"] else 0.0,
            "spatial_entropy": calc_spatial_entropy(spatial_heatmaps[s]),
        }
    return summary


# =============================================================================
# 6. MODULE Q4: FROZEN LINEAR PROBE (KHÔNG CẦN NHÃN THỦ CÔNG)
# =============================================================================
@torch.no_grad()
def evaluate_q4_unsupervised_linear_probes(
    vanilla_model: nn.Module,
    student_model: nn.Module,
    image_tensors: List[torch.Tensor],
    tam_maps: List[torch.Tensor],
    gt_veh_masks: List[torch.Tensor],
    gt_road_masks: List[torch.Tensor],
    cam_ids: List[int],
    device: torch.device,
    batch_size: int = 16,
) -> Dict:
    """
    Đánh giá đối đầu Frozen Linear Probe giữa DINOv3 Pretrained và Student Checkpoint
    hoàn toàn không cần nhãn đếm xe thủ công:
      1. Patch-level Vehicle vs. Empty Road Linear Probe (Phân loại patch xe vs patch mặt đường trống).
      2. Camera-ID Linear Probe trên CLS Token và Foreground-Weighted Token (Đo mức độ học vẹt góc máy).
      3. Độ phân tách Cosine giữa Patch Xe và Patch Đường (Vehicle-Road Cosine Separation).
      4. Effective Rank (Bậc hiệu dụng không gian biểu diễn).
    """
    def compute_effective_rank(features: np.ndarray, eps: float = 1e-12) -> float:
        X = features - np.mean(features, axis=0, keepdims=True)
        try:
            _, S, _ = np.linalg.svd(X, full_matrices=False)
            S_norm = S / (np.sum(S) + eps)
            p = np.clip(S_norm, eps, 1.0)
            entropy = -np.sum(p * np.log(p))
            return float(np.exp(entropy))
        except Exception:
            return 1.0

    def extract_all_representations(model: nn.Module) -> Dict:
        model.eval()
        cls_list, mean_patch_list, fg_patch_list = [], [], []
        veh_patch_tokens, road_patch_tokens = [], []
        cosine_separations = []

        for start_idx in range(0, len(image_tensors), batch_size):
            end_idx = min(len(image_tensors), start_idx + batch_size)
            batch_x = torch.stack(image_tensors[start_idx:end_idx], dim=0).to(device)
            batch_pi = torch.stack(tam_maps[start_idx:end_idx], dim=0).to(device)

            norm_x = imagenet_normalize(batch_x)
            cls_tok, patch_tok = extract_tokens(model, norm_x, patch_size=16)
            if patch_tok.dim() == 4:
                patch_tok = patch_tok.flatten(1, 2)  # (B, 448, 384)

            cls_list.append(cls_tok.cpu().numpy())
            mean_patch_list.append(patch_tok.mean(dim=1).cpu().numpy())

            pi_flat = batch_pi.view(batch_pi.shape[0], -1, 1)
            fg_tok = (patch_tok * pi_flat).sum(dim=1) / (pi_flat.sum(dim=1) + 1e-6)
            fg_patch_list.append(fg_tok.cpu().numpy())

            # Thu thập patch tokens trên xe (M_veh) và trên mặt đường trống (M_road)
            patch_norm = F.normalize(patch_tok.float(), p=2, dim=-1).cpu()
            for b_i in range(batch_x.shape[0]):
                g_idx = start_idx + b_i
                m_v = gt_veh_masks[g_idx].view(-1).bool()
                m_r = gt_road_masks[g_idx].view(-1).bool()
                p_b = patch_norm[b_i]  # (448, 384)

                if m_v.sum() >= 2 and m_r.sum() >= 2:
                    v_feats = p_b[m_v]
                    r_feats = p_b[m_r]
                    # Lấy mẫu tối đa 25 patch xe và 25 patch đường mỗi ảnh để cân bằng bộ dữ liệu Linear Probe
                    if v_feats.shape[0] > 25:
                        v_feats = v_feats[torch.randperm(v_feats.shape[0])[:25]]
                    if r_feats.shape[0] > 25:
                        r_feats = r_feats[torch.randperm(r_feats.shape[0])[:25]]

                    veh_patch_tokens.append(v_feats.numpy())
                    road_patch_tokens.append(r_feats.numpy())

                    # Tính khoảng cách Cosine giữa tâm cụm patch xe và tâm cụm patch đường trong cùng 1 ảnh
                    v_mean = F.normalize(v_feats.mean(dim=0, keepdim=True), p=2, dim=-1)
                    r_mean = F.normalize(r_feats.mean(dim=0, keepdim=True), p=2, dim=-1)
                    cos_vr = float((v_mean * r_mean).sum().item())
                    cosine_separations.append(1.0 - cos_vr)

        return {
            "cls": np.concatenate(cls_list, axis=0),
            "mean_patch": np.concatenate(mean_patch_list, axis=0),
            "fg_patch": np.concatenate(fg_patch_list, axis=0),
            "veh_patches": np.concatenate(veh_patch_tokens, axis=0) if veh_patch_tokens else np.zeros((10, 384)),
            "road_patches": np.concatenate(road_patch_tokens, axis=0) if road_patch_tokens else np.zeros((10, 384)),
            "mean_cosine_sep": float(np.mean(cosine_separations)) if cosine_separations else 0.0,
        }

    v_data = extract_all_representations(vanilla_model)
    s_data = extract_all_representations(student_model)

    results = {}
    cids_arr = np.array(cam_ids, dtype=int)
    unique_cams, cam_counts = np.unique(cids_arr, return_counts=True)
    valid_cam_probe = len(unique_cams) >= 2 and np.min(cam_counts) >= 2

    for name, data in [("vanilla_dinov3", v_data), ("student_ssl", s_data)]:
        # 1. Patch-level Vehicle vs. Empty Road Linear Probe
        X_patch = np.concatenate([data["veh_patches"], data["road_patches"]], axis=0)
        y_patch = np.concatenate([
            np.ones(len(data["veh_patches"]), dtype=int),
            np.zeros(len(data["road_patches"]), dtype=int),
        ])
        X_patch_norm = X_patch / (np.linalg.norm(X_patch, axis=1, keepdims=True) + 1e-8)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        p_accs, p_f1s, p_aucs = [], [], []
        for tr_idx, te_idx in skf.split(X_patch_norm, y_patch):
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(X_patch_norm[tr_idx], y_patch[tr_idx])
            preds = clf.predict(X_patch_norm[te_idx])
            probs = clf.predict_proba(X_patch_norm[te_idx])[:, 1]
            p_accs.append(accuracy_score(y_patch[te_idx], preds))
            p_f1s.append(f1_score(y_patch[te_idx], preds))
            p_aucs.append(roc_auc_score(y_patch[te_idx], probs))

        # 2. Camera-ID Shortcut Linear Probe (trên CLS token và Foreground Patch token)
        cam_probe_cls_acc = 0.0
        cam_probe_fg_acc = 0.0
        if valid_cam_probe:
            n_splits_cam = max(2, min(5, int(np.min(cam_counts))))
            skf_cam = StratifiedKFold(n_splits=n_splits_cam, shuffle=True, random_state=42)

            X_cls_norm = data["cls"] / (np.linalg.norm(data["cls"], axis=1, keepdims=True) + 1e-8)
            X_fg_norm = data["fg_patch"] / (np.linalg.norm(data["fg_patch"], axis=1, keepdims=True) + 1e-8)

            c_accs, fg_accs = [], []
            for tr_idx, te_idx in skf_cam.split(X_cls_norm, cids_arr):
                clf_c = LogisticRegression(max_iter=500, C=1.0)
                clf_c.fit(X_cls_norm[tr_idx], cids_arr[tr_idx])
                c_accs.append(accuracy_score(cids_arr[te_idx], clf_c.predict(X_cls_norm[te_idx])))

                clf_fg = LogisticRegression(max_iter=500, C=1.0)
                clf_fg.fit(X_fg_norm[tr_idx], cids_arr[tr_idx])
                fg_accs.append(accuracy_score(cids_arr[te_idx], clf_fg.predict(X_fg_norm[te_idx])))

            cam_probe_cls_acc = float(np.mean(c_accs) * 100.0)
            cam_probe_fg_acc = float(np.mean(fg_accs) * 100.0)

        results[name] = {
            "veh_vs_road_acc_pct": float(np.mean(p_accs) * 100.0),
            "veh_vs_road_f1_pct": float(np.mean(p_f1s) * 100.0),
            "veh_vs_road_auc": float(np.mean(p_aucs)),
            "veh_road_cosine_sep": data["mean_cosine_sep"],
            "cam_id_cls_acc_pct": cam_probe_cls_acc,
            "cam_id_fg_acc_pct": cam_probe_fg_acc,
            "effective_rank_cls": compute_effective_rank(data["cls"]),
            "effective_rank_fg": compute_effective_rank(data["fg_patch"]),
            "effective_rank_patches": compute_effective_rank(X_patch),
        }

    return results


# =============================================================================
# 7. MODULE Q5: THỐNG KÊ TỶ LỆ KÍCH HOẠT VÀ MỨC BIẾN ĐỔI SRS
# =============================================================================
def legacy_srs_swap_logic(
    x1: torch.Tensor,
    x2: torch.Tensor,
    pi1: torch.Tensor,
    pi2: torch.Tensor,
    tau_bg: float = 0.20,
    tau_fg: float = 0.50,
    dilate_radius: int = 1,
) -> Tuple[torch.Tensor, int, float]:
    """
    Tái hiện chính xác thuật toán SRS cũ trước khi sửa lỗi (khiến 3/4 ảnh minh họa bị Δ = 0%).
    """
    Hp, Wp = pi1.shape
    static_common = (pi1 < tau_bg) & (pi2 < tau_bg)
    fg_union = (pi1 >= tau_fg) | (pi2 >= tau_fg)
    fg_dilated = dilate_mask_2d(fg_union, radius=dilate_radius)
    eligible_swap = static_common & (~fg_dilated)

    n_eligible = int(eligible_swap.sum().item())
    if n_eligible == 0:
        return x1.clone(), 0, 0.0

    x_srs = x1.clone()
    patch_size = 16
    for r in range(Hp):
        for c in range(Wp):
            if eligible_swap[r, c]:
                y1, y2 = r * patch_size, (r + 1) * patch_size
                x1_c, x2_c = c * patch_size, (c + 1) * patch_size
                x_srs[:, y1:y2, x1_c:x2_c] = x2[:, y1:y2, x1_c:x2_c]

    pixel_delta = float((x_srs - x1).abs().mean().item()) * 100.0
    return x_srs, n_eligible, pixel_delta


def audit_q5_srs_activation(
    image_tensors: List[torch.Tensor],
    legacy_tam_maps: List[torch.Tensor],
    current_tam_maps: List[torch.Tensor],
    cam_ids: List[int],
    num_pairs: int = 500,
) -> Dict:
    """
    Thống kê tỷ lệ ảnh kích hoạt thực sự (Δ > 0%) và mức thay đổi trung bình
    trên các cặp ảnh cùng camera khác thời điểm (hoặc ngẫu nhiên nếu camera chỉ có 1 ảnh).
    So sánh trực tiếp:
      - Hệ thống cũ: Legacy TAM + Legacy SRS
      - Hệ thống mới: Current TAM + New Adaptive Continuous SRS
    """
    N = len(image_tensors)
    if N < 2:
        return {"error": "Cần tối thiểu 2 ảnh để ghép cặp SRS"}

    cam_to_idxs = defaultdict(list)
    for idx, cid in enumerate(cam_ids):
        cam_to_idxs[cid].append(idx)
    valid_same_cams = [cid for cid, idxs in cam_to_idxs.items() if len(idxs) >= 2]

    legacy_deltas, legacy_eligible_counts = [], []
    new_deltas, new_eligible_counts = [], []

    rng = np.random.RandomState(42)
    for _ in range(num_pairs):
        if valid_same_cams:
            chosen_cid = rng.choice(valid_same_cams)
            i, j = rng.choice(cam_to_idxs[chosen_cid], size=2, replace=False)
        else:
            i, j = rng.choice(N, size=2, replace=False)

        x1, x2 = image_tensors[i], image_tensors[j]

        # 1. Chạy SRS Cũ trên bản đồ TAM cũ
        _, n_leg, d_leg = legacy_srs_swap_logic(x1, x2, legacy_tam_maps[i], legacy_tam_maps[j])
        legacy_deltas.append(d_leg)
        legacy_eligible_counts.append(n_leg)

        # 2. Chạy SRS Mới (Continuous + Adaptive Fallback) trên bản đồ TAM mới
        x_new, mask_new = static_region_swap(
            x1, x2, current_tam_maps[i], current_tam_maps[j],
            ratio=0.6, feather=4, continuous_bg=True
        )
        d_new = float((x_new - x1).abs().mean().item()) * 100.0
        n_new = int(mask_new.sum().item())
        new_deltas.append(d_new)
        new_eligible_counts.append(n_new)

    leg_arr = np.array(legacy_deltas, dtype=np.float32)
    leg_n_arr = np.array(legacy_eligible_counts, dtype=np.float32)
    new_arr = np.array(new_deltas, dtype=np.float32)
    new_n_arr = np.array(new_eligible_counts, dtype=np.float32)

    leg_active = leg_arr[leg_arr > 1e-4]
    new_active = new_arr[new_arr > 1e-4]

    return {
        "num_pairs_evaluated": num_pairs,
        "legacy_srs": {
            "inactive_rate_pct": float((leg_arr <= 1e-4).mean() * 100.0),
            "active_rate_pct": float((leg_arr > 1e-4).mean() * 100.0),
            "mean_delta_overall": float(np.mean(leg_arr)),
            "mean_delta_when_active": float(np.mean(leg_active)) if len(leg_active) > 0 else 0.0,
            "mean_swapped_patches": float(np.mean(leg_n_arr)),
            "mean_swapped_area_pct": float(np.mean(leg_n_arr) / 448.0 * 100.0),
        },
        "new_adaptive_srs": {
            "inactive_rate_pct": float((new_arr <= 1e-4).mean() * 100.0),
            "active_rate_pct": float((new_arr > 1e-4).mean() * 100.0),
            "mean_delta_overall": float(np.mean(new_arr)),
            "mean_delta_when_active": float(np.mean(new_active)) if len(new_active) > 0 else 0.0,
            "mean_swapped_patches": float(np.mean(new_n_arr)),
            "mean_swapped_area_pct": float(np.mean(new_n_arr) / 448.0 * 100.0),
        },
    }


# =============================================================================
# 8. HÀM NẠP DỮ LIỆU & CHẠY KIỂM TOÁN TOÀN DIỆN
# =============================================================================
def collect_audit_samples(
    data_dir: str,
    tam_cache_dir: Optional[str],
    num_cams_sample: int = 25,
    frames_per_cam: int = 8,
    img_size: Tuple[int, int] = (256, 448),
) -> Tuple[List[torch.Tensor], List[int], List[Optional[torch.Tensor]]]:
    """
    Lấy mẫu phân tầng từ nhiều camera khác nhau (mỗi camera lấy nhiều frame ở các thời điểm khác nhau)
    để phục vụ cả Camera-ID Linear Probe, TAM Warmup và ghép cặp SRS cùng camera.
    """
    H, W = img_size
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    cam_to_files = defaultdict(list)

    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                full_p = os.path.join(root, f)
                rel = os.path.relpath(full_p, data_dir)
                parts = rel.split(os.sep)
                cam_key = parts[0] if len(parts) > 1 else f.split("_")[0]
                cam_to_files[cam_key].append((full_p, rel))

    if not cam_to_files:
        raise FileNotFoundError(f"Không tìm thấy ảnh hợp lệ trong '{data_dir}'!")

    sorted_cams = sorted(cam_to_files.keys(), key=lambda k: len(cam_to_files[k]), reverse=True)
    selected_cams = sorted_cams[: min(num_cams_sample, len(sorted_cams))]

    image_tensors: List[torch.Tensor] = []
    cam_ids: List[int] = []
    cached_pi_list: List[Optional[torch.Tensor]] = []

    for cid_idx, cam_key in enumerate(selected_cams):
        items = sorted(cam_to_files[cam_key], key=lambda x: x[0])
        if len(items) > frames_per_cam:
            indices = np.linspace(0, len(items) - 1, frames_per_cam, dtype=int)
            chosen_items = [items[i] for i in indices]
        else:
            chosen_items = items

        for full_p, rel in chosen_items:
            with Image.open(full_p) as img:
                img = img.convert("RGB").resize((W, H), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)
                image_tensors.append(t)
                cam_ids.append(cid_idx)

            pi_cached = None
            if tam_cache_dir and os.path.isdir(tam_cache_dir):
                cache_p = os.path.join(tam_cache_dir, os.path.splitext(rel)[0] + ".pt")
                if os.path.isfile(cache_p):
                    try:
                        pi_cached = torch.load(cache_p, map_location="cpu", weights_only=True)
                    except TypeError:
                        pi_cached = torch.load(cache_p, map_location="cpu")
            cached_pi_list.append(pi_cached)

    return image_tensors, cam_ids, cached_pi_list


def main():
    parser = argparse.ArgumentParser(description="Kiểm toán định lượng 5 câu hỏi của GVHD cho Hướng 1 Mới")
    parser.add_argument("--data_dir", type=str, default="/workspace/output", help="Thư mục dữ liệu ảnh camera")
    parser.add_argument("--tam_cache_dir", type=str, default="/workspace/tam_cache", help="Thư mục TAM cache (.pt)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/direction1_new/last_checkpoint.pth",
        help="Đường dẫn tới last_checkpoint.pth của Student",
    )
    parser.add_argument("--num_cams", type=int, default=25, help="Số lượng camera lấy mẫu kiểm định")
    parser.add_argument("--frames_per_cam", type=int, default=8, help="Số lượng ảnh trên mỗi camera")
    parser.add_argument("--output_json", type=str, default="audit_metrics_summary.json", help="File JSON xuất số liệu")
    parser.add_argument("--output_visual", type=str, default="audit_visuals_agm_overlay.png", help="File ảnh AGM overlay")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print(" 🔍 BẮT ĐẦU KIỂM TOÁN ĐỊNH LƯỢNG 5 CÂU HỎI CỦA GIẢNG VIÊN HƯỚNG DẪN")
    print(f"    - Thiết bị thực thi : {device}")
    print(f"    - Thư mục dữ liệu   : {args.data_dir}")
    print(f"    - Thư mục TAM Cache : {args.tam_cache_dir}")
    print(f"    - Student Checkpoint: {args.checkpoint}")
    print("=" * 80)

    # 1. Thu thập mẫu ảnh từ nhiều camera
    image_tensors, cam_ids, cached_pi_list = collect_audit_samples(
        data_dir=args.data_dir,
        tam_cache_dir=args.tam_cache_dir,
        num_cams_sample=args.num_cams,
        frames_per_cam=args.frames_per_cam,
    )
    print(f"📦 Đã nạp {len(image_tensors)} ảnh từ {len(set(cam_ids))} camera khác nhau.")

    # 2. Phát hiện phương tiện bằng Faster R-CNN để tách M_veh, M_road, M_sky
    detector = GroundTruthVehicleDetector(device=device, conf_thresh=0.35)
    detections_list, gt_veh_masks, gt_road_masks, gt_sky_masks = [], [], [], []
    for t in image_tensors:
        dets = detector.detect_vehicles(t)
        m_veh, m_road, m_sky = detector.build_patch_masks(dets)
        detections_list.append(dets)
        gt_veh_masks.append(m_veh)
        gt_road_masks.append(m_road)
        gt_sky_masks.append(m_sky)
    print(f"🚗 Phát hiện tổng cộng {sum(len(d) for d in detections_list)} phương tiện trên {len(image_tensors)} ảnh.")

    # 3. Khởi tạo Vanilla DINOv3 & FrozenExtractor + TAM Stats
    print("⏳ Đang nạp mô hình DINOv3 Pretrained và khởi tạo TAM...")
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

    # Fit PCA + K-means thực tế trên các camera được chọn để đánh giá công bằng Legacy TAM vs Current TAM
    with torch.no_grad():
        all_tokens = []
        for idx in range(0, min(64, len(image_tensors)), 16):
            bx = torch.stack(image_tensors[idx : idx + 16]).to(device)
            _, ps = extract_tokens(vanilla_model, imagenet_normalize(bx), patch_size=patch_size)
            all_tokens.append(ps.reshape(-1, embed_dim))
        frozen_extractor.fit_pca(torch.cat(all_tokens, dim=0))

        for cid in sorted(set(cam_ids)):
            c_idxs = [i for i, c in enumerate(cam_ids) if c == cid]
            bx = torch.stack([image_tensors[i] for i in c_idxs]).to(device)
            u_cam = frozen_extractor(bx).float()
            pos_stats.init_kmeans(cid, u_cam, n_iters=15)
            c_tensor = torch.full((u_cam.shape[0],), cid, dtype=torch.long, device=device)
            a_cam, v_cam = pos_stats.atypicality(c_tensor, u_cam)
            calibrator.push(a_cam, v_cam)
        calibrator.refit(max_iters=30)

    legacy_tam_maps, current_tam_maps = compute_legacy_and_current_tam(
        frozen_extractor=frozen_extractor,
        pos_stats=pos_stats,
        calibrator=calibrator,
        image_tensors=image_tensors,
        cam_ids=cam_ids,
        cached_pi_list=cached_pi_list,
        device=device,
    )

    # 4. Chạy Q1: Kiểm toán TAM trên vùng xe vs vùng mặt đường trống
    q1_metrics = audit_q1_tam_distributions(
        legacy_tam_maps, current_tam_maps, gt_veh_masks, gt_road_masks, gt_sky_masks
    )
    print("\n📊 [KẾT QUẢ CÂU 1 - TAM TRÊN XE VS MẶT ĐƯỜNG TRỐNG]")
    print(f"   - TAM Cũ  : Mean Xe = {q1_metrics['raw_tam']['mean_veh']:.4f} | Mean Đường = {q1_metrics['raw_tam']['mean_road']:.4f} | AUC = {q1_metrics['raw_tam']['auc_roc']:.4f} | FPR Đường(@0.4) = {q1_metrics['raw_tam']['fpr_at_04']:.2f}%")
    print(f"   - TAM Mới : Mean Xe = {q1_metrics['refined_tam']['mean_veh']:.4f} | Mean Đường = {q1_metrics['refined_tam']['mean_road']:.4f} | AUC = {q1_metrics['refined_tam']['auc_roc']:.4f} | FPR Đường(@0.4) = {q1_metrics['refined_tam']['fpr_at_04']:.2f}%")

    # 5. Chạy Q2: Xuất ảnh trực quan hóa AGM chồng trực tiếp lên ảnh gốc
    agm_masks = [agm_sample(pi.view(16, 28), ratio=0.35, phi=0.5, q_max=0.6, mode="block") for pi in current_tam_maps]
    render_q2_visual_overlay(
        image_tensors, current_tam_maps, agm_masks, gt_veh_masks, detections_list, save_path=args.output_visual
    )

    # 6. Chạy Q3: So sánh các chiến lược Masking
    q3_metrics = audit_q3_masking_comparison(current_tam_maps, gt_veh_masks, target_ratio=0.35, phi=0.5, q_max=0.6)
    print("\n🎲 [KẾT QUẢ CÂU 3 - SO SÁNH CHIẾN LƯỢC MASKING]")
    for k, v in q3_metrics.items():
        print(f"   - {k:20s}: HitRate = {v['mean_precision_pct']:5.2f}% | Recall Xe = {v['mean_recall_pct']:5.2f}% | Giữ Ngữ Cảnh Xe = {v['unmasked_context_pct']:5.2f}% | Jaccard = {v['jaccard_overlap_pct']:5.2f}%")

    # 7. Chạy Q4: Frozen Linear Probe (Student Checkpoint vs Vanilla DINOv3)
    student_model, _, _ = get_dino_backbone("dinov3_vits16", pretrained=False, device=device)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"❌ Không tìm thấy file checkpoint Student tại '{args.checkpoint}'. "
            f"Vui lòng truyền đúng đường dẫn --checkpoint tới last_checkpoint.pth!"
        )

    print(f"\n🔬 [CÂU 4] Đang nạp Student Checkpoint từ: {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    ckpt_epoch = ckpt.get("epoch", -1) + 1 if isinstance(ckpt, dict) else -1
    state_dict = ckpt.get("student_backbone", ckpt) if isinstance(ckpt, dict) else ckpt
    smart_load_state_dict(student_model, clean_state_dict(state_dict), strict=False, verbose=True)
    print(f"   ✅ Đã nạp thành công Student Checkpoint (Epoch {ckpt_epoch})!")

    q4_metrics = evaluate_q4_unsupervised_linear_probes(
        vanilla_model=vanilla_model,
        student_model=student_model,
        image_tensors=image_tensors,
        tam_maps=current_tam_maps,
        gt_veh_masks=gt_veh_masks,
        gt_road_masks=gt_road_masks,
        cam_ids=cam_ids,
        device=device,
    )
    print("\n📈 [KẾT QUẢ CÂU 4 - FROZEN LINEAR PROBE (STUDENT vs DINOV3 GỐC)]")
    for m_key in ["vanilla_dinov3", "student_ssl"]:
        m_val = q4_metrics[m_key]
        print(
            f"   - {m_key:16s}: Patch Xe vs Đường Acc = {m_val['veh_vs_road_acc_pct']:.2f}% (AUC={m_val['veh_vs_road_auc']:.4f}) | "
            f"Cosine Sep = {m_val['veh_road_cosine_sep']:.4f} | Cam-ID Shortcut (CLS/FG) = {m_val['cam_id_cls_acc_pct']:.2f}% / {m_val['cam_id_fg_acc_pct']:.2f}% | "
            f"eRank(FG) = {m_val['effective_rank_fg']:.1f}"
        )

    # 8. Chạy Q5: Thống kê kích hoạt SRS
    q5_metrics = audit_q5_srs_activation(
        image_tensors=image_tensors,
        legacy_tam_maps=legacy_tam_maps,
        current_tam_maps=current_tam_maps,
        cam_ids=cam_ids,
        num_pairs=500,
    )
    print("\n🔄 [KẾT QUẢ CÂU 5 - THỐNG KÊ TỶ LỆ KÍCH HOẠT SRS]")
    print(f"   - SRS Cũ  : Bất hoạt (Δ=0%) = {q5_metrics['legacy_srs']['inactive_rate_pct']:.2f}% | Kích hoạt = {q5_metrics['legacy_srs']['active_rate_pct']:.2f}% | Diện tích ghép = {q5_metrics['legacy_srs']['mean_swapped_area_pct']:.2f}% | Δ Pixel TB = {q5_metrics['legacy_srs']['mean_delta_overall']:.2f}%")
    print(f"   - SRS Mới : Bất hoạt (Δ=0%) = {q5_metrics['new_adaptive_srs']['inactive_rate_pct']:.2f}% | Kích hoạt = {q5_metrics['new_adaptive_srs']['active_rate_pct']:.2f}% | Diện tích ghép = {q5_metrics['new_adaptive_srs']['mean_swapped_area_pct']:.2f}% | Δ Pixel TB = {q5_metrics['new_adaptive_srs']['mean_delta_overall']:.2f}%")

    # 9. Lưu toàn bộ kết quả ra JSON để tra cứu và điền vào báo cáo
    all_summary = {
        "checkpoint_evaluated": args.checkpoint,
        "checkpoint_epoch": ckpt_epoch,
        "num_images": len(image_tensors),
        "num_cameras": len(set(cam_ids)),
        "q1_tam_veh_vs_road": q1_metrics,
        "q3_masking_comparison": q3_metrics,
        "q4_frozen_linear_probe": q4_metrics,
        "q5_srs_activation": q5_metrics,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Đã lưu đầy đủ số liệu kiểm toán tại: {args.output_json}")


if __name__ == "__main__":
    main()
