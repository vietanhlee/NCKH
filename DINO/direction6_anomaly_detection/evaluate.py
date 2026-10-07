import os
import json
import argparse
import sys
from pathlib import Path

# Chống xung đột OpenMP và hỗ trợ tiếng Việt trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from direction6_anomaly_detection.features import DINOv3PatchFeatureExtractor
from direction6_anomaly_detection.pooling import TemporalFeaturePooler
from direction6_anomaly_detection.bank import NormalMemoryBank
from direction6_anomaly_detection.score import AnomalyScorer
from direction6_anomaly_detection.camera_fault import CameraFaultClassifier
from direction6_anomaly_detection.events import PersistenceEventTracker
from direction6_anomaly_detection.synth_events import SyntheticAnomalyGenerator


def evaluate_anomaly_detection_pipeline(
    extractor: DINOv3PatchFeatureExtractor,
    memory_bank: NormalMemoryBank,
    test_sequence: torch.Tensor,
    road_mask: torch.Tensor,
    gt_frame_labels: np.ndarray,
    window_size: int = 5,
    min_consecutive: int = 3,
    save_dir: Optional[str] = "checkpoints/direction6_anomaly_detection",
) -> Dict[str, float]:
    """
    Chạy toàn bộ pipeline kiểm thử đánh giá trên một chuỗi video thử nghiệm.

    Args:
        extractor: Bộ trích xuất DINOv3 đã cấu hình.
        memory_bank: Ngân hàng đặc trưng bình thường đã nén coreset.
        test_sequence: Chuỗi khung hình (T, 3, H, W).
        road_mask: Mặt nạ mặt đường (H, W).
        gt_frame_labels: Mảng nhãn (T,) với 1 là bất thường, 0 là bình thường.
        window_size: Kích thước cửa sổ trượt W.
        min_consecutive: Số bước liên tiếp tối thiểu N.
        save_dir: Thư mục lưu kết quả metrics JSON và biểu đồ trực quan hóa.
    """
    device = test_sequence.device
    T, C, H, W = test_sequence.shape

    pooler = TemporalFeaturePooler(window_size=window_size)
    scorer = AnomalyScorer(k_nearest=1, top_k_ratio=0.05)
    classifier = CameraFaultClassifier()
    tracker = PersistenceEventTracker(min_consecutive_windows=min_consecutive)

    # 1. Trích xuất đặc trưng patch và hạ mẫu road mask
    with torch.no_grad():
        all_patch_feats, h_p, w_p = extractor.extract_patch_tokens(test_sequence)
        patch_road_mask = extractor.downsample_mask_to_patches(road_mask, h_p, w_p).to(device)

    # 2. Hiệu chuẩn ngưỡng từ tập bình thường (các frame có gt = 0)
    normal_indices = np.where(gt_frame_labels == 0)[0]
    normal_scores = []
    
    with torch.no_grad():
        for idx in normal_indices[:min(len(normal_indices), 20)]:
            feat = all_patch_feats[idx:idx+1]
            p_scores = scorer.compute_patch_scores(feat, memory_bank.bank)
            r_score, _ = scorer.aggregate_frame_score(p_scores, patch_road_mask[0:1])
            normal_scores.append(r_score.item())

    if len(normal_scores) > 0:
        tracker.calibrate_threshold(np.array(normal_scores))
    else:
        tracker.threshold = 0.40

    # 3. Chạy qua chuỗi thời gian kiểm thử
    raw_road_scores = []
    raw_static_scores = []
    all_patch_scores = []
    confirmed_anomaly_flags = []
    detection_step = None

    for t in range(T):
        patch_feat_t = all_patch_feats[t:t+1]
        pooled_feat_t = pooler.update(patch_feat_t)

        with torch.no_grad():
            patch_scores = scorer.compute_patch_scores(pooled_feat_t, memory_bank.bank)
            road_score, static_score = scorer.aggregate_frame_score(
                patch_scores, patch_road_mask[0:1]
            )

        r_val = float(road_score.item())
        s_val = float(static_score.item())
        raw_road_scores.append(r_val)
        raw_static_scores.append(s_val)
        all_patch_scores.append(patch_scores.detach().cpu())

        diag = classifier.classify(r_val, s_val, tracker.threshold, tracker.threshold * 1.1)
        alert = tracker.update(step_idx=t, score=r_val, event_type=diag["event_type"])

        is_confirmed = 1 if (alert is not None and alert.get("status") == "CONFIRMED") else 0
        confirmed_anomaly_flags.append(is_confirmed)

        if is_confirmed and detection_step is None and gt_frame_labels[t] == 1:
            detection_step = t

    raw_road_scores = np.array(raw_road_scores)
    raw_static_scores = np.array(raw_static_scores)
    confirmed_anomaly_flags = np.array(confirmed_anomaly_flags)

    # 4. Tính AUROC
    try:
        auroc = float(roc_auc_score(gt_frame_labels, raw_road_scores))
    except Exception:
        auroc = 0.50

    # Tính độ trễ phát hiện (Detection Delay)
    first_gt_step = np.where(gt_frame_labels == 1)[0]
    if len(first_gt_step) > 0 and detection_step is not None:
        delay = max(0, detection_step - int(first_gt_step[0]))
    else:
        delay = -1.0

    print("📊 [Direction 6: Anomaly Detection Performance]")
    print(f"   Frame-level AUROC:      {auroc:.4f}")
    print(f"   Calibrated Threshold:   {tracker.threshold:.4f}")
    print(f"   Detection Delay Steps:  {delay} steps")
    print(f"   Total Confirmed Alerts: {int(confirmed_anomaly_flags.sum())}")

    results = {
        "auroc": auroc,
        "threshold": tracker.threshold,
        "detection_delay": delay,
        "total_alerts": int(confirmed_anomaly_flags.sum()),
        "mean_normal_score": float(np.mean(raw_road_scores[gt_frame_labels == 0])) if (gt_frame_labels == 0).any() else 0.0,
        "mean_anomaly_score": float(np.mean(raw_road_scores[gt_frame_labels == 1])) if (gt_frame_labels == 1).any() else 0.0,
    }

    # 5. Lưu metrics JSON và biểu đồ trực quan hóa
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        metrics_path = os.path.join(save_dir, "anomaly_metrics.json")
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"📁 [Metrics] Đã lưu báo cáo đánh giá sự cố tại: {metrics_path}")
        except Exception as e_m:
            print(f"⚠️ [Metrics Warning] {e_m}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Tìm frame đại diện bất thường và bình thường
            anomaly_indices = np.where(gt_frame_labels == 1)[0]
            rep_anom_idx = int(anomaly_indices[0]) if len(anomaly_indices) > 0 else int(np.argmax(raw_road_scores))
            rep_norm_idx = int(normal_indices[0]) if len(normal_indices) > 0 else 0

            # Tạo Heatmap spatial của frame bất thường
            anom_patch_score = all_patch_scores[rep_anom_idx]  # (1, N)
            heatmap_tensor = scorer.generate_heatmap(anom_patch_score, h_p, w_p, target_size=(H, W))
            heatmap_np = heatmap_tensor[0, 0].numpy()

            fig, axs = plt.subplots(1, 4, figsize=(20, 4.8), dpi=150)

            # Panel (a): Frame RGB so sánh
            anom_frame = test_sequence[rep_anom_idx].permute(1, 2, 0).cpu().numpy()
            anom_frame = (anom_frame - anom_frame.min()) / (anom_frame.max() - anom_frame.min() + 1e-6)
            axs[0].imshow(anom_frame)
            axs[0].set_title(f"(a) Incident Frame (t={rep_anom_idx})\n(Injected/Real Anomaly)", fontsize=11, fontweight="bold")
            axs[0].axis("off")

            # Panel (b): DINOv3 Patch Anomaly Heatmap
            im_heat = axs[1].imshow(heatmap_np, cmap="jet")
            axs[1].set_title(f"(b) DINOv3 Patch Anomaly Heatmap\nScore: {raw_road_scores[rep_anom_idx]:.3f}", fontsize=11, fontweight="bold")
            axs[1].axis("off")
            plt.colorbar(im_heat, ax=axs[1], fraction=0.046, pad=0.04)

            # Panel (c): Anomaly Score Timeline vs Threshold
            time_axis = np.arange(T)
            axs[2].plot(time_axis, raw_road_scores, "b-o", linewidth=1.5, markersize=4, label="Road Anomaly Score")
            axs[2].axhline(y=tracker.threshold, color="r", linestyle="--", linewidth=1.8, label=f"Threshold ($\\tau={tracker.threshold:.2f}$)")
            if len(anomaly_indices) > 0:
                axs[2].axvspan(anomaly_indices[0], anomaly_indices[-1], color="orange", alpha=0.25, label="GT Anomaly Interval")
            alert_steps = np.where(confirmed_anomaly_flags == 1)[0]
            if len(alert_steps) > 0:
                axs[2].plot(alert_steps, raw_road_scores[alert_steps], "r*", markersize=12, label="Confirmed Alert")
            axs[2].set_xlabel("Video Frame Step (t)")
            axs[2].set_ylabel("Anomaly Distance")
            axs[2].set_title(f"(c) Detection Timeline (AUROC: {auroc:.3f})", fontsize=11, fontweight="bold")
            axs[2].grid(True, linestyle="--", alpha=0.5)
            axs[2].legend(fontsize=8, loc="upper left")

            # Panel (d): Phân tách Lỗi Camera vs Sự cố Giao thông (Spatial Diagnosis)
            axs[3].scatter(raw_road_scores[gt_frame_labels == 0], raw_static_scores[gt_frame_labels == 0], 
                           c="green", alpha=0.7, marker="o", label="Normal Frames")
            if (gt_frame_labels == 1).any():
                axs[3].scatter(raw_road_scores[gt_frame_labels == 1], raw_static_scores[gt_frame_labels == 1], 
                               c="red", alpha=0.8, marker="^", s=50, label="Incident Frames")
            axs[3].axvline(x=tracker.threshold, color="r", linestyle=":", alpha=0.6)
            axs[3].axhline(y=tracker.threshold * 1.1, color="purple", linestyle=":", alpha=0.6)
            axs[3].set_xlabel("Road Anomaly Score ($s_{road}$)")
            axs[3].set_ylabel("Static Non-road Score ($s_{static}$)")
            axs[3].set_title("(d) Diagnosis: Incident vs Camera Fault", fontsize=11, fontweight="bold")
            axs[3].grid(True, linestyle="--", alpha=0.5)
            axs[3].legend(fontsize=8, loc="lower right")

            plt.tight_layout()
            chart_path = os.path.join(save_dir, "anomaly_evaluation.png")
            plt.savefig(chart_path, bbox_inches="tight")
            plt.close()
            print(f"📊 [Charts] Đã lưu biểu đồ đánh giá sự cố tại: {chart_path}")
        except Exception as e_plot:
            print(f"⚠️ [Anomaly Chart Warning] {e_plot}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direction 6 Anomaly Detection Evaluation")
    parser.add_argument("--save_dir", type=str, default="checkpoints/direction6_anomaly_detection", help="Thư mục lưu báo cáo")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị tính toán ('cuda' hoặc 'cpu')")
    cli_args, _ = parser.parse_known_args()

    try:
        from common.gpu_utils import get_available_devices
        primary_dev, num_gpus, gpu_names = get_available_devices()
    except Exception:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        primary_dev = torch.device("cuda:0" if num_gpus > 0 else "cpu")
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]

    device = primary_dev if "cuda" in cli_args.device.lower() and num_gpus > 0 else torch.device("cpu")
    print(f"🚀 Đang chạy kiểm thử Direction 6 Anomaly Detection Evaluation Pipeline trên {device} (Số GPU: {num_gpus})...")

    class DummyBackbone(torch.nn.Module):
        def forward(self, x):
            B, C, H, W = x.shape
            N = (H // 16) * (W // 16)
            return torch.randn(B, N, 384, device=x.device)

    backbone = DummyBackbone().to(device)
    extractor = DINOv3PatchFeatureExtractor(backbone, feature_dim=384, proj_dim=128)

    bank = NormalMemoryBank("cam_01", "morning_peak", feature_dim=128)
    sim_normal_feats = torch.randn(500, 128)
    bank.fit_coreset(sim_normal_feats, subsampling_ratio=0.2)

    T = 25
    test_seq = torch.randn(T, 3, 256, 448, device=device)
    road_mask = torch.ones(256, 448, device=device)
    test_seq, _ = SyntheticAnomalyGenerator.inject_static_obstacle(test_seq, road_mask, start_t=10, duration=8)
    gt_labels = np.zeros(T, dtype=int)
    gt_labels[10:18] = 1

    results = evaluate_anomaly_detection_pipeline(
        extractor, bank, test_seq, road_mask, gt_labels, window_size=3, min_consecutive=2, save_dir=cli_args.save_dir
    )
    print("✅ Hoàn tất kiểm thử Direction 6!")
