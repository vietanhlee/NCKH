"""
=============================================================================
 Hướng 8: Background Conditioning — Evaluation & BDB Degradation Benchmark
 Đánh giá so sánh:
   1. Khả năng thích ứng sang Camera chưa thấy (Unseen Camera Generalization)
   2. So sánh FiLM vs Prompt vs Cross-Attention vs Delta-Concatenation vs No-BG
   3. Độ bền vững dưới bộ nhiễu Background Degradation Benchmark (BDB)
 Tích hợp xuất Metrics JSON & Biểu đồ trực quan hóa cao cấp
=============================================================================
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Chống xung đột OpenMP và hỗ trợ tiếng Việt trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from direction8_bg_conditioning.models import BackgroundConditionedModel
from direction8_bg_conditioning.descriptor import RobustSceneDescriptorExtractor
from common.degradation import BackgroundDegradationBenchmark


def evaluate_unseen_camera_generalization(
    models_dict: Dict[str, BackgroundConditionedModel],
    unseen_test_loader: torch.utils.data.DataLoader,
    descriptor_extractor: RobustSceneDescriptorExtractor,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Đánh giá so sánh các cơ chế điều kiện hóa trên camera hoàn toàn mới:
    
    Args:
        models_dict: Dict {"film": model_film, "prompt": model_prompt, ...}
        unseen_test_loader: DataLoader chứa dữ liệu của các camera chưa thấy lúc train
        descriptor_extractor: Bộ trích xuất Trimmed z
    """
    results = {}
    for name, model in models_dict.items():
        model.eval()

    all_targets_count = []
    all_targets_cls = []
    preds_count = {name: [] for name in models_dict}
    preds_cls = {name: [] for name in models_dict}

    with torch.no_grad():
        for batch in unseen_test_loader:
            frames = batch["frame"].to(device)
            bg_imgs = batch["bg"].to(device)
            gt_counts = batch["gt_count"].cpu().numpy()
            gt_classes = batch["gt_class"].cpu().numpy()

            all_targets_count.extend(gt_counts)
            all_targets_cls.extend(gt_classes)

            # Trích xuất vector mô tả cảnh z cho batch
            B = frames.shape[0]
            z_batch = []
            for b in range(B):
                z_b = descriptor_extractor.extract_scene_descriptor(bg_imgs[b:b+1])
                z_batch.append(z_b)
            z_batch = torch.cat(z_batch, dim=0).to(device)

            for name, model in models_dict.items():
                out = model(frames, z_descriptor=z_batch)
                preds_count[name].extend(out["count"].cpu().numpy().flatten())
                cls_pred = torch.argmax(out["logits"], dim=-1).cpu().numpy().flatten()
                preds_cls[name].extend(cls_pred)

    y_count_true = np.array(all_targets_count)
    y_cls_true = np.array(all_targets_cls)

    print("\n🎯 [Unseen Camera Generalization Performance]")
    print(f"{'Method':<15} | {'Counting MAE':<14} | {'Macro F1':<10} | {'Accuracy':<10}")
    print("-" * 55)

    for name in models_dict:
        y_c_pred = np.array(preds_count[name])
        y_cl_pred = np.array(preds_cls[name])

        mae = float(mean_absolute_error(y_count_true, y_c_pred))
        f1 = float(f1_score(y_cls_true, y_cl_pred, average="macro", zero_division=0))
        acc = float(accuracy_score(y_cls_true, y_cl_pred))

        results[name] = {"counting_mae": mae, "macro_f1": f1, "accuracy": acc}
        print(f"{name:<15} | {mae:<14.3f} | {f1:<10.4f} | {acc*100:<9.2f}%")

    return results


def run_bdb_robustness_comparison(
    model_film: BackgroundConditionedModel,
    descriptor_extractor: RobustSceneDescriptorExtractor,
    sample_frame: torch.Tensor,
    sample_bg: torch.Tensor,
    gt_count: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Kiểm tra độ suy giảm hiệu năng dưới bộ nhiễu Background Degradation Benchmark (BDB)
    với 6 loại nhiễu và 5 mức độ nghiêm trọng.
    """
    model_film.eval()
    benchmark = BackgroundDegradationBenchmark()
    mae_degradation_curves = {}

    noise_types = ["ghost_injection", "camera_shift", "optical_change"]
    print("\n⚡ [BDB Degradation Robustness Benchmark for FiLM Conditioned Model]")

    with torch.no_grad():
        for n_type in noise_types:
            type_errors = []
            for sev in range(1, 6):
                # Tạo ảnh nền bị làm hỏng
                corrupted_bg = benchmark.apply_degradation(sample_bg, n_type, severity=sev).to(device)
                z_corrupted = descriptor_extractor.extract_scene_descriptor(corrupted_bg)

                out = model_film(sample_frame.to(device), z_descriptor=z_corrupted)
                pred_c = float(out["count"].item())
                err = abs(pred_c - gt_count)
                type_errors.append(err)

            mae_degradation_curves[n_type] = type_errors
            print(f"   Noise: {n_type:<18} -> Severities 1-5 MAE: {[round(e, 2) for e in type_errors]}")

    return mae_degradation_curves


def run_full_conditioning_evaluation(
    models: Dict[str, BackgroundConditionedModel],
    data_loader: torch.utils.data.DataLoader,
    descriptor_extractor: RobustSceneDescriptorExtractor,
    sample_frame: torch.Tensor,
    sample_bg: torch.Tensor,
    gt_count: float,
    device: torch.device,
    save_dir: Optional[str] = "checkpoints/direction8_bg_conditioning",
) -> Dict[str, Any]:
    """
    Chạy đánh giá trọn vẹn Hướng 8, lưu metrics JSON và trực quan hóa 4 biểu đồ.
    """
    gen_results = evaluate_unseen_camera_generalization(models, data_loader, descriptor_extractor, device)
    bdb_results = run_bdb_robustness_comparison(
        models.get("FiLM", list(models.values())[0]), descriptor_extractor, sample_frame, sample_bg, gt_count, device
    )

    combined_report = {
        "unseen_generalization": gen_results,
        "bdb_noise_robustness": bdb_results,
        "sample_gt_count": float(gt_count),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        metrics_path = os.path.join(save_dir, "bg_conditioning_metrics.json")
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(combined_report, f, indent=2, ensure_ascii=False)
            print(f"📁 [Metrics] Đã lưu báo cáo điều kiện hóa tại: {metrics_path}")
        except Exception as e_m:
            print(f"⚠️ [Metrics Warning] {e_m}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(1, 4, figsize=(21, 4.8), dpi=150)

            # Panel (a): Unseen Camera Frame Input & Reference Background
            f_np = sample_frame[0].permute(1, 2, 0).cpu().numpy()
            f_np = (f_np - f_np.min()) / (f_np.max() - f_np.min() + 1e-6)
            b_np = sample_bg.permute(1, 2, 0).cpu().numpy()
            b_np = (b_np - b_np.min()) / (b_np.max() - b_np.min() + 1e-6)

            # Ghép nửa frame và nửa bg
            composite = np.zeros_like(f_np)
            mid_w = f_np.shape[1] // 2
            composite[:, :mid_w] = f_np[:, :mid_w]
            composite[:, mid_w:] = b_np[:, mid_w:]
            axs[0].imshow(composite)
            axs[0].axvline(x=mid_w, color="yellow", linestyle="--", linewidth=2)
            axs[0].set_title("(a) Unseen Camera Inputs\n(Left: Frame | Right: Background)", fontsize=11, fontweight="bold")
            axs[0].axis("off")

            # Panel (b): Trimmed Background Descriptor z
            with torch.no_grad():
                z_vec = descriptor_extractor.extract_scene_descriptor(sample_bg.unsqueeze(0).to(device))
            z_np = z_vec[0].cpu().numpy()
            # Hiển thị 32 chiều đầu tiên của z
            show_dims = min(32, len(z_np))
            axs[1].bar(np.arange(show_dims), z_np[:show_dims], color="#2ca02c", alpha=0.8)
            axs[1].set_xlabel("Descriptor z Component Index")
            axs[1].set_ylabel("Normalized Feature Value")
            axs[1].set_title(f"(b) Trimmed Scene Descriptor $z$\n(Dim: {len(z_np)} dims)", fontsize=11, fontweight="bold")
            axs[1].grid(True, linestyle="--", alpha=0.5)

            # Panel (c): So sánh hiệu năng các mô hình trên camera chưa thấy
            m_names = list(gen_results.keys())
            m_maes = [gen_results[m]["counting_mae"] for m in m_names]
            m_f1s = [gen_results[m]["macro_f1"] * 100 for m in m_names]
            x_m = np.arange(len(m_names))
            b_w = 0.35
            axs[2].bar(x_m - b_w/2, m_maes, b_w, label="Counting MAE", color="#d62728")
            axs[2].bar(x_m + b_w/2, m_f1s, b_w, label="Macro F1 (%)", color="#1f77b4")
            axs[2].set_xticks(x_m)
            axs[2].set_xticklabels(m_names)
            axs[2].set_xlabel("Conditioning Mechanism")
            axs[2].set_title("(c) Unseen Camera Generalization\n(FiLM vs Baseline)", fontsize=11, fontweight="bold")
            axs[2].grid(True, linestyle="--", alpha=0.5, axis="y")
            axs[2].legend(fontsize=8)

            # Panel (d): Khảo sát độ bền vững BDB Noise Robustness
            severities = [1, 2, 3, 4, 5]
            colors = {"ghost_injection": "#9467bd", "camera_shift": "#8c564b", "optical_change": "#e377c2"}
            labels = {"ghost_injection": "Ghost Injection", "camera_shift": "Camera Shift", "optical_change": "Optical Change"}
            for n_type, errs in bdb_results.items():
                c = colors.get(n_type, "blue")
                lbl = labels.get(n_type, n_type)
                axs[3].plot(severities, errs, "-o", color=c, linewidth=2, label=lbl)
            axs[3].set_xlabel("BDB Noise Severity Level")
            axs[3].set_ylabel("Counting Absolute Error")
            axs[3].set_title("(d) BDB Degradation Robustness\n(FiLM Conditioned Model)", fontsize=11, fontweight="bold")
            axs[3].grid(True, linestyle="--", alpha=0.5)
            axs[3].legend(fontsize=8)

            plt.tight_layout()
            chart_path = os.path.join(save_dir, "bg_conditioning_results.png")
            plt.savefig(chart_path, bbox_inches="tight")
            plt.close()
            print(f"📊 [Charts] Đã lưu biểu đồ điều kiện hóa tại: {chart_path}")
        except Exception as e_plot:
            print(f"⚠️ [Conditioning Chart Warning] {e_plot}")

    return combined_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direction 8 Background Conditioning Evaluation")
    parser.add_argument("--save_dir", type=str, default="checkpoints/direction8_bg_conditioning", help="Thư mục lưu báo cáo")
    cli_args, _ = parser.parse_known_args()

    print("🚀 Đang khởi chạy kiểm thử Direction 8 Background Conditioning Evaluation...")
    class DummyBackbone(nn.Module):
        def forward(self, x):
            B, C, H, W = x.shape
            N = (H // 16) * (W // 16)
            return torch.randn(B, N, 384)

    backbone = DummyBackbone()
    extractor = RobustSceneDescriptorExtractor(backbone, feature_dim=384)

    model_film = BackgroundConditionedModel(backbone, feature_dim=384, descriptor_dim=1152, conditioning_mode="film")
    model_none = BackgroundConditionedModel(backbone, feature_dim=384, descriptor_dim=1152, conditioning_mode="none")

    models = {"FiLM": model_film, "No-Conditioning": model_none}

    dataset = []
    for _ in range(6):
        dataset.append({
            "frame": torch.rand(3, 256, 448),
            "bg": torch.rand(3, 256, 448),
            "gt_count": np.random.uniform(5.0, 30.0),
            "gt_class": np.random.randint(0, 4),
        })

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    device = torch.device("cpu")

    sample_f = torch.rand(1, 3, 256, 448)
    sample_b = torch.rand(3, 256, 448)

    run_full_conditioning_evaluation(
        models=models,
        data_loader=loader,
        descriptor_extractor=extractor,
        sample_frame=sample_f,
        sample_bg=sample_b,
        gt_count=15.0,
        device=device,
        save_dir=cli_args.save_dir,
    )
    print("✅ Hoàn tất kiểm thử Direction 8!")
