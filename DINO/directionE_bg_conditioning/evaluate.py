"""
=============================================================================
 Hướng E: Background Conditioning — Evaluation & BDB Degradation Benchmark
 Đánh giá so sánh:
   1. Khả năng thích ứng sang Camera chưa thấy (Unseen Camera Generalization)
   2. So sánh FiLM vs Prompt vs Cross-Attention vs Delta-Concatenation vs No-BG
   3. Độ bền vững dưới bộ nhiễu Background Degradation Benchmark (BDB)
=============================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from directionE_bg_conditioning.models import BackgroundConditionedModel
from directionE_bg_conditioning.descriptor import RobustSceneDescriptorExtractor
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


if __name__ == "__main__":
    print("Testing Direction E Evaluation Pipeline...")
    # Demo mock backbone
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

    # Giả lập 6 mẫu test
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

    evaluate_unseen_camera_generalization(models, loader, extractor, device)

    # Chạy thử BDB
    sample_f = torch.rand(1, 3, 256, 448)
    sample_b = torch.rand(3, 256, 448)
    run_bdb_robustness_comparison(model_film, extractor, sample_f, sample_b, gt_count=15.0, device=device)
