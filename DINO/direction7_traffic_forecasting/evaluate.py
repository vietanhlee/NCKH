"""
=============================================================================
 Hướng 7: Traffic Forecasting — Evaluation & Robustness Stress Test
 Đánh giá định lượng trên các chân trời h = 15', 30', 60'
 và Khảo sát độ bền khi hệ thống camera mất tín hiệu 10% - 50%
=============================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from direction7_traffic_forecasting.models import CityScaleTrafficForecastingModel


def compute_masked_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """
    Tính MAE, RMSE trên các vị trí quan sát được (mask == 1).
    """
    valid = mask > 0.5
    if np.sum(valid) == 0:
        return {"mae": 0.0, "rmse": 0.0}

    diff = pred[valid] - target[valid]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return {"mae": mae, "rmse": rmse}


def evaluate_forecasting_horizons(
    model: CityScaleTrafficForecastingModel,
    data_loader: torch.utils.data.DataLoader,
    physical_adj: torch.Tensor,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Đánh giá chi tiết theo từng tầm dự báo:
      - 15 phút (bước t = 2)
      - 30 phút (bước t = 5)
      - 60 phút (bước t = 11)
    """
    model.eval()
    physical_adj = physical_adj.to(device)

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in data_loader:
            x_in = batch["x_input"].to(device)
            y_tar = batch["y_target"].cpu().numpy()
            m_tar = batch["m_target"].cpu().numpy()

            out = model(x_in, physical_adj)
            pred = out["continuous_pred"].cpu().numpy()

            all_preds.append(pred)
            all_targets.append(y_tar)
            all_masks.append(m_tar)

    all_preds = np.concatenate(all_preds, axis=0)      # (B_tot, N, T_out)
    all_targets = np.concatenate(all_targets, axis=0)  # (B_tot, N, T_out)
    all_masks = np.concatenate(all_masks, axis=0)      # (B_tot, N, T_out)

    results = {}
    horizons = {
        "15min": 2,   # bước 3 (chỉ số 2)
        "30min": 5,   # bước 6 (chỉ số 5)
        "60min": 11,  # bước 12 (chỉ số 11)
    }

    print("\n📈 [Multi-Horizon Forecasting Performance]")
    for h_name, h_step in horizons.items():
        if h_step < all_preds.shape[-1]:
            p_h = all_preds[:, :, h_step]
            t_h = all_targets[:, :, h_step]
            m_h = all_masks[:, :, h_step]

            metrics = compute_masked_metrics(p_h, t_h, m_h)
            results[h_name] = metrics
            print(f"   Horizon {h_name:6s} -> MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f}")

    return results


def run_missing_camera_stress_test(
    model: CityScaleTrafficForecastingModel,
    data_loader: torch.utils.data.DataLoader,
    physical_adj: torch.Tensor,
    device: torch.device,
    failure_rates: List[float] = [0.0, 0.10, 0.20, 0.30, 0.50],
) -> Dict[str, float]:
    """
    Thử nghiệm độ bền hệ thống khi camera rớt mạng ngẫu nhiên từ 10% đến 50%.
    """
    model.eval()
    physical_adj = physical_adj.to(device)
    stress_results = {}

    print("\n⚡ [Missing Camera Robustness Stress Test]")
    for rate in failure_rates:
        maes = []
        with torch.no_grad():
            for batch in data_loader:
                x_in = batch["x_input"].to(device)  # (B, C, N, T_in)
                y_tar = batch["y_target"].cpu().numpy()
                m_tar = batch["m_target"].cpu().numpy()

                # Mô phỏng rớt mạng: che ngẫu nhiên tỷ lệ `rate` camera
                B, C, N, T = x_in.shape
                if rate > 0:
                    keep_mask = (torch.rand(B, 1, N, 1, device=device) >= rate).float()
                    x_in_corrupted = x_in * keep_mask
                else:
                    x_in_corrupted = x_in

                out = model(x_in_corrupted, physical_adj)
                pred = out["continuous_pred"].cpu().numpy()

                m = compute_masked_metrics(pred, y_tar, m_tar)
                maes.append(m["mae"])

        avg_mae = float(np.mean(maes))
        key = f"failure_{int(rate * 100)}pct"
        stress_results[key] = avg_mae
        print(f"   Camera Offline {int(rate*100):2d}% -> Overall MAE: {avg_mae:.4f}")

    return stress_results


if __name__ == "__main__":
    print("Testing Direction D Evaluation Pipeline...")
    N = 20
    T_in = 12
    T_out = 12
    model = CityScaleTrafficForecastingModel(num_nodes=N, in_channels=5, hidden_channels=16, out_steps=T_out)
    phys_adj = torch.eye(N)

    # Dữ liệu giả lập
    B = 4
    dummy_x = torch.randn(B, 5, N, T_in)
    dummy_y = torch.rand(B, N, T_out)
    dummy_m = torch.ones(B, N, T_out)

    dataset = [{"x_input": dummy_x[i], "y_target": dummy_y[i], "m_target": dummy_m[i]} for i in range(B)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    device = torch.device("cpu")
    evaluate_forecasting_horizons(model, loader, phys_adj, device)
    run_missing_camera_stress_test(model, loader, phys_adj, device)
