"""
=============================================================================
 Hướng 7: Traffic Forecasting — Evaluation & Robustness Stress Test
 Đánh giá định lượng trên các chân trời h = 15', 30', 60'
 và Khảo sát độ bền khi hệ thống camera mất tín hiệu 10% - 50%
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
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
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

    return results, all_preds, all_targets, all_masks


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


def run_full_forecasting_evaluation(
    model: CityScaleTrafficForecastingModel,
    data_loader: torch.utils.data.DataLoader,
    physical_adj: torch.Tensor,
    device: torch.device,
    save_dir: Optional[str] = "checkpoints/direction7_traffic_forecasting",
) -> Dict[str, Any]:
    """
    Chạy đánh giá toàn diện Hướng 7, lưu metrics JSON và trực quan hóa 4 biểu đồ.
    """
    horizon_results, all_preds, all_targets, all_masks = evaluate_forecasting_horizons(
        model, data_loader, physical_adj, device
    )
    stress_results = run_missing_camera_stress_test(
        model, data_loader, physical_adj, device
    )

    combined_report = {
        "horizons": horizon_results,
        "stress_test": stress_results,
        "num_nodes": int(physical_adj.shape[0]),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        metrics_path = os.path.join(save_dir, "forecasting_metrics.json")
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(combined_report, f, indent=2, ensure_ascii=False)
            print(f"📁 [Metrics] Đã lưu báo cáo dự báo lưu lượng tại: {metrics_path}")
        except Exception as e_m:
            print(f"⚠️ [Metrics Warning] {e_m}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(1, 4, figsize=(21, 4.8), dpi=150)

            # Panel (a): Ma trận kề đồ thị mạng lưới camera
            adj_np = physical_adj.detach().cpu().numpy()
            im0 = axs[0].imshow(adj_np, cmap="viridis")
            axs[0].set_title(f"(a) Camera Graph Adjacency\n({adj_np.shape[0]} Nodes)", fontsize=11, fontweight="bold")
            axs[0].set_xlabel("Camera Node Index")
            axs[0].set_ylabel("Camera Node Index")
            plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            # Panel (b): Đường cong dự báo đa tầm (15', 30', 60')
            # Lấy 1 node đại diện trên chuỗi thời gian
            sample_node = 0
            t_steps = np.arange(1, all_preds.shape[-1] + 1) * 5  # mỗi bước 5 phút -> 5, 10, ..., 60 phút
            axs[1].plot(t_steps, all_targets[0, sample_node, :], "k-o", linewidth=2, label="Ground Truth Flow")
            axs[1].plot(t_steps, all_preds[0, sample_node, :], "r--s", linewidth=2, label="ST-GNN Predicted")
            axs[1].axvline(x=15, color="green", linestyle=":", alpha=0.7, label="15 min (Step 3)")
            axs[1].axvline(x=30, color="orange", linestyle=":", alpha=0.7, label="30 min (Step 6)")
            axs[1].axvline(x=60, color="purple", linestyle=":", alpha=0.7, label="60 min (Step 12)")
            axs[1].set_xlabel("Forecast Horizon (minutes)")
            axs[1].set_ylabel("Traffic Flow Rate")
            axs[1].set_title(f"(b) Multi-Horizon Forecast\n(Node #{sample_node})", fontsize=11, fontweight="bold")
            axs[1].grid(True, linestyle="--", alpha=0.5)
            axs[1].legend(fontsize=8, loc="upper right")

            # Panel (c): Khảo sát độ bền camera rớt mạng (Stress test)
            rates = [0, 10, 20, 30, 50]
            stress_maes = [stress_results.get(f"failure_{r}pct", 0.0) for r in rates]
            axs[2].plot(rates, stress_maes, "m-D", linewidth=2, markersize=7)
            axs[2].set_xlabel("Camera Offline Rate (%)")
            axs[2].set_ylabel("Overall Forecasting MAE")
            axs[2].set_title("(c) Missing Camera Stress Test\n(Robustness Degradation)", fontsize=11, fontweight="bold")
            axs[2].grid(True, linestyle="--", alpha=0.5)
            for r, m in zip(rates, stress_maes):
                axs[2].annotate(f"{m:.3f}", (r, m), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

            # Panel (d): Sai số MAE vs RMSE theo từng chân trời
            h_labels = list(horizon_results.keys())
            mae_vals = [horizon_results[h]["mae"] for h in h_labels]
            rmse_vals = [horizon_results[h]["rmse"] for h in h_labels]
            x_pos = np.arange(len(h_labels))
            bar_w = 0.35
            axs[3].bar(x_pos - bar_w/2, mae_vals, bar_w, label="MAE", color="#1f77b4")
            axs[3].bar(x_pos + bar_w/2, rmse_vals, bar_w, label="RMSE", color="#ff7f0e")
            axs[3].set_xticks(x_pos)
            axs[3].set_xticklabels(h_labels)
            axs[3].set_xlabel("Horizon Window")
            axs[3].set_ylabel("Error Metric")
            axs[3].set_title("(d) Horizon Error Comparison\n(15' vs 30' vs 60')", fontsize=11, fontweight="bold")
            axs[3].grid(True, linestyle="--", alpha=0.5, axis="y")
            axs[3].legend(fontsize=8)

            plt.tight_layout()
            chart_path = os.path.join(save_dir, "forecasting_results.png")
            plt.savefig(chart_path, bbox_inches="tight")
            plt.close()
            print(f"📊 [Charts] Đã lưu biểu đồ dự báo tại: {chart_path}")
        except Exception as e_plot:
            print(f"⚠️ [Forecasting Chart Warning] {e_plot}")

    return combined_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direction 7 Traffic Forecasting Evaluation")
    parser.add_argument("--save_dir", type=str, default="checkpoints/direction7_traffic_forecasting", help="Thư mục lưu báo cáo")
    cli_args, _ = parser.parse_known_args()

    print("🚀 Đang khởi chạy kiểm thử Direction 7 Traffic Forecasting Suite...")
    N = 20
    T_in = 12
    T_out = 12
    model = CityScaleTrafficForecastingModel(num_nodes=N, in_channels=5, hidden_channels=16, out_steps=T_out)
    phys_adj = torch.eye(N)

    B = 4
    dummy_x = torch.randn(B, 5, N, T_in)
    dummy_y = torch.rand(B, N, T_out)
    dummy_m = torch.ones(B, N, T_out)

    dataset = [{"x_input": dummy_x[i], "y_target": dummy_y[i], "m_target": dummy_m[i]} for i in range(B)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    device = torch.device("cpu")

    run_full_forecasting_evaluation(model, loader, phys_adj, device, save_dir=cli_args.save_dir)
    print("✅ Hoàn tất kiểm thử Direction 7!")
