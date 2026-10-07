"""
Thực nghiệm kiểm toán sâu: Phân tích phân bố khoảng cách hành lang kề cận,
độ cao lắp đặt camera và phân vùng địa bàn thực tế cho bài báo IC4SD-TrafficSnap.
"""
import os
import sys
import pandas as pd
import numpy as np

# Thiết lập UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

routes_path = "zenodo_bundle/metadata/routes.csv"
edges_path = "zenodo_bundle/graph/edges.csv"

routes = pd.read_csv(routes_path)
edges = pd.read_csv(edges_path)

print("=" * 70)
print("1. THỰC NGHIỆM PHÂN BỐ CỰ LY HÀNH LANG KỀ CẬN (|E| = 2,450 CẠNH)")
print("=" * 70)
dists = edges["distance_m"].values
n_edges = len(dists)
mean_dist = np.mean(dists)
std_dist = np.std(dists)
median_dist = np.median(dists)
q25 = np.percentile(dists, 25)
q75 = np.percentile(dists, 75)
p90 = np.percentile(dists, 90)

le_500 = np.sum(dists <= 500)
le_1000 = np.sum(dists <= 1000)
le_1500 = np.sum(dists <= 1500)
le_2000 = np.sum(dists <= 2000)
gt_3000 = np.sum(dists > 3000)

print(f"Tổng số cạnh định hướng hợp lệ: {n_edges:,}")
print(f"Khoảng cách trung bình: {mean_dist:.1f} ± {std_dist:.1f} m")
print(f"Trung vị (Median): {median_dist:.1f} m, IQR: [{q25:.1f}, {q75:.1f}] m, p90: {p90:.1f} m")
print(f"Tối thiểu: {np.min(dists):.1f} m, Tối đa: {np.max(dists):.1f} m")
print(f"Cạnh siêu lân cận (≤ 500 m): {le_500:,} cạnh ({le_500/n_edges*100:.2f}%)")
print(f"Cạnh trong bán kính 1.0 km (≤ 1,000 m): {le_1000:,} cạnh ({le_1000/n_edges*100:.2f}%)")
print(f"Cạnh tương tác Gaussian (≤ 1,500 m): {le_1500:,} cạnh ({le_1500/n_edges*100:.2f}%)")
print(f"Cạnh hành lang trung (≤ 2,000 m): {le_2000:,} cạnh ({le_2000/n_edges*100:.2f}%)")
print(f"Cạnh liên quận dài (> 3,000 m): {gt_3000:,} cạnh ({gt_3000/n_edges*100:.2f}%)")

print("\n" + "=" * 70)
print("2. THỰC NGHIỆM ĐỘ CAO LẮP ĐẶT CAMERA (ELEVATION)")
print("=" * 70)
elev_col = [c for c in routes.columns if "elevation" in c.lower() or "height" in c.lower()]
if elev_col:
    elev = routes[elev_col[0]].dropna().values
    print(f"Cột độ cao: {elev_col[0]}, Số lượng trạm ghi nhận: {len(elev)}")
    print(f"Độ cao trung bình: {np.mean(elev):.2f} ± {np.std(elev):.2f} m")
    print(f"Trung vị: {np.median(elev):.1f} m, IQR: [{np.percentile(elev, 25):.1f}, {np.percentile(elev, 75):.1f}] m")
    print(f"Dải độ cao: [{np.min(elev):.1f}, {np.max(elev):.1f}] m")
else:
    print("Không có cột elevation trong routes.csv, kiểm tra danh sách cột:", list(routes.columns))

print("\n" + "=" * 70)
print("3. THỰC NGHIỆM PHÂN BỐ ĐỊA BÀN HÀNH CHÍNH & MẬT ĐỘ TRẠM")
print("=" * 70)
dist_cols = [c for c in routes.columns if any(k in c.lower() for k in ["district", "quan", "local", "location", "address"])]
print("Các cột địa chỉ / quận:", dist_cols)
for c in dist_cols:
    vc = routes[c].value_counts()
    print(f"\n--- Phân bố theo {c} (Top 10) ---")
    print(vc.head(10))
