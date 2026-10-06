"""
=============================================================================
Module: validate_graph_topology.py
Nghiệp vụ: Technical Validation V4 - Kiểm định tính hợp lệ hình học & phổ đồ thị
           của ma trận khoảng cách mạng lưới đường bộ 608 nút camera tại TP.HCM.
=============================================================================
"""

import os
import sys
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def audit_network_topology(excel_path: str, distance_threshold_km: float = 5.0) -> Dict[str, Any]:
    """
    Phân tích toàn diện đồ thị không gian mạng lưới giao thông 608 nút.
    """
    print(f"[Graph Topology Audit] Loading network matrix from: {excel_path}")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Missing graph file at: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=0, index_col=0)
    node_ids = [int(col) for col in df.columns if str(col).strip().isdigit()]
    N = len(node_ids)

    dist_mat = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    # Loại bỏ self-loops
    np.fill_diagonal(dist_mat, 0.0)

    # Lọc khoảng cách dương hợp lệ
    positive_mask = (dist_mat > 0.0) & (dist_mat <= distance_threshold_km)
    valid_distances = dist_mat[positive_mask]
    num_directed_edges = int(np.sum(positive_mask))

    # Đánh giá chi tiết tính chất có hướng:
    # 1. Cặp liên kết 1 chiều (strictly one-way)
    # 2. Cặp liên kết 2 chiều nhưng lệch khoảng cách (bidirectional distance asymmetry do dải phân cách/đường vòng)
    edge_set = set()
    for i in range(N):
        for j in range(N):
            if i != j and 0.0 < dist_mat[i, j] <= distance_threshold_km:
                edge_set.add((i, j))

    pairs_checked = set()
    strictly_one_way = 0
    bidirectional_pairs = 0
    asym_bidirectional = 0

    for (i, j) in edge_set:
        pair = tuple(sorted([i, j]))
        if pair in pairs_checked:
            continue
        pairs_checked.add(pair)
        has_fwd = (i, j) in edge_set
        has_bwd = (j, i) in edge_set
        if has_fwd and has_bwd:
            bidirectional_pairs += 1
            if abs(dist_mat[i, j] - dist_mat[j, i]) > 0.05:  # > 50m
                asym_bidirectional += 1
        else:
            strictly_one_way += 1

    total_connected_pairs = len(pairs_checked)

    # Trọng số RBF Gaussian
    sigma = float(np.std(valid_distances)) if len(valid_distances) > 0 else 1.0
    if sigma < 1e-4:
        sigma = 1.0

    W = np.zeros((N, N), dtype=np.float64)
    W[positive_mask] = np.exp(-((dist_mat[positive_mask] / sigma) ** 2))
    np.fill_diagonal(W, 0.0)

    # Phân tích bậc (Degree)
    out_degrees = np.sum(W > 0, axis=1)
    in_degrees = np.sum(W > 0, axis=0)

    # 1. Phân tích ma trận chuyển tiếp có hướng (Directed Transition Matrix): P_f = D_out^{-1} * W
    d_out = np.sum(W, axis=1)
    d_out_inv = np.power(d_out, -1.0, where=d_out > 0)
    d_out_inv[d_out <= 0] = 0.0
    P_forward = np.diag(d_out_inv) @ W
    # Với ma trận ngẫu nhiên (row-stochastic), bán kính phổ là 1.0
    spectral_radius_P = float(np.max(np.abs(np.linalg.eigvals(P_forward))))

    # 2. Phân tích Symmetrized Normalized Laplacian L_sym = I - D^{-1/2} W_sym D^{-1/2}
    W_sym = (W + W.T) / 2.0
    deg_sym = np.sum(W_sym, axis=1)
    d_inv_sqrt = np.power(deg_sym, -0.5, where=deg_sym > 0)
    d_inv_sqrt[deg_sym <= 0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)

    L_sym = np.eye(N) - (D_inv_sqrt @ W_sym @ D_inv_sqrt)
    eigenvalues = np.linalg.eigvalsh(L_sym)

    lambda_min = float(eigenvalues[0])
    lambda_max = float(eigenvalues[-1])
    spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0

    report = {
        "num_nodes": N,
        "num_directed_edges": num_directed_edges,
        "density_pct": float(num_directed_edges / (N * (N - 1)) * 100.0),
        "total_connected_pairs": total_connected_pairs,
        "strictly_one_way_pairs": strictly_one_way,
        "strictly_one_way_pct": float(strictly_one_way / total_connected_pairs * 100.0),
        "bidirectional_pairs": bidirectional_pairs,
        "bidirectional_pct": float(bidirectional_pairs / total_connected_pairs * 100.0),
        "asym_bidirectional_pairs": asym_bidirectional,
        "asym_bidirectional_pct": float(asym_bidirectional / bidirectional_pairs * 100.0) if bidirectional_pairs > 0 else 0.0,
        "mean_distance_km": float(np.mean(valid_distances)) if len(valid_distances) > 0 else 0.0,
        "median_distance_km": float(np.median(valid_distances)) if len(valid_distances) > 0 else 0.0,
        "std_distance_km": float(np.std(valid_distances)) if len(valid_distances) > 0 else 0.0,
        "mean_out_degree": float(np.mean(out_degrees)),
        "max_out_degree": int(np.max(out_degrees)),
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "spectral_gap": spectral_gap,
    }

    print("\n--- Graph Topology Audit Summary ---")
    print(f"Network Nodes:                {N}")
    print(f"Directed Edges (<{distance_threshold_km}km):        {num_directed_edges} (Sparsity: {100.0 - report['density_pct']:.2f}%)")
    print(f"Total Connected Pairs:        {total_connected_pairs}")
    print(f"  - Strictly One-Way Pairs:   {strictly_one_way} ({report['strictly_one_way_pct']:.2f}%)")
    print(f"  - Bidirectional Pairs:      {bidirectional_pairs} ({report['bidirectional_pct']:.2f}%)")
    print(f"  - Asymmetric Distance Pairs:{asym_bidirectional} / {bidirectional_pairs} ({report['asym_bidirectional_pct']:.2f}% of 2-way)")
    print(f"Routing Distance (km):        {report['mean_distance_km']:.2f} +/- {report['std_distance_km']:.2f} (Median: {report['median_distance_km']:.2f})")
    print(f"Node Out-Degree:              Mean={report['mean_out_degree']:.2f}, Max={report['max_out_degree']}")
    print(f"Normalized Laplacian:         lambda_min={lambda_min:.4f}, lambda_max={lambda_max:.4f}, gap={spectral_gap:.4f}")
    return report


if __name__ == "__main__":
    matrix_file = sys.argv[1] if len(sys.argv) > 1 else "g:/nckh/DINO/direction_data_article/zenodo_bundle/metadata/road_network_distance.xlsx"
    audit_network_topology(matrix_file)
