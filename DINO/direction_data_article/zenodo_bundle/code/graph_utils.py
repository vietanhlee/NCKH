"""
=============================================================================
Module: graph_utils.py
Mục đích: Cung cấp các tiện ích xử lý đồ thị không gian mạng lưới đường bộ
          (Spatial Road Network Graph) cho IC4SD-Traffic-HCM:
          - Tính ma trận kề trọng số từ khoảng cách routing thực tế OpenStreetMap
          - Tính Normalized Laplacian và Scaled Chebyshev Laplacian (STGCN, ASTGCN)
          - Tính Transition Matrix (DCRNN, Graph WaveNet)
=============================================================================
"""

import os
from typing import Tuple, List, Optional, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_road_graph(
    graph_source_path: str,
    sigma_scale: float = 1.0,
    distance_threshold_km: float = 6.0,
    sheet_name: Union[str, int] = 0,
) -> Tuple[np.ndarray, List[int]]:
    """
    Đọc ma trận khoảng cách mạng lưới từ tệp NPY, CSV hoặc Excel và tính toán ma trận kề trọng số Gauss.

    Công thức trọng số RBF Gaussian kernel:
        W_ij = exp(- (d_ij / sigma)^2) nếu 0 < d_ij <= threshold và i != j
        W_ij = 0 nếu ngược lại

    Args:
        graph_source_path (str): Đường dẫn đến distance_m.npy, road_network_distance.csv hoặc .xlsx.
        sigma_scale (float): Hệ số tỷ lệ độ lệch chuẩn sigma.
        distance_threshold_km (float): Ngưỡng khoảng cách tối đa để thiết lập cạnh đồ thị (km, mặc định: 6.0).
        sheet_name: Tên hoặc chỉ số sheet nếu dùng tệp Excel.

    Returns:
        Tuple: (weight_matrix [N, N], node_ids_list)
    """
    if not os.path.exists(graph_source_path):
        raise FileNotFoundError(f"Không tìm thấy tệp ma trận khoảng cách tại: {graph_source_path}")

    ext = os.path.splitext(graph_source_path)[1].lower()
    if ext == ".npy":
        raw_mat = np.load(graph_source_path)
        # Nếu đơn vị là mét (giá trị trung bình > 50), đổi sang km
        if np.nanmean(raw_mat[raw_mat > 0]) > 50.0:
            dist_mat = raw_mat / 1000.0
        else:
            dist_mat = raw_mat.copy()
        node_ids = list(range(1, dist_mat.shape[0] + 1))
    elif ext == ".csv":
        df = pd.read_csv(graph_source_path, index_col=0)
        node_ids = [int(col) for col in df.columns if str(col).strip().isdigit()]
        dist_mat = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        # Đọc dữ liệu từ Excel
        df = pd.read_excel(graph_source_path, sheet_name=sheet_name, index_col=0)
        node_ids = [int(col) for col in df.columns if str(col).strip().isdigit()]
        dist_mat = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    N = dist_mat.shape[0]

    # Tính sigma từ các khoảng cách dương hợp lệ
    valid_distances = dist_mat[(dist_mat > 0.0) & (dist_mat <= distance_threshold_km)]
    if len(valid_distances) > 0:
        sigma = np.std(valid_distances) * sigma_scale
        if sigma < 1e-5:
            sigma = np.mean(valid_distances) * sigma_scale
    else:
        sigma = 1.0

    # Khởi tạo ma trận trọng số W
    weights = np.zeros((N, N), dtype=np.float32)
    valid_mask = (dist_mat > 0.0) & (dist_mat <= distance_threshold_km)

    weights[valid_mask] = np.exp(-((dist_mat[valid_mask] / (sigma + 1e-9)) ** 2))

    # Đảm bảo đường chéo bằng 0 (không có self-loop trong ma trận khoảng cách)
    np.fill_diagonal(weights, 0.0)

    return weights, node_ids


def calculate_normalized_laplacian(adj: np.ndarray, symmetrize: bool = True) -> np.ndarray:
    """
    Tính Normalized Graph Laplacian cho đồ thị:
        L = I - D^{-1/2} * W * D^{-1/2}
    
    Lưu ý học thuật:
        Đồ thị đường bộ thực tế mang tính có hướng (asymmetric). Đối với các mô hình phổ
        (spectral methods như STGCN / ChebNet), ma trận W cần được đối xứng hóa 
        (W_sym = 0.5 * (W + W^T)). Đối với các mô hình khuếch tán có hướng (DCRNN / Graph WaveNet),
        khuyến nghị sử dụng calculate_dual_directed_transition_matrices() để bảo toàn tính một chiều.

    Args:
        adj (np.ndarray): Ma trận kề trọng số [N, N].
        symmetrize (bool): Tự động đối xứng hóa ma trận kề nếu True (mặc định: True).

    Returns:
        np.ndarray: Ma trận Laplacian chuẩn hóa [N, N].
    """
    W = (adj + adj.T) / 2.0 if symmetrize else adj.copy()
    N = W.shape[0]
    degree = np.sum(W, axis=1)
    d_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    d_inv_sqrt[degree <= 0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # L = I - D^{-1/2} * W * D^{-1/2}
    L_norm = np.eye(N, dtype=np.float32) - (D_inv_sqrt @ W @ D_inv_sqrt).astype(np.float32)
    return L_norm


def compute_chebyshev_laplacian(adj: np.ndarray, lambda_max: Optional[float] = None) -> np.ndarray:
    """
    Tính Scaled Chebyshev Graph Laplacian cho xấp xỉ đa thức Chebyshev (ChebNet / STGCN):
        L_tilde = (2 / lambda_max) * L - I

    Args:
        adj (np.ndarray): Ma trận kề trọng số [N, N].
        lambda_max (float, optional): Trị riêng lớn nhất của ma trận Laplacian L. Nếu None, sẽ tự tính.

    Returns:
        np.ndarray: Scaled Laplacian [N, N] có phổ giá trị nằm trong đoạn [-1, 1].
    """
    L_norm = calculate_normalized_laplacian(adj, symmetrize=True)
    N = adj.shape[0]

    if lambda_max is None:
        try:
            eigenvalues = np.linalg.eigvalsh(L_norm)
            lambda_max = float(eigenvalues[-1])
        except Exception:
            lambda_max = 2.0

    if lambda_max < 1e-4:
        lambda_max = 2.0

    L_tilde = (2.0 / lambda_max) * L_norm - np.eye(N, dtype=np.float32)
    return L_tilde.astype(np.float32)


def calculate_random_walk_matrix(adj: np.ndarray) -> np.ndarray:
    """
    Tính ma trận chuyển tiếp Random Walk cho DCRNN / Graph WaveNet:
        P = D_out^{-1} * W
    """
    degree = np.sum(adj, axis=1)
    d_inv = np.power(degree, -1.0, where=degree > 0)
    d_inv[degree <= 0] = 0.0
    D_inv = np.diag(d_inv)
    return (D_inv @ adj).astype(np.float32)


def calculate_dual_directed_transition_matrices(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính cặp ma trận chuyển tiếp có hướng kép (Dual-Transition Directed Walk) theo DCRNN (Li et al., ICLR 2018):
        - Forward Transition  (Dòng xuôi theo hướng giao thông): P_f = D_out^{-1} * W
        - Backward Transition (Dòng ngược cảm nhận ngược dòng):  P_b = D_in^{-1} * W^T

    Args:
        adj (np.ndarray): Ma trận kề có hướng thực tế [N, N].

    Returns:
        Tuple[np.ndarray, np.ndarray]: (P_forward, P_backward), mỗi ma trận có kích thước [N, N].
    """
    # 1. Forward transition: P_f = D_out^{-1} * W
    d_out = np.sum(adj, axis=1)
    d_out_inv = np.power(d_out, -1.0, where=d_out > 0)
    d_out_inv[d_out <= 0] = 0.0
    P_forward = np.diag(d_out_inv) @ adj

    # 2. Backward transition: P_b = D_in^{-1} * W^T
    adj_T = adj.T
    d_in = np.sum(adj_T, axis=1)
    d_in_inv = np.power(d_in, -1.0, where=d_in > 0)
    d_in_inv[d_in <= 0] = 0.0
    P_backward = np.diag(d_in_inv) @ adj_T

    return P_forward.astype(np.float32), P_backward.astype(np.float32)
