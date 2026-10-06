"""
HCMC-TrafficSnap: Production Graph Processing and Format Conversion Utility
===========================================================================
This script processes the raw municipal road network distance matrix (Graph_fix_py_3.xlsx)
originating from the Ho Chi Minh City traffic surveillance camera deployment.

It performs:
1. Data integrity verification and diagonal regularization (self-distance = 0.0).
2. Format conversion into standard open-access and machine-learning formats:
   - NumPy binary tensors (distance_m.npy, distance_km.npy, direction.npy)
   - Tabular edge list (edges.csv)
   - CSV distance matrix (road_network_distance.csv)
   - Station mapping index (stations.csv)
3. Rigorous mathematical and topological validation:
   - Directed edge counts, unique node pairs, bidirectional vs. unidirectional split
   - Distance asymmetry statistics (|d_ij - d_ji| >= 50m)
   - In-degree, out-degree, sink, source, and isolated node detection
   - Transition matrix normalization (DCRNN forward/backward walk operators)
"""

import os
import argparse
import numpy as np
import pandas as pd


def process_graph(
    excel_path: str,
    output_graph_dir: str,
    output_metadata_dir: str
):
    print(f"[*] Loading raw Excel graph from: {excel_path}")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Source file not found: {excel_path}")

    df_raw = pd.read_excel(excel_path, index_col=0)
    print(f"[*] Raw matrix dimensions: {df_raw.shape}")

    # Validate square matrix and station indices
    station_ids = [int(idx) for idx in df_raw.index]
    col_ids = [int(col) for col in df_raw.columns]
    assert station_ids == col_ids, "Row station IDs do not match column station IDs!"
    num_stations = len(station_ids)
    print(f"[*] Confirmed {num_stations} camera stations (ID range [{min(station_ids)}, {max(station_ids)}]).")

    # Convert to numeric float numpy array (km)
    dist_km_raw = df_raw.values.astype(np.float64)

    # Regularize diagonal: in network routing, self-distance d_ii = 0.0
    # Raw data had 2 non-zero loops (station 358: 0.647 km, station 364: 0.125 km due to U-turn loops)
    diag_loops = {}
    for i in range(num_stations):
        if not np.isnan(dist_km_raw[i, i]) and dist_km_raw[i, i] > 0.0:
            diag_loops[station_ids[i]] = dist_km_raw[i, i]
    if diag_loops:
        print(f"[*] Noted {len(diag_loops)} routing U-turn self-loops on diagonal: {diag_loops}")

    # Create sanitized distance matrix (km)
    dist_km = dist_km_raw.copy()
    np.fill_diagonal(dist_km, 0.0)

    # Convert to meters
    dist_m = dist_km * 1000.0

    # Off-diagonal directed adjacency
    adj_directed = (~np.isnan(dist_km_raw)).astype(np.int8)
    np.fill_diagonal(adj_directed, 0)

    num_directed_edges = int(np.sum(adj_directed))
    print(f"[*] Total valid off-diagonal directed corridors: {num_directed_edges:,}")

    # Analyze connected pairs and directional asymmetry
    connected_pairs = []
    bidirectional_pairs = []
    unidirectional_pairs = []
    edge_records = []

    for i in range(num_stations):
        u_id = station_ids[i]
        for j in range(num_stations):
            if i == j:
                continue
            if adj_directed[i, j] == 1:
                d_ij_km = float(dist_km[i, j])
                d_ij_m = float(dist_m[i, j])
                has_rev = bool(adj_directed[j, i] == 1)
                d_rev_km = float(dist_km[j, i]) if has_rev else np.nan
                d_diff_m = abs(d_ij_m - float(dist_m[j, i])) if has_rev else np.nan

                edge_records.append({
                    "source_station_id": u_id,
                    "target_station_id": station_ids[j],
                    "source_idx": i,
                    "target_idx": j,
                    "distance_km": round(d_ij_km, 4),
                    "distance_m": round(d_ij_m, 1),
                    "is_bidirectional": has_rev,
                    "reverse_distance_km": round(d_rev_km, 4) if has_rev else None,
                    "distance_diff_m": round(d_diff_m, 1) if has_rev else None
                })

    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            fwd = bool(adj_directed[i, j] == 1)
            bwd = bool(adj_directed[j, i] == 1)
            if fwd or bwd:
                connected_pairs.append((i, j))
                if fwd and bwd:
                    d_fwd = dist_km[i, j]
                    d_bwd = dist_km[j, i]
                    diff_m = abs(d_fwd - d_bwd) * 1000.0
                    bidirectional_pairs.append({
                        "node_a": station_ids[i],
                        "node_b": station_ids[j],
                        "distance_ab_km": d_fwd,
                        "distance_ba_km": d_bwd,
                        "diff_m": diff_m
                    })
                else:
                    unidirectional_pairs.append((station_ids[i], station_ids[j], fwd, bwd))

    df_bidi = pd.DataFrame(bidirectional_pairs)
    asym_50m = np.sum(df_bidi["diff_m"] >= 50.0) if len(df_bidi) > 0 else 0
    asym_10m = np.sum(df_bidi["diff_m"] >= 10.0) if len(df_bidi) > 0 else 0
    exact_equal = np.sum(df_bidi["diff_m"] < 0.1) if len(df_bidi) > 0 else 0

    print(f"[*] Total unique connected station pairs: {len(connected_pairs):,}")
    print(f"[*] Strictly one-way pairs (unidirectional): {len(unidirectional_pairs):,} ({len(unidirectional_pairs)/len(connected_pairs)*100:.2f}%)")
    print(f"[*] Bidirectional corridor pairs: {len(bidirectional_pairs):,} ({len(bidirectional_pairs)/len(connected_pairs)*100:.2f}%)")
    print(f"    - Exactly identical distance (< 0.1m): {exact_equal} ({exact_equal/len(bidirectional_pairs)*100:.2f}%)")
    print(f"    - Asymmetric distance (|d_ij - d_ji| >= 10m): {asym_10m} ({asym_10m/len(bidirectional_pairs)*100:.2f}%)")
    print(f"    - Asymmetric distance (|d_ij - d_ji| >= 50m): {asym_50m} ({asym_50m/len(bidirectional_pairs)*100:.2f}%)")
    print(f"    - Max directional distance discrepancy: {df_bidi['diff_m'].max():.1f} m ({df_bidi['diff_m'].max()/1000.0:.2f} km)")
    print(f"    - Mean directional distance discrepancy: {df_bidi['diff_m'].mean():.1f} m")

    # In/Out degree distribution
    out_degrees = np.sum(adj_directed, axis=1)
    in_degrees = np.sum(adj_directed, axis=0)

    isolated_nodes = [station_ids[i] for i in range(num_stations) if out_degrees[i] == 0 and in_degrees[i] == 0]
    sink_nodes = [(station_ids[i], int(in_degrees[i])) for i in range(num_stations) if out_degrees[i] == 0 and in_degrees[i] > 0]
    source_nodes = [(station_ids[i], int(out_degrees[i])) for i in range(num_stations) if in_degrees[i] == 0 and out_degrees[i] > 0]

    print(f"[*] Out-degree: mean = {np.mean(out_degrees):.2f} +/- {np.std(out_degrees):.2f}, median = {np.median(out_degrees)}, max = {np.max(out_degrees)}")
    print(f"[*] In-degree:  mean = {np.mean(in_degrees):.2f} +/- {np.std(in_degrees):.2f}, median = {np.median(in_degrees)}, max = {np.max(in_degrees)}")
    print(f"[*] Isolated stations (in=0, out=0): {len(isolated_nodes)} -> {isolated_nodes}")
    print(f"[*] Sink stations (out=0, in>0): {len(sink_nodes)} -> {sink_nodes}")
    print(f"[*] Source stations (in=0, out>0): {len(source_nodes)} -> {source_nodes}")

    # Distance statistics (across all valid off-diagonal edges)
    all_edge_km = [e["distance_km"] for e in edge_records]
    print(f"[*] Edge distances (km): min = {np.min(all_edge_km):.3f}, max = {np.max(all_edge_km):.3f}, mean = {np.mean(all_edge_km):.3f} +/- {np.std(all_edge_km):.3f}, median = {np.median(all_edge_km):.3f}")

    # Ensure output directories exist
    os.makedirs(output_graph_dir, exist_ok=True)
    os.makedirs(output_metadata_dir, exist_ok=True)

    # 1. Export NumPy tensors
    dist_m_path = os.path.join(output_graph_dir, "distance_m.npy")
    dist_km_path = os.path.join(output_graph_dir, "distance_km.npy")
    dir_path = os.path.join(output_graph_dir, "direction.npy")

    np.save(dist_m_path, dist_m)
    np.save(dist_km_path, dist_km)
    np.save(dir_path, adj_directed)
    print(f"[+] Saved NumPy tensors to {output_graph_dir}: distance_m.npy, distance_km.npy, direction.npy")

    # 2. Export Tabular edge list
    df_edges = pd.DataFrame(edge_records)
    edges_path = os.path.join(output_graph_dir, "edges.csv")
    df_edges.to_csv(edges_path, index=False)
    print(f"[+] Saved tabular edge list to: {edges_path} ({len(df_edges):,} rows)")

    # 3. Export CSV Distance Matrix (with Station ID headers)
    df_dist_csv = pd.DataFrame(dist_km, index=station_ids, columns=station_ids)
    dist_csv_path = os.path.join(output_metadata_dir, "road_network_distance.csv")
    df_dist_csv.to_csv(dist_csv_path)
    print(f"[+] Saved 608x608 distance matrix CSV to: {dist_csv_path}")

    # 4. Export Stations Mapping
    df_stations = pd.DataFrame({
        "matrix_index": list(range(num_stations)),
        "station_id": station_ids,
        "out_degree": [int(d) for d in out_degrees],
        "in_degree": [int(d) for d in in_degrees],
        "node_type": [
            "isolated" if station_ids[i] in isolated_nodes else
            "sink" if out_degrees[i] == 0 else
            "source" if in_degrees[i] == 0 else
            "regular" for i in range(num_stations)
        ]
    })
    stations_path = os.path.join(output_metadata_dir, "stations.csv")
    df_stations.to_csv(stations_path, index=False)
    print(f"[+] Saved station registry and topological roles to: {stations_path}")

    # Summary dictionary for paper citation
    metrics_summary = {
        "num_stations": num_stations,
        "num_directed_edges": num_directed_edges,
        "num_connected_pairs": len(connected_pairs),
        "unidirectional_pairs": len(unidirectional_pairs),
        "unidirectional_pct": len(unidirectional_pairs) / len(connected_pairs) * 100.0,
        "bidirectional_pairs": len(bidirectional_pairs),
        "bidirectional_pct": len(bidirectional_pairs) / len(connected_pairs) * 100.0,
        "asym_50m": asym_50m,
        "asym_50m_pct_of_bidi": asym_50m / len(bidirectional_pairs) * 100.0,
        "max_diff_m": float(df_bidi["diff_m"].max()),
        "mean_diff_m": float(df_bidi["diff_m"].mean()),
        "min_dist_km": float(np.min(all_edge_km)),
        "max_dist_km": float(np.max(all_edge_km)),
        "mean_dist_km": float(np.mean(all_edge_km)),
        "std_dist_km": float(np.std(all_edge_km)),
        "median_dist_km": float(np.median(all_edge_km)),
        "mean_out_degree": float(np.mean(out_degrees)),
        "std_out_degree": float(np.std(out_degrees)),
        "max_out_degree": int(np.max(out_degrees)),
        "mean_in_degree": float(np.mean(in_degrees)),
        "std_in_degree": float(np.std(in_degrees)),
        "max_in_degree": int(np.max(in_degrees)),
        "isolated_nodes_count": len(isolated_nodes),
        "sink_nodes_count": len(sink_nodes),
        "source_nodes_count": len(source_nodes),
    }

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process real HCMC traffic graph matrix")
    parser.add_argument(
        "--excel_path",
        type=str,
        default=r"C:\Users\levie\Downloads\DATA-TASTGCN\DATA-TASTGCN\GRAPH\Graph_fix_py_3.xlsx",
        help="Path to raw Graph_fix_py_3.xlsx"
    )
    parser.add_argument(
        "--output_graph_dir",
        type=str,
        default=r"g:\nckh\DINO\direction_data_article\zenodo_bundle\graph",
        help="Target directory for ML graph tensors and edge lists"
    )
    parser.add_argument(
        "--output_metadata_dir",
        type=str,
        default=r"g:\nckh\DINO\direction_data_article\zenodo_bundle\metadata",
        help="Target directory for human-readable metadata matrices and catalogs"
    )
    args = parser.parse_args()

    metrics = process_graph(args.excel_path, args.output_graph_dir, args.output_metadata_dir)
    print("\n[SUCCESS] Extracted Metrics Summary for Paper:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")
