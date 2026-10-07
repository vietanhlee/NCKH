import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import math
import networkx as nx

# 1. Routes
routes_path = 'DINO/direction_data_article/zenodo_bundle/metadata/routes.csv'
routes_df = pd.read_csv(routes_path)
print(f"=== 1. ROUTES AUDIT ===")
print(f"Total stations in routes.csv: {len(routes_df)}")
print(f"Columns: {routes_df.columns.tolist()}")
print(f"Latitude range: [{routes_df['latitude'].min():.6f}, {routes_df['latitude'].max():.6f}]")
print(f"Longitude range: [{routes_df['longitude'].min():.6f}, {routes_df['longitude'].max():.6f}]")
if 'camera_elevation_m' in routes_df.columns:
    print(f"Elevation range: [{routes_df['camera_elevation_m'].min()}, {routes_df['camera_elevation_m'].max()}]")

# 2. Road Network Distance & Graph
edges_path = 'DINO/direction_data_article/zenodo_bundle/graph/edges.csv'
edges_df = pd.read_csv(edges_path)
print(f"\n=== 2. EDGES AUDIT ===")
print(f"Total edges in edges.csv: {len(edges_df)}")
print(f"Columns in edges.csv: {edges_df.columns.tolist()}")
print(f"Distance stats (m): mean={edges_df['distance_m'].mean():.2f}, std={edges_df['distance_m'].std():.2f}, min={edges_df['distance_m'].min()}, max={edges_df['distance_m'].max()}, median={edges_df['distance_m'].median():.2f}")
print(f"Distance percentiles (m): p25={np.percentile(edges_df['distance_m'], 25):.2f}, p50={np.percentile(edges_df['distance_m'], 50):.2f}, p75={np.percentile(edges_df['distance_m'], 75):.2f}")

# Direction matrix & Distance matrix
dir_path = 'DINO/direction_data_article/zenodo_bundle/graph/direction.npy'
dist_path = 'DINO/direction_data_article/zenodo_bundle/graph/distance_km.npy'
dir_mat = np.load(dir_path)
dist_mat = np.load(dist_path)
print(f"\n=== 3. MATRIX AUDIT ===")
print(f"Direction matrix shape: {dir_mat.shape}, sum of edges: {dir_mat.sum()}")
print(f"Distance matrix shape: {dist_mat.shape}")

# Pairwise connectivity
# A connected pair (u, v) with u < v
N = dir_mat.shape[0]
connected_pairs = []
bidi_pairs = []
oneway_pairs = []

for i in range(N):
    for j in range(i + 1, N):
        e_ij = dir_mat[i, j]
        e_ji = dir_mat[j, i]
        if e_ij > 0 and e_ji > 0:
            connected_pairs.append((i, j, 'bidi'))
            bidi_pairs.append((i, j))
        elif e_ij > 0 or e_ji > 0:
            connected_pairs.append((i, j, 'oneway'))
            oneway_pairs.append((i, j))

print(f"Total connected node pairs: {len(connected_pairs)}")
print(f"Oneway pairs: {len(oneway_pairs)} ({len(oneway_pairs)/len(connected_pairs)*100:.2f}%)")
print(f"Bidirectional pairs: {len(bidi_pairs)} ({len(bidi_pairs)/len(connected_pairs)*100:.2f}%)")

# Check distance asymmetry on bidirectional pairs
deltas = []
deltas_m_from_km = []
for i, j in bidi_pairs:
    d_ij_km = dist_mat[i, j]
    d_ji_km = dist_mat[j, i]
    diff_m = abs(d_ij_km - d_ji_km) * 1000.0
    deltas.append(diff_m)

deltas = np.array(deltas)
print(f"\nBidirectional distance asymmetry audit:")
print(f"Mean delta: {np.mean(deltas):.2f} m, std: {np.std(deltas):.2f} m, median: {np.median(deltas):.2f} m, max: {np.max(deltas):.2f} m")

# Count for different thresholds
cnt_gt_50 = np.sum(deltas > 50.0)
cnt_ge_50 = np.sum(deltas >= 50.0)
cnt_strict_gt_50_round = np.sum(np.round(deltas, 1) > 50.0)
cnt_strict_ge_50_round = np.sum(np.round(deltas, 1) >= 50.0)
cnt_exact_50 = np.sum(np.isclose(deltas, 50.0, atol=0.1))

print(f"Pairs with delta > 50.0m: {cnt_gt_50} ({cnt_gt_50/len(bidi_pairs)*100:.2f}%)")
print(f"Pairs with delta >= 50.0m: {cnt_ge_50} ({cnt_ge_50/len(bidi_pairs)*100:.2f}%)")
print(f"Pairs with delta exact 50.0m: {cnt_exact_50}")
print(f"Metric symmetric pairs (< 50m): {len(bidi_pairs) - cnt_ge_50} ({(len(bidi_pairs) - cnt_ge_50)/len(bidi_pairs)*100:.2f}%)")
print(f"Metric symmetric pairs (<= 50m): {len(bidi_pairs) - cnt_gt_50} ({(len(bidi_pairs) - cnt_gt_50)/len(bidi_pairs)*100:.2f}%)")

# Why 238 vs 452?
# Let's check: 238 / 690 = 34.4927%, 452 / 690 = 65.5072%
# Where does 238 come from?
# Let's check road_network_distance.csv directly
rnd_path = 'DINO/direction_data_article/zenodo_bundle/metadata/road_network_distance.csv'
rnd_df = pd.read_csv(rnd_path)
print(f"\nroad_network_distance.csv shape: {rnd_df.shape}")
print(f"Columns: {rnd_df.columns.tolist()[:5]}...")

# 4. Degree and Graph components
G = nx.DiGraph()
for i in range(N):
    G.add_node(i)
for i in range(N):
    for j in range(N):
        if dir_mat[i, j] > 0:
            G.add_edge(i, j, weight=dist_mat[i, j])

in_degrees = [d for n, d in G.in_degree()]
out_degrees = [d for n, d in G.out_degree()]
print(f"\n=== 4. GRAPH TOPOLOGY AUDIT ===")
print(f"In-degree: mean={np.mean(in_degrees):.2f}, std={np.std(in_degrees):.2f}, median={np.median(in_degrees)}, max={np.max(in_degrees)}, min={np.min(in_degrees)}")
print(f"Out-degree: mean={np.mean(out_degrees):.2f}, std={np.std(out_degrees):.2f}, median={np.median(out_degrees)}, max={np.max(out_degrees)}, min={np.min(out_degrees)}")

sinks = [n for n in G.nodes() if G.out_degree(n) == 0 and G.in_degree(n) > 0]
sources = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0]
isolated = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
print(f"Sinks ({len(sinks)}): {sinks}")
print(f"Sources ({len(sources)}): {sources}")
print(f"Isolated ({len(isolated)}): {isolated}")

wcc = list(nx.weakly_connected_components(G))
scc = list(nx.strongly_connected_components(G))
print(f"Weakly Connected Components: {len(wcc)}, sizes: {[len(c) for c in sorted(wcc, key=len, reverse=True)]}")
print(f"Strongly Connected Components: {len(scc)}")

# 5. Network Tortuosity
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0 # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

# Let's inspect edges.csv for asymmetry
bidi_edges = edges_df[edges_df['is_bidirectional'] == True]
print(f"\nEdges.csv bidi edges count: {len(bidi_edges)} (each pair has 2 directed edges -> {len(bidi_edges)/2} pairs)")

# Check distance_diff_m in edges.csv
edges_pairs_diff = {}
for idx, row in bidi_edges.iterrows():
    u = int(row['source_station_id'])
    v = int(row['target_station_id'])
    pair = tuple(sorted([u, v]))
    if pair not in edges_pairs_diff:
        edges_pairs_diff[pair] = row['distance_diff_m']

edges_diff_vals = np.array(list(edges_pairs_diff.values()))
print(f"Edges.csv bidi pairs: {len(edges_diff_vals)}")
print(f"Edges.csv diff > 50m: {np.sum(edges_diff_vals > 50.0)}")
print(f"Edges.csv diff >= 50m: {np.sum(edges_diff_vals >= 50.0)}")
print(f"Edges.csv diff < 50m: {np.sum(edges_diff_vals < 50.0)}")
print(f"Edges.csv diff == 50m: {np.sum(np.isclose(edges_diff_vals, 50.0, atol=1e-5))}")

# Check where 232 vs 238 came from!
print(f"Edges.csv diff > 50.0: {np.sum(edges_diff_vals > 50.0)} ({np.sum(edges_diff_vals > 50.0)/690*100:.2f}%)")
print(f"Edges.csv diff >= 50.0: {np.sum(edges_diff_vals >= 50.0)} ({np.sum(edges_diff_vals >= 50.0)/690*100:.2f}%)")

# Now check station indices vs station IDs in routes.csv
tortuosities = []
for idx, row in edges_df.iterrows():
    u_idx = int(row['source_idx'])
    v_idx = int(row['target_idx'])
    d_net = row['distance_m']
    u_row = routes_df.iloc[u_idx]
    v_row = routes_df.iloc[v_idx]
    d_hav = haversine(u_row['latitude'], u_row['longitude'], v_row['latitude'], v_row['longitude'])
    if d_hav >= 100.0:
        tortuosities.append(d_net / d_hav)

tortuosities = np.array(tortuosities)
print(f"\n=== 5. TORTUOSITY AUDIT (N={len(tortuosities)}) ===")
print(f"Mean tau: {np.mean(tortuosities):.4f} +/- {np.std(tortuosities):.4f}")
print(f"Median tau: {np.median(tortuosities):.4f}")
print(f"IQR tau: [{np.percentile(tortuosities, 25):.4f}, {np.percentile(tortuosities, 75):.4f}]")
print(f"p90 tau: {np.percentile(tortuosities, 90):.4f}")

# 6. Nearest Neighbor Euclidean Distance between stations
nn_distances = []
coords = routes_df[['latitude', 'longitude']].values
for i in range(len(coords)):
    dists = []
    for j in range(len(coords)):
        if i != j:
            d = haversine(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            dists.append(d)
    nn_distances.append(min(dists))

nn_distances = np.array(nn_distances)
print(f"\n=== 6. NEAREST NEIGHBOR DISTANCE AUDIT (N={len(nn_distances)}) ===")
print(f"Mean NN: {np.mean(nn_distances):.2f} +/- {np.std(nn_distances):.2f} m")
print(f"Median NN: {np.median(nn_distances):.2f} m")
print(f"IQR NN: [{np.percentile(nn_distances, 25):.2f}, {np.percentile(nn_distances, 75):.2f}] m")
print(f"Min NN: {np.min(nn_distances):.2f} m, Max NN: {np.max(nn_distances):.2f} m")
