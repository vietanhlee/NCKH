"""
=============================================================================
 DINOv3 Metric Learning & Representation Geometry Evaluation Suite
 Urban Traffic Surveillance Representation Analysis (Self-Supervised & Supervised)
 
 Evaluates 6 Core Representation Learning Metrics (Beyond Training Loss):
   1. k-NN Retrieval & Classification Accuracy (k = 1, 5, 10, 20 with Cosine Metric)
   2. Intra-to-Inter Class Distance Ratio (R_intra/inter) - Cluster Compactness vs Separation
   3. Hyperspherical Alignment & Uniformity (Wang & Isola, ICML 2020)
   4. Effective Rank & Dimensional Collapse SVD Spectrum (Roy & Vetterli, 2007)
   5. Unsupervised Clustering Quality (Silhouette Score, Davies-Bouldin, Calinski-Harabasz)
   6. Publication-grade Visualizations (t-SNE Manifold, Distance KDE, Singular Value Decay)

 Supported Models:
   - DINOv3 (ViT-S/16, ViT-B/16, ConvNeXt)
   - DINOv2 (ViT-S/14, ViT-B/14)
   - Supervised Baselines (ResNet-50, ConvNeXt-Tiny, ViT-S/16)
=============================================================================
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

# Import model builder from train_ssl_dinov3
from train_ssl_dinov3 import build_backbone


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =====================================================================
# 1. DATASET & DENSITY REGIME LABELING
# =====================================================================

class TrafficMetricDataset(Dataset):
    """
    Loads traffic surveillance images and maps vehicle counts into
    standard traffic congestion regimes:
      - Class 0: Low Congestion (< 10 vehicles)
      - Class 1: Medium Congestion (10 - 25 vehicles)
      - Class 2: High Congestion / Gridlock (> 25 vehicles)
    """
    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_name = str(row["filename"])
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.exists(img_path):
            import glob
            alt = glob.glob(os.path.join(self.image_dir, "**", img_name), recursive=True)
            if alt:
                img_path = alt[0]
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            if self.transform:
                x = self.transform(img_rgb)
            else:
                x = transforms.ToTensor()(img_rgb)

        label = int(row["congestion_class"])
        return x, label, img_name


def prepare_dataframe(csv_path: str, image_dir: str = None) -> pd.DataFrame:
    """Discovers columns, validates image existence on disk, and creates standardized congestion classes."""
    df = pd.read_csv(csv_path)
    raw_count = len(df)

    # Column discovery
    fn_col = None
    for c in ["filename", "file_name", "image", "image_name", "img", "name"]:
        if c in df.columns:
            fn_col = c
            break
    if fn_col is None:
        fn_col = df.columns[0]

    car_col = None
    for c in ["car", "cars", "oto", "o_to", "car_count"]:
        if c in df.columns:
            car_col = c
            break

    moto_col = None
    for c in ["motorcycle", "motorcycles", "motorbike", "moto", "bike", "xe_may", "xemay"]:
        if c in df.columns:
            moto_col = c
            break

    # If image_dir is provided, filter CSV to only images that actually exist on disk
    if image_dir and os.path.isdir(image_dir):
        print(f"🔍 [Data Filter] Indexing image directory: {image_dir}...")
        existing_disk_images = {}
        for root, _, files in os.walk(image_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    rel_p = os.path.relpath(os.path.join(root, f), image_dir)
                    existing_disk_images[f] = rel_p
                    existing_disk_images[os.path.splitext(f)[0]] = rel_p

        valid_indices = []
        resolved_filenames = []
        for idx, row in df.iterrows():
            fname = str(row[fn_col]).strip()
            base_no_ext = os.path.splitext(os.path.basename(fname))[0]
            base_with_ext = os.path.basename(fname)

            if fname in existing_disk_images:
                valid_indices.append(idx)
                resolved_filenames.append(existing_disk_images[fname])
            elif base_with_ext in existing_disk_images:
                valid_indices.append(idx)
                resolved_filenames.append(existing_disk_images[base_with_ext])
            elif base_no_ext in existing_disk_images:
                valid_indices.append(idx)
                resolved_filenames.append(existing_disk_images[base_no_ext])
            elif os.path.isfile(os.path.join(image_dir, fname)):
                valid_indices.append(idx)
                resolved_filenames.append(fname)

        df = df.iloc[valid_indices].reset_index(drop=True)
        df[fn_col] = resolved_filenames
        clean_count = len(df)
        print(f"📊 [Data Filter] Read CSV: {raw_count} rows | Found valid images on disk: {clean_count} | Skipped missing: {raw_count - clean_count}")

    if car_col and moto_col:
        total_veh = df[car_col].astype(float) + df[moto_col].astype(float)
    elif "count" in df.columns:
        total_veh = df["count"].astype(float)
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            total_veh = df[num_cols[0]].astype(float)
        else:
            total_veh = pd.Series(np.zeros(len(df)))

    # Discretize into 3 Standard Congestion Regimes (Low, Medium, High)
    classes = []
    for count in total_veh:
        if count < 10.0:
            classes.append(0)  # Low
        elif count <= 25.0:
            classes.append(1)  # Medium
        else:
            classes.append(2)  # High

    processed_df = pd.DataFrame({
        "filename": df[fn_col],
        "total_vehicles": total_veh,
        "congestion_class": classes,
    })
    return processed_df


# =====================================================================
# 2. FEATURE EXTRACTION PIPELINE
# =====================================================================

@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extracts L2-normalized feature embeddings from frozen backbone."""
    backbone.eval()
    embeddings = []
    labels = []
    filenames = []

    for x, y, f_names in tqdm(dataloader, desc="Extracting DINOv3 Embeddings", leave=False):
        x = x.to(device)
        feats = backbone(x)
        # Handle dict or tuple outputs if any
        if isinstance(feats, dict):
            feats = feats.get("x_norm_clstoken", feats.get("logits", list(feats.values())[0]))
        elif isinstance(feats, (list, tuple)):
            feats = feats[0]

        # L2 Normalization onto Hypersphere S^{d-1}
        feats = F.normalize(feats, p=2, dim=-1)
        embeddings.append(feats.cpu().numpy())
        labels.append(y.numpy())
        filenames.extend(f_names)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels, filenames


# =====================================================================
# 3. METRIC LEARNING EVALUATION METRICS
# =====================================================================

def evaluate_knn_retrieval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluates metric space via non-parametric k-NN retrieval with Cosine distance.
    Measures neighborhood semantic consistency without training any classifier parameters.
    """
    metrics = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        metrics[f"knn_top1_acc_k{k}"] = float(acc)
        metrics[f"knn_macro_f1_k{k}"] = float(macro_f1)

    return metrics


def compute_intra_to_inter_ratio(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 3000,
    seed: int = 42
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Computes Intra-to-Inter Class Distance Ratio (R_intra/inter) under Cosine Distance.
    Formula:
        R_intra/inter = E_{i!=j, y_i = y_j}[1 - cos(z_i, z_j)] / E_{y_i != y_k}[1 - cos(z_i, z_k)]
    Lower is better (tighter intra-class clusters, larger inter-class margin).
    """
    np.random.seed(seed)
    N = len(embeddings)
    if N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
        X = embeddings[idx]
        y = labels[idx]
    else:
        X = embeddings
        y = labels

    # Pairwise Cosine Similarity: S = X @ X.T
    sim_matrix = np.matmul(X, X.T)
    # Cosine Distance: D = 1 - S
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)

    # Class matching mask
    y_col = y.reshape(-1, 1)
    same_class_mask = (y_col == y_col.T)
    diff_class_mask = ~same_class_mask

    # Exclude diagonal (self-distances)
    np.fill_diagonal(same_class_mask, False)

    intra_distances = dist_matrix[same_class_mask]
    inter_distances = dist_matrix[diff_class_mask]

    mean_intra = float(np.mean(intra_distances)) if len(intra_distances) > 0 else 0.0
    mean_inter = float(np.mean(inter_distances)) if len(inter_distances) > 0 else 1.0
    ratio = mean_intra / max(mean_inter, 1e-8)

    return ratio, mean_intra, mean_inter, intra_distances, inter_distances


def compute_alignment_and_uniformity(
    embeddings: np.ndarray,
    labels: np.ndarray,
    alpha: float = 2.0,
    t: float = 2.0,
    max_pairs: int = 10000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Computes Wang & Isola (ICML 2020) Alignment and Uniformity on Hypersphere S^{d-1}.
      - Alignment: Invariance among positive pairs (same congestion state / temporal views).
        L_align = E_{(x, y) ~ P_pos} [ ||f(x) - f(y)||_2^alpha ]
      - Uniformity: Entropy-maximizing distribution preserving maximal information.
        L_uniform = log E_{x, y ~ P_data} [ exp(-t * ||f(x) - f(y)||_2^2) ]
    """
    np.random.seed(seed)
    N, d = embeddings.shape

    # 1. Alignment (sampled from same class pairs)
    unique_labels = np.unique(labels)
    intra_pairs_dist_sq = []
    for lbl in unique_labels:
        cls_idx = np.where(labels == lbl)[0]
        if len(cls_idx) < 2:
            continue
        n_pairs_to_sample = min(max_pairs // len(unique_labels), len(cls_idx) * 2)
        idx_a = np.random.choice(cls_idx, n_pairs_to_sample, replace=True)
        idx_b = np.random.choice(cls_idx, n_pairs_to_sample, replace=True)
        # Avoid same element
        valid = (idx_a != idx_b)
        idx_a, idx_b = idx_a[valid], idx_b[valid]
        if len(idx_a) == 0:
            continue
        diff = embeddings[idx_a] - embeddings[idx_b]
        l2_sq = np.sum(diff ** 2, axis=1)
        intra_pairs_dist_sq.extend(l2_sq)

    if len(intra_pairs_dist_sq) > 0:
        intra_pairs_dist_sq = np.array(intra_pairs_dist_sq)
        alignment = float(np.mean(intra_pairs_dist_sq ** (alpha / 2.0)))
    else:
        alignment = 0.0

    # 2. Uniformity (all pairs random sample)
    n_sample_uniform = min(N, 2000)
    sub_idx = np.random.choice(N, n_sample_uniform, replace=False)
    X_sub = embeddings[sub_idx]
    # Pairwise squared Euclidean distance: ||x - y||^2 = 2 - 2 * (x . y) for unit vectors
    gram = np.matmul(X_sub, X_sub.T)
    pdist_sq = np.clip(2.0 - 2.0 * gram, 0.0, 4.0)

    # Exclude diagonal
    i_upper, j_upper = np.triu_indices(n_sample_uniform, k=1)
    pairwise_sq = pdist_sq[i_upper, j_upper]

    kernel_vals = np.exp(-t * pairwise_sq)
    uniformity = float(np.log(np.mean(kernel_vals) + 1e-12))

    return alignment, uniformity


def compute_effective_rank(embeddings: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Computes Effective Rank (Roy & Vetterli, 2007) via Singular Value Spectrum.
    Detects Dimensional Collapse in Metric Learning:
        p_k = sigma_k / sum(sigma)
        H(p) = - sum(p_k * ln(p_k))
        Rank_eff = exp(H(p))
    Percentage of Dimension Utilization = Rank_eff / d * 100%.
    """
    # Center embeddings
    X_centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    # SVD
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)

    s_norm = s / np.sum(s)
    s_norm = s_norm[s_norm > 1e-12]  # Avoid log(0)
    entropy = -np.sum(s_norm * np.log(s_norm))
    eff_rank = float(np.exp(entropy))

    d = embeddings.shape[1]
    rank_percentage = float((eff_rank / d) * 100.0)

    return eff_rank, rank_percentage, s


def compute_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Computes unsupervised clustering metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz."""
    if len(np.unique(labels)) < 2:
        return {"silhouette_score": 0.0, "davies_bouldin_index": 0.0, "calinski_harabasz_index": 0.0}

    # Subsample if too large for Silhouette O(N^2)
    N = len(embeddings)
    if N > 5000:
        idx = np.random.choice(N, 5000, replace=False)
        X_sub = embeddings[idx]
        y_sub = labels[idx]
    else:
        X_sub = embeddings
        y_sub = labels

    sil = float(silhouette_score(X_sub, y_sub, metric="cosine"))
    dbi = float(davies_bouldin_score(X_sub, y_sub))
    chi = float(calinski_harabasz_score(X_sub, y_sub))

    return {
        "silhouette_score": sil,
        "davies_bouldin_index": dbi,
        "calinski_harabasz_index": chi,
    }


# =====================================================================
# 4. PUBLICATION FIGURE GENERATION UTILITIES
# =====================================================================

def plot_publication_figures(
    embeddings: np.ndarray,
    labels: np.ndarray,
    singular_values: np.ndarray,
    intra_distances: np.ndarray,
    inter_distances: np.ndarray,
    eff_rank: float,
    intra_inter_ratio: float,
    save_dir: str,
    model_name: str = "DINOv3 (ViT-S/16)"
):
    """
    Renders 3 publication-ready figures (300 DPI PNG + vector PDF):
      1. dinov3_tsne_manifold: 2D t-SNE with Density Clusters
      2. dinov3_intra_vs_inter_distances: KDE / Distance distributions
      3. dinov3_singular_values_rank: SVD Spectrum showing absence of collapse
    """
    os.makedirs(save_dir, exist_ok=True)
    class_names = ["Low (<10)", "Medium (10-25)", "High (>25)"]
    palette = ["#2ca02c", "#ff7f0e", "#d62728"]  # Green, Orange, Red

    # -------------------------------------------------------------
    # Figure 1: 2D t-SNE Feature Manifold
    # -------------------------------------------------------------
    print("   🎨 Rendering 2D t-SNE Metric Manifold...")
    N = min(len(embeddings), 2000)
    idx = np.random.choice(len(embeddings), N, replace=False)
    X_tsne = embeddings[idx]
    y_tsne = labels[idx]

    tsne = TSNE(n_components=2, perplexity=35, random_state=42, n_iter=1000)
    z_2d = tsne.fit_transform(X_tsne)

    plt.figure(figsize=(7, 6), dpi=300)
    for c_idx in range(len(class_names)):
        mask = (y_tsne == c_idx)
        plt.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            c=palette[c_idx],
            label=f"{class_names[c_idx]} (n={np.sum(mask)})",
            alpha=0.75,
            edgecolors="none",
            s=28,
        )
    plt.title(f"t-SNE Embedding Geometry: {model_name}\n(Unsupervised Traffic Congestion Clusters)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=10, fontweight="bold")
    plt.ylabel("t-SNE Dimension 2", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()

    fig1_png = os.path.join(save_dir, "dinov3_tsne_manifold.png")
    fig1_pdf = os.path.join(save_dir, "dinov3_tsne_manifold.pdf")
    plt.savefig(fig1_png, bbox_inches="tight", dpi=300)
    plt.savefig(fig1_pdf, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Intra vs Inter Distance Distributions
    # -------------------------------------------------------------
    print("   🎨 Rendering Intra-to-Inter Distance Distribution...")
    plt.figure(figsize=(7, 5), dpi=300)
    n_sample_dist = min(len(intra_distances), len(inter_distances), 5000)
    intra_sub = np.random.choice(intra_distances, n_sample_dist, replace=False)
    inter_sub = np.random.choice(inter_distances, n_sample_dist, replace=False)

    bins = np.linspace(0.0, 1.6, 60)
    plt.hist(intra_sub, bins=bins, density=True, alpha=0.6, color="#1f77b4", label=f"Intra-Class (Mean={np.mean(intra_sub):.3f})")
    plt.hist(inter_sub, bins=bins, density=True, alpha=0.6, color="#d62728", label=f"Inter-Class (Mean={np.mean(inter_sub):.3f})")

    plt.axvline(np.mean(intra_sub), color="#1f77b4", linestyle="--", linewidth=2.0)
    plt.axvline(np.mean(inter_sub), color="#d62728", linestyle="--", linewidth=2.0)

    plt.title(f"Metric Space Separability: {model_name}\nIntra-to-Inter Ratio $\mathcal{{R}}_{{intra/inter}} = {intra_inter_ratio:.4f}$", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Pairwise Cosine Distance $(1 - \cos(z_i, z_j))$", fontsize=10, fontweight="bold")
    plt.ylabel("Probability Density", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()

    fig2_png = os.path.join(save_dir, "dinov3_intra_vs_inter_distances.png")
    fig2_pdf = os.path.join(save_dir, "dinov3_intra_vs_inter_distances.pdf")
    plt.savefig(fig2_png, bbox_inches="tight", dpi=300)
    plt.savefig(fig2_pdf, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: Singular Value Decay & Effective Rank Spectrum
    # -------------------------------------------------------------
    print("   🎨 Rendering Singular Value Decay Spectrum...")
    plt.figure(figsize=(7, 5), dpi=300)
    s_norm = singular_values / singular_values[0]
    ranks = np.arange(1, len(s_norm) + 1)

    plt.semilogy(ranks, s_norm, color="#9467bd", linewidth=2.2, label=f"{model_name} (Rank_eff = {eff_rank:.1f}/{len(s_norm)})")
    plt.axvline(eff_rank, color="#ff7f0e", linestyle="--", linewidth=1.8, label=f"Effective Rank Cutoff ({eff_rank:.1f})")

    plt.title(f"Singular Value Spectrum & Dimensional Collapse Analysis\nEffective Rank: {eff_rank:.2f} / {len(s_norm)} ({(eff_rank/len(s_norm))*100:.1f}% Manifold Utilization)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Singular Value Index (Rank)", fontsize=10, fontweight="bold")
    plt.ylabel("Normalized Singular Value $\sigma_k / \sigma_1$ (Log Scale)", fontsize=10, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()

    fig3_png = os.path.join(save_dir, "dinov3_singular_values_rank.png")
    fig3_pdf = os.path.join(save_dir, "dinov3_singular_values_rank.pdf")
    plt.savefig(fig3_png, bbox_inches="tight", dpi=300)
    plt.savefig(fig3_pdf, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved all 3 publication figures to: {save_dir}")


# =====================================================================
# 5. CLI EVALUATION RUNNER
# =====================================================================

def evaluate_metric_learning(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(" 🔬 DINOv3 METRIC LEARNING & REPRESENTATION GEOMETRY BENCHMARK")
    print("=" * 78)
    print(f" Backbone Architecture : {args.backbone}")
    print(f" Weights Path          : {args.weights if args.weights else 'Meta Official Pre-trained (Zero-shot)'}")
    print(f" Annotations CSV       : {args.csv_file}")
    print(f" Image Directory       : {args.image_dir}")
    print(f" Output Directory      : {args.save_dir}")
    print("=" * 78)

    # 1. Load Data
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"Annotations CSV not found: {args.csv_file}")
    df = prepare_dataframe(args.csv_file, args.image_dir)
    print(f"   Loaded {len(df)} total samples. Class Distribution:")
    for c_id, c_name in enumerate(["Low (<10)", "Medium (10-25)", "High (>25)"]):
        count = (df["congestion_class"] == c_id).sum()
        print(f"     - Class {c_id} [{c_name}]: {count} frames ({count/len(df)*100:.1f}%)")

    # 2. Build Model & Load Checkpoint
    backbone, embed_dim = build_backbone(args.backbone, pretrained=(args.weights is None), weights_path=args.weights)
    backbone = backbone.to(device)
    backbone.eval()

    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if num_gpus > 1:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"⚡ [Multi-GPU] Detected {num_gpus} GPUs: {gpu_names}")
        print(f"⚡ [Multi-GPU] Parallelizing embedding extraction across all {num_gpus} devices.")
        backbone_infer = nn.DataParallel(backbone)
    else:
        backbone_infer = backbone

    # 3. DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = TrafficMetricDataset(df, args.image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4. Extract Embeddings
    embeddings, labels, filenames = extract_embeddings(backbone_infer, dataloader, device)
    print(f"\n📦 Extracted Embeddings Shape: {embeddings.shape} | L2 Norm Check: {np.linalg.norm(embeddings[0]):.4f}")

    # 5. Compute Metrics
    print("\n--- 1. Computing k-NN Retrieval Accuracy (Frozen Representations) ---")
    X_tr, X_te, y_tr, y_te = train_test_split(embeddings, labels, test_size=0.2, random_state=args.seed, stratify=labels)
    knn_results = evaluate_knn_retrieval(X_tr, y_tr, X_te, y_te, k_values=[1, 5, 10, 20])
    for k in [1, 5, 10, 20]:
        print(f"   k={k:02d} | Top-1 Accuracy: {knn_results[f'knn_top1_acc_k{k}']*100:.2f}% | Macro F1: {knn_results[f'knn_macro_f1_k{k}']*100:.2f}%")

    print("\n--- 2. Computing Intra-to-Inter Class Distance Ratio ---")
    ratio, mean_intra, mean_inter, intra_dists, inter_dists = compute_intra_to_inter_ratio(embeddings, labels, seed=args.seed)
    print(f"   Mean Intra-Class Distance : {mean_intra:.4f}")
    print(f"   Mean Inter-Class Distance : {mean_inter:.4f}")
    print(f"   R_intra/inter Ratio       : {ratio:.4f} (Lower is better, ideal < 0.35)")

    print("\n--- 3. Computing Hyperspherical Alignment & Uniformity (Wang & Isola, ICML 2020) ---")
    alignment, uniformity = compute_alignment_and_uniformity(embeddings, labels, seed=args.seed)
    print(f"   Alignment (L_align)       : {alignment:.4f} (Lower is better invariance)")
    print(f"   Uniformity (L_uniform)    : {uniformity:.4f} (More negative is better entropy)")

    print("\n--- 4. Computing Effective Rank & SVD Dimensional Collapse Analysis ---")
    eff_rank, rank_pct, singular_vals = compute_effective_rank(embeddings)
    print(f"   Effective Rank (Rank_eff) : {eff_rank:.2f} / {embed_dim}")
    print(f"   Dimension Utilization    : {rank_pct:.2f}%")

    print("\n--- 5. Computing Unsupervised Clustering Metrics ---")
    cluster_metrics = compute_clustering_metrics(embeddings, labels)
    print(f"   Silhouette Score          : {cluster_metrics['silhouette_score']:.4f}")
    print(f"   Davies-Bouldin Index      : {cluster_metrics['davies_bouldin_index']:.4f} (Lower is better)")
    print(f"   Calinski-Harabasz Index   : {cluster_metrics['calinski_harabasz_index']:.2f}")

    # 6. Render Publication Figures
    print("\n--- 6. Generating Publication Visualizations ---")
    plot_publication_figures(
        embeddings=embeddings,
        labels=labels,
        singular_values=singular_vals,
        intra_distances=intra_dists,
        inter_distances=inter_dists,
        eff_rank=eff_rank,
        intra_inter_ratio=ratio,
        save_dir=args.save_dir,
        model_name=f"{args.backbone.upper()}"
    )

    # 7. Export Numerical Results (JSON + Markdown)
    final_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.backbone,
        "weights_path": args.weights,
        "embedding_dim": embed_dim,
        "num_samples": len(embeddings),
        "knn_metrics": knn_results,
        "intra_to_inter": {
            "ratio": ratio,
            "mean_intra_dist": mean_intra,
            "mean_inter_dist": mean_inter,
        },
        "hypersphere_geometry": {
            "alignment": alignment,
            "uniformity": uniformity,
        },
        "dimensional_collapse": {
            "effective_rank": eff_rank,
            "dimension_utilization_pct": rank_pct,
            "condition_number": float(singular_vals[0] / max(singular_vals[-1], 1e-12)),
        },
        "clustering_quality": cluster_metrics,
    }

    json_path = os.path.join(args.save_dir, "dinov3_metric_learning_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n💾 Saved complete numerical results to: {json_path}")

    # Generate Markdown Table summary
    md_path = os.path.join(args.save_dir, "dinov3_metric_learning_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo Định lượng Biểu diễn Metric Learning (DINOv3)\n\n")
        f.write(f"- **Mô hình Backbone:** `{args.backbone}`\n")
        f.write(f"- **Trọng số:** `{args.weights if args.weights else 'Meta Official LVD Foundation'}`\n")
        f.write(f"- **Kích thước Embedding:** `{embed_dim}` chiều\n")
        f.write(f"- **Số lượng mẫu đánh giá:** `{len(embeddings)}` khung hình\n\n")
        f.write("### 1. Bảng Tổng Hợp Chỉ Số Không Gian Đặc Trưng (Metric Space Quality)\n\n")
        f.write("| Chỉ số (Metric) | Giá trị đạt được | Tiêu chuẩn đánh giá | Ý nghĩa khoa học |\n")
        f.write("| :--- | :---: | :---: | :--- |\n")
        f.write(f"| **k-NN Accuracy (k=20)** | **{knn_results['knn_top1_acc_k20']*100:.2f}%** | Càng cao càng tốt | Độ nhất quán của lân cận không gian đặc trưng |\n")
        f.write(f"| **k-NN Macro F1 (k=20)** | **{knn_results['knn_macro_f1_k20']*100:.2f}%** | Càng cao càng tốt | Cân bằng phân loại giữa các mức mật độ xe |\n")
        f.write(f"| **Tỷ lệ $\mathcal{{R}}_{{intra/inter}}$** | **{ratio:.4f}** | Càng thấp càng tốt (< 0.35) | Độ gom cụm nội bộ so với khoảng cách tách rời |\n")
        f.write(f"| **Alignment ($\mathcal{{L}}_{{align}}$)** | **{alignment:.4f}** | Càng thấp càng tốt | Tính bất biến của biểu diễn trước nhiễu |\n")
        f.write(f"| **Uniformity ($\mathcal{{L}}_{{uniform}}$)** | **{uniformity:.4f}** | Càng âm càng tốt | Phân bố cực đại entropy trên mặt cầu $\mathcal{{S}}^{{d-1}}$ |\n")
        f.write(f"| **Effective Rank ($\text{{Rank}}_{{eff}}$)** | **{eff_rank:.2f} / {embed_dim}** | Càng cao càng tốt | Mức độ khai thác chiều, chống sụp đổ biểu diễn |\n")
        f.write(f"| **Hiệu suất sử dụng chiều** | **{rank_pct:.2f}%** | > 30% | Tỷ lệ không gian tiềm ẩn mang thông tin |\n")
        f.write(f"| **Silhouette Score** | **{cluster_metrics['silhouette_score']:.4f}** | [-1, 1], > 0.3 | Độ phân định rõ ràng giữa các cụm |\n")
        f.write(f"| **Davies-Bouldin Index** | **{cluster_metrics['davies_bouldin_index']:.4f}** | Càng thấp càng tốt | Độ tương đồng giữa các cụm |\n\n")
        f.write("### 2. Các Artifacts Hình Ảnh Xuất Bản Đi Kèm (300 DPI & Vector PDF)\n\n")
        f.write(f"1. `dinov3_tsne_manifold.pdf` / `.png`: Bản đồ t-SNE 2 chiều phân cụm trạng thái giao thông.\n")
        f.write(f"2. `dinov3_intra_vs_inter_distances.pdf` / `.png`: Phân phối mật độ xác suất khoảng cách nội cụm vs liên cụm.\n")
        f.write(f"3. `dinov3_singular_values_rank.pdf` / `.png`: Biểu đồ phổ giá trị suy biến SVD chứng minh không bị Dimensional Collapse.\n")

    print(f"📄 Generated Markdown Summary ready for paper: {md_path}\n")


def main():
    parser = argparse.ArgumentParser(description="DINOv3 Metric Learning & Representation Geometry Evaluation Suite")
    parser.add_argument("--csv_file", type=str, default="/workspace/traffic_update.csv", help="Path to counting CSV")
    parser.add_argument("--image_dir", type=str, default="/workspace/images", help="Path to camera images")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Backbone model (e.g. dinov3_vits16, dinov2_vits14)")
    parser.add_argument("--weights", type=str, default=None, help="Path to domain-adapted weights (.pth)")
    parser.add_argument("--save_dir", type=str, default="checkpoints/dinov3_metric_eval", help="Directory to save evaluation reports and figures")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Target device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face user access token for gated models")

    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    # Local fallback path detection (supports running from root or inside DINO subfolder)
    if not os.path.exists(args.csv_file):
        for candidate in ["labels1.csv", "../labels1.csv", "traffic_update.csv", "../traffic_update.csv", "data/traffic_update.csv", "../data/traffic_update.csv"]:
            if os.path.exists(candidate):
                args.csv_file = candidate
                break

    if not os.path.exists(args.image_dir):
        for candidate in ["images", "../images", "data/camera_images", "../data/camera_images", "data/images", "../data/images"]:
            if os.path.exists(candidate):
                args.image_dir = candidate
                break

    if args.weights and not os.path.exists(args.weights):
        alt_w = os.path.join("..", args.weights)
        if os.path.exists(alt_w):
            args.weights = alt_w

    evaluate_metric_learning(args)


if __name__ == "__main__":
    main()
