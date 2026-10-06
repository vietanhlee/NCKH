# HCMC-TrafficSnap: A City-Scale Multi-Camera Image Time-Series and Directed Spatial Graph Dataset for Heterogeneous Urban Traffic

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22929940.svg)](https://doi.org/10.5281/zenodo.22929940)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

---

## 1. Overview

**HCMC-TrafficSnap** is an open-access, city-scale benchmark composed of **over 1,000,000 unlabeled surveillance camera snapshots** ($\approx 1.22$ million total frames) coupled with an **OpenStreetMap-derived directed spatial road network graph**, acquired across **608 active municipal surveillance camera stations** in **Ho Chi Minh City (HCMC), Vietnam**.

The dataset spans **7 continuous calendar days (168 consecutive hours)**, totaling approximately **80 GB** of visual imagery capturing the dynamics of Southeast Asian urban mobility—a regime characterized by intense motorcycle density (70–80% modal share), high spatial occlusions, non-lane-based flow, and tropical monsoon illumination shifts.

### Key Benchmark Characteristics
- **Scale:** 608 active municipal camera stations (indexed in range `[1, 657]`).
- **Visual Modality:** Discrete $512 \times 288$ pixel JPEG snapshots with an average file size of $65.87 \pm 9.19$~KB.
- **Temporal Span:** 7 continuous days (Monday 00:00 to Sunday 23:59 ICT) sampled at nominal 5-minute intervals ($\Delta T \approx 300$~s, $\sim 12$ snapshots/hour/station).
- **Crawl Clock Offset:** Average client crawl time lag relative to hardware camera clocks is $\Delta t_{\text{lag}} = 15.0 \pm 4.2$~s.
- **Spatial Topology:** $608 \times 608$ directed road routing distance matrix from OpenStreetMap (2,420 valid corridors $\le 5.0$~km), capturing 1,054 strictly one-way links and 232 distance-asymmetric bidirectional pairs.
- **Privacy by Design:** Camera elevations ($>6$~m) and sub-Nyquist ground sampling distance physically guarantee zero resolvable human faces or vehicle license plates (0.00% PII).

---

## 2. Benchmark Specifications

| Attribute | Specification |
|:---|:---|
| **Geographic Location** | Ho Chi Minh City, Vietnam ($10^\circ 46' N, 106^\circ 40' E$) |
| **Active Camera Stations** | 608 fixed surveillance cameras |
| **Observation Duration** | 7 continuous calendar days (168 consecutive hours) |
| **Total Snapshot Volume** | $>1,000,000$ ($\approx 1.22 \times 10^6$ frames) |
| **Total Archive Size** | $\approx 80 \text{ GB}$ (uncompressed JPEG daily tarballs) |
| **Image Resolution** | Uniform $512 \times 288$ pixels (16:9 streaming snapshots) |
| **Image Naming Format** | `{station_id}_{unix_timestamp}.jpg` |
| **Annotation State** | **Unlabeled (Raw image time-series snapshots)** |
| **Spatial Graph** | $608 \times 608$ directed road network routing distance matrix (OpenStreetMap) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |

---

## 3. Repository Structure

```text
HCMC-TrafficSnap/
├── LICENSE                                    # CC BY 4.0 Open-Access Legal Instrument
├── .zenodo.json                               # Zenodo Deposition Ingestion Metadata
├── CITATION.cff                               # Citation File Format (v1.2.0)
├── README.md                                  # Human-Readable Documentation & Quickstart
├── data_dictionary.md                         # Canonical Attribute Specification
├── checksums.sha256                           # SHA-256 integrity verification hashes
│
├── metadata/                                  # Spatial metadata and topology
│   ├── routes.csv                             # Camera station geographic & administrative metadata
│   ├── road_network_distance.xlsx             # Directed OpenStreetMap distance matrix (Excel)
│   └── road_network_distance.csv              # Directed OpenStreetMap distance matrix (CSV)
│
├── graph/                                     # Machine-learning ready graph tensors
│   ├── distance_m.npy                         # Directed routing distances in meters (608x608)
│   ├── direction.npy                          # Directional connectivity adjacency (608x608)
│   └── edges.csv                              # Tabular list of 2,420 valid directed corridor links
│
├── sample_preview/                            # Starter demonstration subset (~500 MB)
│   └── sample_camera_sequences/               # Sample continuous frames from representative stations
│
└── code/                                      # Reference loaders and preprocessing utilities
    ├── __init__.py
    ├── load_unlabeled_images.py               # PyTorch Dataset implementation for snapshots
    ├── camera_sampler.py                      # Multi-camera multi-day SSL batch sampler
    └── graph_utils.py                         # Dual-transition directed random walk & Laplacian tools
```

---

## 4. Quick-Start Usage

### 4.1 PyTorch Dataset & Batch Sampling
```python
from code.load_unlabeled_images import TrafficCameraUnlabeledDataset
from code.camera_sampler import CameraGroupedSampler
from torch.utils.data import DataLoader

# 1. Initialize dataset
dataset = TrafficCameraUnlabeledDataset(
    image_dir="sample_preview/sample_camera_sequences"
)

# 2. Multi-camera SSL sampler (e.g., 8 cameras, 4 snapshots each)
sampler = CameraGroupedSampler(
    dataset=dataset,
    cameras_per_batch=8,
    samples_per_camera=4,
    shuffle=True
)

loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)
for images, metadata in loader:
    print(f"Batch shape: {images.shape}")
    print(f"Camera IDs: {metadata['station_id']}")
    break
```

### 4.2 Loading Directed Spatial Graph
```python
from code.graph_utils import load_road_graph, calculate_dual_directed_transition_matrices

# 1. Load weighted directed adjacency
adj, node_ids = load_road_graph(
    excel_path="metadata/road_network_distance.xlsx",
    distance_threshold_km=5.0
)

# 2. Compute DCRNN directed random walk matrices
P_forward, P_backward = calculate_dual_directed_transition_matrices(adj)
print(f"Forward transition: {P_forward.shape}, Backward transition: {P_backward.shape}")
```

---

## 5. Citation

```bibtex
@article{le2026hcmctrafficsnap,
  title={{HCMC-TrafficSnap: A City-Scale Multi-Camera Image Time-Series and Directed Spatial Graph Dataset for Heterogeneous Urban Traffic}},
  author={Le, Viet-Anh and Nguyen-Trong, Khanh},
  journal={Data in Brief},
  year={2026},
  doi={10.5281/zenodo.22929940}
}
```
