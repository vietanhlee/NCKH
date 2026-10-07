# IC4SD-TrafficSnap: A City-Scale Multi-Camera Image Time-Series and Geospatial Road Network Dataset for Heterogeneous Urban Traffic

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22929940.svg)](https://doi.org/10.5281/zenodo.22929940)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: ODbL](https://img.shields.io/badge/License-ODbL-blue.svg)](https://opendatacommons.org/licenses/odbl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

---

## 1. Overview

**IC4SD-TrafficSnap** is an open-access multimodal dataset coupling **714,123 discrete surveillance camera snapshots** (44.38 GiB / 47.66 GB in standard stream JPEG format) across **608 indexed camera stations** and **geospatial road network mapping attributes** in **Ho Chi Minh City, Vietnam** (covering former urban districts of the metropolitan core).

As a direct derived representation based on physical road network geometry and sequential corridor adjacency on OpenStreetMap, a $608 \times 608$ directed spatial graph topology is provided for plug-and-play spatio-temporal modeling.

### Key Modalities & Specifications
- **Surveillance Stations:** 608 indexed municipal stations spanning arterial corridors and intersections across former urban districts in Ho Chi Minh City, fully mapped to geospatial coordinates and network topology.
- **Visual Image Time-Series:** Discrete $512 \times 288$ pixel JPEG snapshots ($65.17 \pm 13.36$~KB) acquired over 93.6 continuous hours (spanning 5 calendar days: Oct 2 to Oct 6, 2026) at an empirical sampling interval of $\Delta T = 269.0 \pm 239.7$~s (median: $263.0$~s; nominal target: $300$~s / 5.0 min).
- **Geospatial Road Network:** Camera GPS coordinates, mounting elevations, and pairwise shortest driving distance matrix ($608 \times 608$) derived via OSRM (v5.27.1) from OpenStreetMap.
- **Derived Directed Spatial Graph:** 2,450 valid directed corridors ($\le 6.0$~km), capturing 1,070 unidirectional links without reverse edges and 232 distance-asymmetric bidirectional pairs (alongside 458 symmetric pairs across 690 two-way connected dyads).
- **Quantified Negligible Privacy Risk:** Elevated mounting ($>6$~m) and downward oblique viewing geometry physically guarantee zero legible human faces or vehicle license plates (95\% CI upper bound $\le 4.2 \times 10^{-6}$ via Rule of Three across the complete census of 714,123 audited frames).

---

## 2. Dataset Specifications

| Attribute | Specification |
|:---|:---|
| **Geographic Location** | Ho Chi Minh City, Vietnam (former urban districts, $10^\circ 46' N, 106^\circ 40' E$) |
| **Indexed Stations** | 608 active surveillance cameras across municipal road network |
| **Observation Duration** | 93.6 continuous hours (spanning 5 calendar days) |
| **Total Snapshot Volume** | 714,123 frames |
| **Archive Storage Size** | 44.38 GiB (binary) / 47.66 GB (decimal) in standard stream JPEG format |
| **Image Resolution** | Uniform $512 \times 288$ pixels (16:9 streaming snapshots) |
| **Image Naming Format** | `{station_id}_{unix_timestamp}.jpg` |
| **Geospatial Road Network** | Station GPS coordinates, elevations, and OSM distance matrix |
| **Derived Graph** | $608 \times 608$ directed road network routing distance matrix ($|E| = 2,450$) |
| **Licensing** | Academic Research (imagery & metadata), ODbL (road network graph), MIT (code) |

---

## 3. Repository Structure

```text
IC4SD-TrafficSnap/
├── LICENSE                                    # Multi-part licensing instrument (Academic, ODbL, MIT)
├── .zenodo.json                               # Zenodo deposition metadata schema
├── CITATION.cff                               # Citation File Format (v1.2.0)
├── README.md                                  # Human-readable documentation & quickstart
├── data_dictionary.md                         # Attribute definitions & schema specifications
├── checksums.sha256                           # SHA-256 integrity verification hashes
│
├── metadata/                                  # Spatial metadata and routing attributes
│   ├── routes.csv                             # Primary camera registry: ID, Location, CamID, coords & elevations
│   ├── stations.csv                           # Station topology registry, degrees & node roles
│   └── road_network_distance.csv              # Directed OpenStreetMap distance matrix (CSV)
│
├── graph/                                     # Machine-learning ready derived graph tensors
│   ├── distance_km.npy                        # Directed routing distances in kilometers (608x608)
│   ├── direction.npy                          # Directional connectivity adjacency (608x608)
│   └── edges.csv                              # Tabular list of 2,450 valid directed corridor links
│
├── sample_preview/                            # Starter demonstration subset
│   └── sample_camera_sequences/               # Sample continuous frames from representative stations
│
└── code/                                      # Reference loaders and preprocessing utilities
    ├── __init__.py
    ├── load_unlabeled_images.py               # PyTorch Dataset implementation for snapshots
    ├── camera_sampler.py                      # Multi-camera multi-day SSL batch sampler
    └── graph_utils.py                         # Dual-transition directed random walk & Laplacian tools
```

---

## 4. Quickstart: PyTorch Data Loading & Graph Ingestion

### Ingestion of Image Time-Series
```python
from code.load_unlabeled_images import CameraSequenceDataset

dataset = CameraSequenceDataset(
    image_dir="sample_preview/sample_camera_sequences",
    resolution=(512, 288)
)
print(f"Loaded {len(dataset)} valid surveillance snapshots.")
```

### Ingestion of Directed Transition Operators
```python
import numpy as np
from code.graph_utils import load_directed_graph_operators

# Computes forward (Pf) and backward (Pb) diffusion matrices
Pf, Pb, W = load_directed_graph_operators(
    distance_npy_path="graph/distance_km.npy",
    sigma=1.09,
    cutoff_km=6.0
)
print("Diffusion operators ready. Pf shape:", Pf.shape)
```

---

## 5. Licensing & Terms of Use

- **Surveillance Snapshots & Metadata:** Aggregated from the public traffic web portal (giaothong.hochiminhcity.gov.vn) for non-commercial academic research and educational use.
- **Geospatial Road Network & Derived Graph Tensors:** Derived from OpenStreetMap data, licensed under the [Open Database License (ODbL v1.0)](https://opendatacommons.org/licenses/odbl/).
- **Code & Utility Scripts:** Licensed under the [MIT License](https://opensource.org/licenses/MIT).
