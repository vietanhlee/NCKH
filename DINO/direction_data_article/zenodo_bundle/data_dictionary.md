# Data Dictionary: HCMC-TrafficSnap Dataset

This document provides the canonical schema, file organization, data types, and attribute definitions for the **HCMC-TrafficSnap** multi-camera image time-series and directed spatial graph dataset.

---

## 1. Directory Structure

```text
HCMC-TrafficSnap/
├── LICENSE                                    # CC BY 4.0 / Research Use Legal Instrument
├── .zenodo.json                               # Zenodo Deposition Ingestion Metadata
├── CITATION.cff                               # Citation File Format (v1.2.0)
├── README.md                                  # Human-Readable Documentation & Quickstart
├── data_dictionary.md                         # Canonical Attribute Specification (This File)
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

## 2. File Specifications & Attribute Schemas

### 2.1 File: `metadata/routes.csv`
- **Total Records:** 608 rows (excluding header).
- **Encoding:** UTF-8, Comma-Separated Values.
- **Description:** Contains geographic and physical characteristics of each active camera station.

| Field Name | Type | Unit | Range / Constraints | Description |
|:---|:---|:---|:---|:---|
| `station_id` | Integer | Identifier | $[1, 657]$ (608 unique active IDs) | Unique numeric identifier for the camera station, matching node indices in the distance matrix and image prefixes. |
| `station_name` | String | Text | UTF-8 String | Vietnamese textual description of the intersection or road link (e.g., `Nga tu Hang Xanh`, `Cau Sai Gon`). |
| `latitude` | Float | Degrees | $[10.65^\circ N, 10.95^\circ N]$ | WGS 84 geographic latitude coordinate of the camera pole. |
| `longitude` | Float | Degrees | $[106.50^\circ E, 106.85^\circ E]$ | WGS 84 geographic longitude coordinate of the camera pole. |
| `district` | String | Categorical | Municipal districts of HCMC | Administrative district location (e.g., `District 1`, `Binh Thanh`, `Thu Duc City`). |
| `road_type` | String | Categorical | `{expressway, arterial, collector, roundabout}` | Functional road classification of the monitored roadway. |
| `camera_elevation_m`| Float | Meters | $[6.0, 15.0]$ | Mounting height of the camera above road surface level. |

---

### 2.2 Files: `metadata/road_network_distance.xlsx` & `metadata/road_network_distance.csv`
- **Dimensions:** $608 \times 608$ matrix.
- **Unit of Distance:** Kilometers (km).
- **Description:** Shortest driving path distance from station $i$ (row) to station $j$ (column) along OpenStreetMap navigable corridors.
- **Interpretation of Special Values:**
  - `NaN` / Blank: Shortest driving distance exceeds the 5.0 km cutoff threshold, or no physical road connection exists.
  - `0.0` (Diagonal): Self-distance ($i = j$).
  - `0.0` (Off-diagonal): Co-located cameras situated on the same physical intersection gantry.

---

### 2.3 Files: `graph/distance_m.npy`, `graph/direction.npy`, `graph/edges.csv`
- **`distance_m.npy`:** Float32 array of shape `(608, 608)` containing routing distances in meters. Unconnected pairs ($>5.0$~km) are encoded as `0.0`.
- **`direction.npy`:** Int32 array of shape `(608, 608)` binary adjacency matrix ($1$ if directed link exists $\le 5.0$~km, $0$ otherwise).
- **`edges.csv`:** Table containing 2,420 rows with columns:
  - `source_station_id`: Origin camera station ID ($1 \le \text{ID} \le 657$).
  - `target_station_id`: Destination camera station ID ($1 \le \text{ID} \le 657$).
  - `distance_meters`: Driving distance along OSM road network in meters.

---

### 2.4 Visual Snapshot Files
- **Naming Pattern:** `{station_id}_{unix_timestamp}.jpg`
- **Format:** RGB JPEG (JFIF standard, 8-bit depth).
- **Resolution:** $512 \times 288$ pixels (16:9 aspect ratio).
- **Nominal Sampling Interval:** $\Delta T \approx 300$~s (5 minutes).
- **Acquisition Timestamp Offset:** $\Delta t_{\text{lag}} = 15.0 \pm 4.2$~s relative to hardware camera clock.
- **Privacy Assurance:** Due to mounting height ($>6$~m) and sub-Nyquist Ground Sampling Distance ($2.73$--$3.25$~cm/pixel), individual faces and license plates are physically unresolvable.
