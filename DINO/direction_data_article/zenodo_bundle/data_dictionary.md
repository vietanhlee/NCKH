# Data Dictionary: IC4SD-TrafficSnap Dataset

This document provides the canonical schema, file organization, data types, and attribute definitions for the **IC4SD-TrafficSnap** multi-camera image time-series and geospatial road network dataset.

---

## 1. Directory Structure

```text
IC4SD-TrafficSnap/
├── LICENSE                                    # Multi-part licensing instrument (Academic, ODbL, MIT)
├── .zenodo.json                               # Zenodo deposition metadata schema
├── CITATION.cff                               # Citation File Format (v1.2.0)
├── README.md                                  # Human-readable documentation & quickstart
├── data_dictionary.md                         # Canonical attribute specification (This File)
├── checksums.sha256                           # SHA-256 integrity verification hashes
│
├── metadata/                                  # Spatial metadata and road network attributes
│   ├── routes.csv                             # Primary camera registry: ID, Location, CamID, coords & elevations
│   ├── stations.csv                           # Station topology registry, degrees & node roles
│   └── road_network_distance.csv              # Directed OpenStreetMap distance matrix (CSV)
│
├── graph/                                     # Machine-learning ready derived graph tensors
│   ├── distance_km.npy                        # Directed routing distances in kilometers (608x608)
│   ├── direction.npy                          # Directional connectivity adjacency (608x608)
│   └── edges.csv                              # Tabular list of 2,450 valid directed corridor links
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
- **Description:** Canonical registry, geographic coordinates, and infrastructure attributes of all 608 traffic surveillance camera stations across Ho Chi Minh City.

| Field Name | Type | Unit | Range / Constraints | Description |
|:---|:---|:---|:---|:---|
| `station_id` | Integer | Identifier | $[1, 657]$ (608 unique active IDs) | Unique numeric station index, corresponding to node indices in graph tensors and file prefix in snapshot archives. |
| `station_name` | String | Text | UTF-8 String | Textual name and landmark of the intersection or road link (e.g., `Trần Quang Khải - Trần Khắc Chân`). |
| `CamID` | String | Hex Token | 24-char hex string | Hardware stream identifier token utilized by the municipal ingestion gateway. |
| `latitude` | Float | Degrees | $[10.65^\circ N, 10.95^\circ N]$ | WGS 84 geographic latitude coordinate of the surveillance camera station. |
| `longitude` | Float | Degrees | $[106.50^\circ E, 106.85^\circ E]$ | WGS 84 geographic longitude coordinate of the surveillance camera station. |
| `district` | String | Categorical | Municipal districts of HCMC | Administrative district location (e.g., `District 1`, `Binh Thanh`, `Thu Duc City`). |
| `road_type` | String | Categorical | `{expressway, arterial, collector, roundabout}` | Functional road classification of the monitored roadway. |
| `camera_elevation_m`| Float | Meters | $[6.0, 15.0]$ | Mounting height of the camera pole above road surface level from technical installation records. |

---

### 2.2 File: `metadata/road_network_distance.csv`
- **Dimensions:** $608 \times 608$ matrix.
- **Unit of Distance:** Kilometers (km).
- **Description:** Shortest driving path distance from station $i$ (row) to station $j$ (column) along OpenStreetMap navigable corridors.
- **Interpretation of Special Values:**
  - `NaN` / Blank: Shortest driving distance exceeds the 6.0 km cutoff threshold, or no physical road connection exists.
  - `0.0` (Diagonal): Self-distance ($i = j$).
  - `0.0` (Off-diagonal): Co-located cameras situated on the same physical intersection gantry.

---

### 2.3 Files: `graph/distance_km.npy`, `graph/direction.npy`, `graph/edges.csv`
- **`distance_km.npy`:** Float64 array of shape `(608, 608)` containing physical shortest driving distances in kilometers. Unconnected pairs ($>6.0$~km) are encoded as `0.0`.
- **`direction.npy`:** Int32 binary array of shape `(608, 608)` representing directional connectivity ($1$ if valid directed corridor exists $\le 6.0$~km, $0$ otherwise).
- **`edges.csv`:** Tabular list of 2,450 directed corridor links with columns:
  - `source_station_id`: Origin camera station ID ($1 \le \text{ID} \le 657$).
  - `target_station_id`: Destination camera station ID ($1 \le \text{ID} \le 657$).
  - `distance_km`: Driving distance along the road network in kilometers.
  - `distance_m`: Driving distance along the road network in meters.
  - `is_bidirectional`: Boolean indicator for whether the reverse edge is present.

---

### 2.4 Visual Snapshot Files
- **Naming Pattern:** `{station_id}_{unix_timestamp}.jpg`
- **Format:** RGB JPEG (JFIF standard, 8-bit depth, compression quality factor $\approx 75$--$80$).
- **Resolution:** $512 \times 288$ pixels (16:9 aspect ratio).
- **Nominal Sampling Interval:** $\Delta T \approx 300$~s (5 minutes), empirical mean $269.0 \pm 239.7$~s (median $263.0$~s).
- **Acquisition Timestamp Offset:** $\Delta t_{\text{lag}} = 15.0 \pm 4.2$~s relative to hardware camera clocks.
- **Privacy Assurance:** Elevated mounting ($>6$~m) and oblique geometry ensure zero legible human faces or license plates (95\% CI upper bound $\le 4.2 \times 10^{-6}$ via Hanley and Lippman-Hand Rule of Three across the complete census of 714,123 audited frames).
