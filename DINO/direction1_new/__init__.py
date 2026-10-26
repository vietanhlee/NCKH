"""
=============================================================================
  PACKAGE: direction1_new (Direction G)
  Vehicle-Centric Self-Supervised Representation Learning from Sparse Static-Camera Feeds
  Học biểu diễn phương tiện bất biến góc quay camera qua TAM + SRS + AGM + RIC + CamAdv.

  Cấu trúc Subpackages chuẩn Production:
    - core          : Mô hình (Student/Teacher/DINOHead), TAM, AGM, SRS, Losses
    - data          : TrafficCameraDataset, SyntheticTrafficDataset, CameraGroupedSampler
    - engine        : CLI Config, Smooth LR Scheduler, Checkpoint Manager, TAM Warmup
    - visualization : 8-Column Progress Visualizer, Paper Figures, Attention & PCA
    - evaluation    : Camera Probe, Downstream CKA/eRank, Audit Pipeline, Ablations
=============================================================================
"""
from .core import (
    DINOHead,
    DirectionGStudentModel,
    DirectionGTeacherModel,
    FrozenExtractor,
    PositionStats,
    GMMCalibrator,
    refine_traffic_foreground_map,
    agm_sample,
    batch_agm_sample,
    map_pi_to_crop,
    static_region_swap,
    batch_static_region_swap,
    dilate_mask_2d,
    DINOLoss,
    iBOTPatchLoss,
    KoLeoLoss,
    RICLoss,
    pool_vehicle_embedding,
    CameraAdversarialHead,
    compute_dann_alpha,
    PatchVarianceRegularizer,
    FeatureAnchorLoss,
)
from .data import (
    TrafficCameraDataset,
    SyntheticTrafficDataset,
    SimpleImageDataset,
    CameraGroupedSampler,
)

__all__ = [
    "DINOHead",
    "DirectionGStudentModel",
    "DirectionGTeacherModel",
    "FrozenExtractor",
    "PositionStats",
    "GMMCalibrator",
    "refine_traffic_foreground_map",
    "agm_sample",
    "batch_agm_sample",
    "map_pi_to_crop",
    "static_region_swap",
    "batch_static_region_swap",
    "dilate_mask_2d",
    "DINOLoss",
    "iBOTPatchLoss",
    "KoLeoLoss",
    "RICLoss",
    "pool_vehicle_embedding",
    "CameraAdversarialHead",
    "compute_dann_alpha",
    "PatchVarianceRegularizer",
    "FeatureAnchorLoss",
    "TrafficCameraDataset",
    "SyntheticTrafficDataset",
    "SimpleImageDataset",
    "CameraGroupedSampler",
]
