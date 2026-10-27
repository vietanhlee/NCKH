"""
=============================================================================
 Subpackage `direction1_new.core`: Các thuật toán và mô hình lõi của Hướng 1 Mới
=============================================================================
"""

from .models import DINOHead, DirectionGStudentModel, DirectionGTeacherModel
from .features import FrozenExtractor
from .tam import PositionStats, GMMCalibrator, refine_traffic_foreground_map
from .masking import agm_sample, batch_agm_sample, map_pi_to_crop
from .srs import (
    static_region_swap,
    batch_static_region_swap,
    batch_vehicle_photometric_jitter,
    dilate_mask_2d,
)
from .losses import (
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
    "batch_vehicle_photometric_jitter",
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
]
