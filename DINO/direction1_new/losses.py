"""
Backward-compatible re-export shim cho direction1_new.losses.
Mã nguồn chính đã được tái cấu trúc sang `direction1_new.core.losses`.
"""
from .core.losses import (
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
