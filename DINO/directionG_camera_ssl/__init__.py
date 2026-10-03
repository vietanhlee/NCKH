"""
Direction G: Vehicle-Centric SSL Representation Learning from Sparse Static-Camera Feeds
"""
from .features import FrozenExtractor
from .tam import PositionStats, GMMCalibrator
from .masking import agm_sample, map_pi_to_crop
from .srs import static_region_swap
from .losses import DINOLoss, iBOTPatchLoss, KoLeoLoss, RICLoss

__all__ = [
    "FrozenExtractor",
    "PositionStats",
    "GMMCalibrator",
    "agm_sample",
    "map_pi_to_crop",
    "static_region_swap",
    "DINOLoss",
    "iBOTPatchLoss",
    "KoLeoLoss",
    "RICLoss",
]
