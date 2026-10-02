"""
=============================================================================
 Hướng 4: Spatio-Temporal DINO for Continuous Road Space Occupancy 
          and Congestion Level Estimation
 Package Initialization
=============================================================================
"""

from .dataset import TemporalTrafficDataset
from .models import SpatioTemporalDensityNet, DeltaSpatialEncoder
from .losses import SpatioTemporalDensityLoss

# Aliases cho tương thích ngược
TemporalTrafficEncoder = SpatioTemporalDensityNet
TemporalContrastiveLoss = SpatioTemporalDensityLoss

__all__ = [
    "TemporalTrafficDataset",
    "SpatioTemporalDensityNet",
    "DeltaSpatialEncoder",
    "SpatioTemporalDensityLoss",
    "TemporalTrafficEncoder",
    "TemporalContrastiveLoss",
]
