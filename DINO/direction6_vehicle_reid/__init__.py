"""
=============================================================================
 Hướng 6: Delta-Guided Unsupervised Vehicle Re-Identification Across Cameras
 Package Initialization
=============================================================================
"""

from .roi_extractor import DeltaRoIExtractor
from .models import VehicleReIDModel
from .losses import TrackletContrastiveLoss
from .matcher import VehicleReIDMatcher

__all__ = ["DeltaRoIExtractor", "VehicleReIDModel", "TrackletContrastiveLoss", "VehicleReIDMatcher"]
