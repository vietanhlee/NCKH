"""
=============================================================================
 Hướng 8: Self-Supervised Road Surface Condition Estimation
 Package Initialization
=============================================================================
"""

from .dataset import RoadSurfaceDataset
from .models import RoadConditionClassifier
from .losses import SurfaceConsistencyLoss

__all__ = ["RoadSurfaceDataset", "RoadConditionClassifier", "SurfaceConsistencyLoss"]
