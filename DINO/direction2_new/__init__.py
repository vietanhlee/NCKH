"""
=============================================================================
 Hướng 2: Physics-Consistent Traffic Scene Decomposition & Road Inpainting
 (direction2_scene_decomposition)
=============================================================================
"""

from .models import (
    TrafficDecompositionNet,
    MultiScaleDecompDecoder,
    align_prior_illumination,
    compute_illumination_invariant_guide,
    inpaint_spatial_features,
)
from .losses import (
    DecompositionLoss,
    SceneDecompositionLossV2,
)
from .dataset import DecompositionDataset
from .scene_fit import SceneBasis, robust_weights
from .solve_ell import solve_ell

__all__ = [
    "TrafficDecompositionNet",
    "MultiScaleDecompDecoder",
    "align_prior_illumination",
    "compute_illumination_invariant_guide",
    "inpaint_spatial_features",
    "DecompositionLoss",
    "SceneDecompositionLossV2",
    "DecompositionDataset",
    "SceneBasis",
    "robust_weights",
    "solve_ell",
]
