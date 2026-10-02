"""
=============================================================================
 Common utilities package for DINO Traffic Self-Supervised Learning
=============================================================================
"""

from .matcher import TrafficPairMatcher
from .subtraction import BackgroundSubtractor
from .backbone_loader import get_dino_backbone, extract_tokens, load_env_credentials
from .reliability import (
    estimate_static_mask_from_sequence,
    estimate_background_reliability,
    check_camera_alignment_phase_correlation,
)
from .degradation import BackgroundDegradationBenchmark
from .corrupt import FrameCorruptionSuite
from .gpu_utils import (
    get_available_devices,
    setup_multi_gpu,
    unwrap_model,
    clean_state_dict,
    smart_load_state_dict,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    "TrafficPairMatcher",
    "BackgroundSubtractor",
    "get_dino_backbone",
    "extract_tokens",
    "load_env_credentials",
    "estimate_static_mask_from_sequence",
    "estimate_background_reliability",
    "check_camera_alignment_phase_correlation",
    "BackgroundDegradationBenchmark",
    "FrameCorruptionSuite",
    "get_available_devices",
    "setup_multi_gpu",
    "unwrap_model",
    "clean_state_dict",
    "smart_load_state_dict",
    "save_checkpoint",
    "load_checkpoint",
]
