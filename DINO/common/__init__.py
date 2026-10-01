"""
=============================================================================
 Common utilities package for DINO Traffic Self-Supervised Learning
=============================================================================
"""

from .matcher import TrafficPairMatcher
from .subtraction import BackgroundSubtractor
from .backbone_loader import get_dino_backbone, extract_tokens
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
    "get_available_devices",
    "setup_multi_gpu",
    "unwrap_model",
    "clean_state_dict",
    "smart_load_state_dict",
    "save_checkpoint",
    "load_checkpoint",
]
