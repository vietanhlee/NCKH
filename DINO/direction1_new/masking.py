"""
Backward-compatible re-export shim cho direction1_new.masking.
Mã nguồn chính đã được tái cấu trúc sang `direction1_new.core.masking`.
"""
from .core.masking import (
    agm_sample,
    batch_agm_sample,
    map_pi_to_crop,
    sample_without_replacement,
)

__all__ = [
    "agm_sample",
    "batch_agm_sample",
    "map_pi_to_crop",
    "sample_without_replacement",
]
