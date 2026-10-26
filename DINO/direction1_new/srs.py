"""
Backward-compatible re-export shim cho direction1_new.srs.
Mã nguồn chính đã được tái cấu trúc sang `direction1_new.core.srs`.
"""
from .core.srs import (
    static_region_swap,
    batch_static_region_swap,
    dilate_mask_2d,
)

__all__ = [
    "static_region_swap",
    "batch_static_region_swap",
    "dilate_mask_2d",
]
