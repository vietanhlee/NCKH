"""
Backward-compatible re-export shim cho direction1_new.sampler.
Mã nguồn chính đã được tái cấu trúc sang `direction1_new.data.sampler`.
"""
from .data.sampler import CameraGroupedSampler

__all__ = ["CameraGroupedSampler"]
