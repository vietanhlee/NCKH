"""
Backward-compatible re-export shim cho direction1_new.features.
Mã nguồn chính đã được tái cấu trúc sang `direction1_new.core.features`.
"""
from .core.features import FrozenExtractor

__all__ = ["FrozenExtractor"]
