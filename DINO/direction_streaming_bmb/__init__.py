"""
=============================================================================
 ST-BMB: Spatio-Temporal Background Memory Bank Suite
 Gói phần mềm phân rã cảnh và phân đoạn phương tiện streaming qua bộ nhớ nền
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .memory import BackgroundMemoryBank
from .attention import MemoryCrossAttention
from .models import StreamingDecompositionNet, BackgroundMemoryEncoder, PhotometricIlluminationAdaptor
from .losses import StreamingDecompositionLoss
from .dataset import SlidingWindowTrafficDataset

__all__ = [
    "BackgroundMemoryBank",
    "MemoryCrossAttention",
    "StreamingDecompositionNet",
    "BackgroundMemoryEncoder",
    "PhotometricIlluminationAdaptor",
    "StreamingDecompositionLoss",
    "SlidingWindowTrafficDataset",
]
