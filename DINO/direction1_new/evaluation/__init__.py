"""
=============================================================================
  SUBPACKAGE: direction1_new.evaluation
  Bao gồm các công cụ đánh giá định lượng và kiểm toán học thuật (Audit & Eval):
    - eval_camera_probe: Đánh giá Shortcut Camera ID Linear Probe
    - eval_downstream: Đánh giá chất lượng biểu diễn, Effective Rank, Linear CKA
    - audit_tam_agm_srs_probe: Kiểm toán toàn diện TAM, AGM, SRS và Downstream Probe
    - run_ablations: Tự động hóa ma trận 7 thí nghiệm Ablation Study
=============================================================================
"""
from .eval_downstream import (
    compute_linear_cka,
    compute_effective_rank,
    evaluate_representation_quality,
)

__all__ = [
    "compute_linear_cka",
    "compute_effective_rank",
    "evaluate_representation_quality",
]
