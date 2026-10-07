"""
=============================================================================
Package: dual_agents
Dự án: IC4SD-TrafficSnap Data Article (Elsevier Data in Brief / Scopus Q1)
Tác giả: Viet-Anh Le & Dr. Khanh Nguyen-Trong (IC4SD Lab, PTIT)
Tiêu chuẩn: Chuẩn sản xuất (Production-Ready)
Mô tả nghiệp vụ:
  Hệ thống Đa Tác Tử Cộng Tác & Phản Biện Khoa Học (Dual-Agent Peer-Review System).
  Bao gồm:
  1. PaperDrafterNarrativeAgent: Tác tử biên soạn, nâng cấp văn phong học thuật
     và tích hợp thực nghiệm ("Agent Chém Gió & Viết Paper").
  2. ScientificReviewerAuditorAgent: Tác tử phản biện độc lập, khách quan,
     trung lập và kiểm toán tính nhất quán số liệu ("Agent Phản Biện & Kiểm Toán").
  3. EmpiricalAuditorEngine: Bộ máy đo đạc và trích xuất Ground-Truth từ dữ liệu gốc.
  4. DualAgentOrchestrator: Trình điều phối vòng lặp tương tác phản biện đa vòng.
=============================================================================
"""

from dual_agents.empirical_auditor_engine import EmpiricalAuditorEngine, haversine_distance_meters
from dual_agents.agent_scientific_reviewer import ScientificReviewerAuditorAgent
from dual_agents.agent_paper_drafter import PaperDrafterNarrativeAgent
from dual_agents.feedback_loop_orchestrator import DualAgentOrchestrator

__all__ = [
    "EmpiricalAuditorEngine",
    "haversine_distance_meters",
    "ScientificReviewerAuditorAgent",
    "PaperDrafterNarrativeAgent",
    "DualAgentOrchestrator"
]
