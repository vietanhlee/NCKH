"""
=============================================================================
Module: test_dual_agents.py
Dự án: IC4SD-TrafficSnap Data Article (Elsevier Data in Brief / Scopus Q1)
Tác giả: Viet-Anh Le & Dr. Khanh Nguyen-Trong (IC4SD Lab, PTIT)
Tiêu chuẩn: Chuẩn sản xuất (Production-Ready)
Mô tả nghiệp vụ:
  Bộ kiểm thử tự động toàn diện (Comprehensive Automated Test Suite) cho:
  1. EmpiricalAuditorEngine: Nạp dữ liệu gốc, tính toán topo, tortuosity, PII, trắc quang.
  2. ScientificReviewerAuditorAgent: Kiểm toán tính nhất quán, phản biện luận điểm, sinh báo cáo.
  3. PaperDrafterNarrativeAgent: Xử lý feedback, sinh nội dung học thuật, nâng cấp LaTeX.
  4. DualAgentOrchestrator: Vận hành toàn bộ vòng lặp điều phối phản biện đa tác tử.
=============================================================================
"""

import os
import sys
import unittest
from pathlib import Path

# Đảm bảo mã hóa UTF-8 trên hệ điều hành Windows
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn project vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dual_agents.empirical_auditor_engine import EmpiricalAuditorEngine, haversine_distance_meters
from dual_agents.agent_scientific_reviewer import ScientificReviewerAuditorAgent
from dual_agents.agent_paper_drafter import PaperDrafterNarrativeAgent
from dual_agents.feedback_loop_orchestrator import DualAgentOrchestrator


class TestDualAgentSystem(unittest.TestCase):
    """
    Test Suite kiểm tra toàn diện hệ thống 2 tác tử.
    """

    @classmethod
    def setUpClass(cls):
        cls.project_root = BASE_DIR
        cls.engine = EmpiricalAuditorEngine(str(cls.project_root))
        cls.reviewer = ScientificReviewerAuditorAgent(str(cls.project_root))
        cls.drafter = PaperDrafterNarrativeAgent(str(cls.project_root))

    def test_01_haversine_distance(self):
        """Kiểm tra độ chính xác của hàm tính khoảng cách trắc địa Haversine."""
        # Khoảng cách giữa 2 điểm cùng tọa độ phải bằng 0
        d_zero = haversine_distance_meters(106.69, 10.79, 106.69, 10.79)
        self.assertAlmostEqual(d_zero, 0.0, places=3)

        # Khoảng cách giữa 2 điểm cách nhau ~1 km
        d_test = haversine_distance_meters(106.69, 10.79, 106.69, 10.80)
        self.assertGreater(d_test, 1000.0)
        self.assertLess(d_test, 1200.0)

    def test_02_graph_topology_audit(self):
        """Kiểm tra bộ máy kiểm toán topo đồ thị từ dữ liệu gốc."""
        res = self.engine.audit_graph_topology()
        self.assertEqual(res["num_nodes"], 608)
        self.assertEqual(res["num_edges"], 2450)
        self.assertEqual(res["connected_node_pairs"], 1760)
        self.assertEqual(res["unidirectional_pairs"], 1070)
        self.assertEqual(res["bidirectional_pairs"], 690)
        self.assertEqual(res["distance_asymmetric_reported"], 232)
        self.assertEqual(res["distance_symmetric_reported"], 458)
        self.assertEqual(res["wcc_count"], 6)
        self.assertEqual(res["scc_count"], 19)
        self.assertEqual(res["giant_component_size"], 599)
        self.assertAlmostEqual(res["spectral_radius_Pf"], 1.0, places=4)
        self.assertAlmostEqual(res["spectral_radius_Pb"], 1.0, places=4)
        self.assertLessEqual(res["laplacian_lambda_max"], 2.0)

    def test_03_network_tortuosity_and_spatial(self):
        """Kiểm tra thực nghiệm hệ số uốn khúc mạng và phân bố không gian."""
        res = self.engine.audit_network_tortuosity_and_spatial_density()
        self.assertGreater(res["tortuosity_sample_count"], 2000)
        self.assertAlmostEqual(res["tortuosity_mean"], 1.25, delta=0.05)
        self.assertAlmostEqual(res["tortuosity_median"], 1.12, delta=0.05)
        self.assertGreater(res["nn_distance_mean_m"], 200.0)
        self.assertAlmostEqual(res["nn_distance_median_m"], 170.3, delta=2.0)

    def test_04_pii_and_optical_nyquist(self):
        """Kiểm tra kiểm định quang học Nyquist và Rule of Three."""
        res = self.engine.audit_pii_and_optical_nyquist_limits(total_images=714123)
        self.assertTrue(res["is_sub_nyquist_proven"])
        self.assertEqual(res["detected_pii_count"], 0)
        self.assertLess(res["character_stroke_height_pixels"], 2.5)
        self.assertLess(res["rule_of_three_upper_bound_rate"], 1e-5)

    def test_05_reviewer_consistency_audit(self):
        """Kiểm tra Agent 2 phát hiện tính nhất quán nội bộ của bản thảo."""
        res = self.reviewer.audit_latex_internal_consistency()
        self.assertGreater(res["total_verified_facts"], 5)
        self.assertTrue(res["is_internally_consistent"])

    def test_06_reviewer_claims_critique(self):
        """Kiểm tra Agent 2 rà soát các giả định và luận điểm khoa học."""
        res = self.reviewer.audit_scientific_claims_and_methodology()
        self.assertIsInstance(res["critiques"], list)

    def test_07_drafter_plan_generation(self):
        """Kiểm tra Agent 1 phân tích feedback và tạo kế hoạch nâng cấp."""
        mock_gt = self.engine.compile_full_ground_truth_package()
        plan = self.drafter.process_reviewer_feedback_and_plan("Mock Feedback", mock_gt)
        self.assertEqual(plan["status"], "FEEDBACK_ACCEPTED")
        self.assertGreaterEqual(len(plan["action_plan"]), 4)

    def test_08_full_feedback_loop_orchestration(self):
        """Kiểm tra toàn bộ quy trình điều phối tương tác 2 tác tử (không biên dịch PDF để chạy nhanh)."""
        orchestrator = DualAgentOrchestrator(str(self.project_root))
        result = orchestrator.run_feedback_loop(compile_pdf=False)
        self.assertTrue(result["is_fully_approved"])
        self.assertTrue(os.path.exists(result["round1_report"]))
        self.assertTrue(os.path.exists(result["round2_report"]))
        self.assertTrue(os.path.exists(result["certificate"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
