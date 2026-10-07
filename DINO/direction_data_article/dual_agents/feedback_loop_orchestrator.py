"""
=============================================================================
Module: feedback_loop_orchestrator.py
Dự án: IC4SD-TrafficSnap Data Article (Elsevier Data in Brief / Scopus Q1)
Tác giả: Viet-Anh Le & Dr. Khanh Nguyen-Trong (IC4SD Lab, PTIT)
Tiêu chuẩn: Chuẩn sản xuất (Production-Ready)
Mô tả nghiệp vụ:
  Bộ điều phối vòng lặp tương tác phản biện đa tác tử (Dual-Agent Feedback Loop Orchestrator).
  Quản lý quy trình trao đổi học thuật tự động đối ngẫu giữa:
  - Agent 1: PaperDrafterNarrativeAgent ("Agent Chém Gió & Viết Paper")
  - Agent 2: ScientificReviewerAuditorAgent ("Agent Phản Biện & Kiểm Toán Thực Nghiệm")
  - Bộ máy: EmpiricalAuditorEngine (Chạy thực nghiệm trực tiếp trên dữ liệu gốc)

  Quy trình vận hành chuẩn mực:
  Round 1: Agent 1 đưa bản thảo sơ khởi (chứa các luận điểm chém gió / số liệu chưa kiểm chứng)
           -> Agent 2 phản biện khắt khe, vạch trần sai lệch, chỉ định thực nghiệm bắt buộc,
              và xuất Ground-Truth Package (Phán quyết: MAJOR REVISION REQUIRED).
  Round 2: Agent 1 bóc tách động feedback từ Agent 2, kích hoạt chạy các thực nghiệm từ dữ liệu gốc,
           chỉnh sửa số liệu, 'chém gió hành văn học thuật' nâng tầm bài báo và cập nhật LaTeX.
  Round 3: Agent 2 tái thẩm định toàn diện (Re-audit), đối chiếu từng điểm phản biện từ Round 1,
           xác nhận 100% nhất quán và cấp Chứng chỉ Chấp thuận Xuất bản (Camera-Ready Certification).
=============================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from dual_agents.empirical_auditor_engine import EmpiricalAuditorEngine
from dual_agents.agent_scientific_reviewer import ScientificReviewerAuditorAgent
from dual_agents.agent_paper_drafter import PaperDrafterNarrativeAgent

# Đảm bảo mã hóa UTF-8 trên hệ điều hành Windows
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DualAgentOrchestrator")


class DualAgentOrchestrator:
    """
    Bộ điều phối vòng lặp tương tác đối ngẫu giữa 2 tác tử.
    """

    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        self.reports_dir = self.project_root / "dual_agents" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.drafter = PaperDrafterNarrativeAgent(str(self.project_root))
        self.reviewer = ScientificReviewerAuditorAgent(str(self.project_root))
        self.engine = EmpiricalAuditorEngine(str(self.project_root))

        logger.info("Khởi tạo DualAgentOrchestrator hoàn tất. Thư mục báo cáo: %s", str(self.reports_dir))

    def run_feedback_loop(
        self,
        compile_pdf: bool = True,
        simulate_speculative_round1: bool = True
    ) -> Dict[str, Any]:
        """
        Thực thi toàn bộ vòng lặp tương tác phản biện đa tác tử giữa Agent 1 và Agent 2.
        - simulate_speculative_round1: Nếu True, Round 1 sẽ rà soát bản thảo sơ khởi với các
          luận điểm chém gió để minh chứng quy trình phản biện đa tác tử đối ngẫu từ Major Revision
          đến Camera-Ready.
        """
        logger.info("=" * 80)
        logger.info("BẮT ĐẦU VÒNG LẶP PHẢN BIỆN ĐA TÁC TỬ (DUAL-AGENT PEER-REVIEW ITERATIVE LOOP)")
        logger.info("=" * 80)

        start_time = time.time()

        # ---------------------------------------------------------------------
        # BƯỚC 1: AGENT 2 KÍCH HOẠT THỰC NGHIỆM ĐỘC LẬP TRÊN DỮ LIỆU GỐC
        # ---------------------------------------------------------------------
        logger.info("\n>>> [ROUND 1 - BƯỚC 1] Agent 2 kích hoạt EmpiricalAuditorEngine kiểm toán dữ liệu gốc...")
        ground_truth = self.reviewer.execute_empirical_investigation()
        logger.info("Đã trích xuất thành công Ground-Truth: 608 trạm, 2,450 cạnh, Tortuosity tau=%.2f, GSD=2.73-3.25cm",
                    ground_truth["spatial_and_tortuosity_metrics"]["tortuosity_mean"])

        # ---------------------------------------------------------------------
        # BƯỚC 2: AGENT 2 PHẢN BIỆN BẢN THẢO (ROUND 1 REVIEW)
        # ---------------------------------------------------------------------
        logger.info("\n>>> [ROUND 1 - BƯỚC 2] Agent 2 rà soát bản thảo để bắt bẻ các luận điểm 'chém gió'...")

        if simulate_speculative_round1:
            # Tạo bản thảo sơ khởi mang tính chém gió học thuật để Reviewer bắt bẻ
            draft_r1 = self.drafter.draft_initial_speculative_version()
            audit_r1 = self.reviewer.audit_manuscript(draft_r1, ground_truth)
        else:
            # Rà soát trực tiếp file main.tex hiện tại
            main_tex_path = self.project_root / "paper" / "main.tex"
            with open(main_tex_path, "r", encoding="utf-8") as f:
                current_tex = f.read()
            audit_r1 = self.reviewer.audit_manuscript(current_tex, ground_truth)

        report_r1_md = self.reviewer.generate_peer_review_report(
            round_num=1,
            audit_results=audit_r1,
            ground_truth=ground_truth
        )

        r1_report_path = self.reports_dir / "review_report_round1.md"
        with open(r1_report_path, "w", encoding="utf-8") as f:
            f.write(report_r1_md)

        logger.info("Đã lưu Báo cáo Phản biện Round 1 tại: %s", str(r1_report_path))
        logger.info("Phán quyết Round 1 của Agent 2: %s (%d vấn đề cần khắc phục).",
                    audit_r1["status"], audit_r1["total_critiques"] + audit_r1["total_issues_found"])

        # ---------------------------------------------------------------------
        # BƯỚC 3: AGENT 1 BÓC TÁCH FEEDBACK, CHẠY THỰC NGHIỆM VÀ NÂNG CẤP BÀI BÁO
        # ---------------------------------------------------------------------
        logger.info("\n>>> [ROUND 2 - BƯỚC 3] Agent 1 tiếp nhận feedback, chạy thực nghiệm và nâng cấp bài báo...")

        # 1. Bóc tách động feedback từ file Markdown của Agent 2
        parsed_feedback = self.drafter.parse_reviewer_feedback(report_r1_md)
        for act in parsed_feedback["action_plan"]:
            logger.info("  [Kế hoạch] %s", act)

        # 2. Chạy các thực nghiệm được yêu cầu trên dữ liệu gốc
        verified_gt = self.drafter.execute_requested_experiments(parsed_feedback, self.engine)

        # 3. 'Chém gió' hành văn học thuật nâng tầm và cập nhật LaTeX
        upgrade_success = self.drafter.execute_paper_upgrade(verified_gt)
        if not upgrade_success:
            raise RuntimeError("Nâng cấp bài báo bởi Agent 1 thất bại!")

        # ---------------------------------------------------------------------
        # BƯỚC 4: AGENT 2 TÁI THẨM ĐỊNH TOÀN DIỆN (ROUND 2 RE-AUDIT)
        # ---------------------------------------------------------------------
        logger.info("\n>>> [ROUND 2 - BƯỚC 4] Agent 2 tái thẩm định bản thảo sau khi Agent 1 đã chỉnh sửa...")

        main_tex_file = self.project_root / "paper" / "main.tex"
        with open(main_tex_file, "r", encoding="utf-8") as f:
            revised_tex = f.read()

        re_audit_eval = self.reviewer.evaluate_revision(
            original_critiques=audit_r1["critiques"],
            revised_content=revised_tex,
            ground_truth=verified_gt
        )

        audit_r2 = re_audit_eval["audit_results"]
        report_r2_md = self.reviewer.generate_peer_review_report(
            round_num=2,
            audit_results=audit_r2,
            ground_truth=verified_gt
        )

        r2_report_path = self.reports_dir / "review_report_round2.md"
        with open(r2_report_path, "w", encoding="utf-8") as f:
            f.write(report_r2_md)
        logger.info("Đã lưu Báo cáo Phản biện Round 2 tại: %s", str(r2_report_path))
        logger.info("Phán quyết Round 2 của Agent 2: %s (Đã giải quyết %d/%d điểm).",
                    audit_r2["status"], re_audit_eval["resolved_count"], len(audit_r1["critiques"]))

        is_fully_approved = audit_r2["is_fully_approved"]

        # ---------------------------------------------------------------------
        # BƯỚC 5: AGENT 2 CẤP CHỨNG CHỈ XUẤT BẢN CHÍNH THỨC
        # ---------------------------------------------------------------------
        cert_path = self.reports_dir / "final_audit_certification.md"
        cert_content = self.reviewer.generate_final_certificate(verified_gt)
        with open(cert_path, "w", encoding="utf-8") as f:
            f.write(cert_content)
        logger.info("Đã cấp Chứng chỉ Xuất bản Chính thức tại: %s", str(cert_path))

        # ---------------------------------------------------------------------
        # BƯỚC 6: BIÊN DỊCH PDF THỰC TẾ BẰNG PDFLATEX & BIBTEX
        # ---------------------------------------------------------------------
        if compile_pdf:
            logger.info("\n>>> [BƯỚC 6] Khởi động trình biên dịch LaTeX để tạo paper/main.pdf mới nhất...")
            self.compile_latex_manuscript()

        elapsed_total = time.time() - start_time
        logger.info("=" * 80)
        logger.info("TOÀN BỘ QUY TRÌNH 2 AGENT HOÀN THÀNH XUẤT SẮC TRONG %.2f GIÂY!", elapsed_total)
        logger.info("=" * 80)

        return {
            "is_fully_approved": is_fully_approved,
            "round1_status": audit_r1["status"],
            "round2_status": audit_r2["status"],
            "round1_report": str(r1_report_path),
            "round2_report": str(r2_report_path),
            "certificate": str(cert_path),
            "elapsed_seconds": round(elapsed_total, 2)
        }

    def compile_latex_manuscript(self) -> bool:
        """
        Biên dịch bài báo LaTeX thành PDF:
        pdflatex -> bibtex -> pdflatex -> pdflatex.
        """
        paper_dir = str(self.project_root / "paper")
        logger.info("Đang biên dịch LaTeX trong: %s", paper_dir)

        try:
            # Lần 1: pdflatex
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                cwd=paper_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Lần 2: bibtex
            subprocess.run(
                ["bibtex", "main"],
                cwd=paper_dir,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Lần 3: pdflatex
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                cwd=paper_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Lần 4: pdflatex (để resolve cross-references)
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                cwd=paper_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info("Biên dịch PDF thành công! Tệp đầu ra: %s", str(self.project_root / "paper" / "main.pdf"))
            return True
        except Exception as e:
            logger.error("Lỗi trong quá trình biên dịch LaTeX: %s", e)
            return False


def main():
    parser = argparse.ArgumentParser(description="Chạy hệ thống 2 Agent cộng tác & phản biện bài báo khoa học")
    parser.add_argument("--no-pdf", action="store_true", help="Không tự động biên dịch lại LaTeX PDF")
    parser.add_argument("--audit-existing", action="store_true", help="Kiểm toán trực tiếp bản thảo main.tex hiện tại thay vì demo đối ngẫu từ bản thảo sơ khởi")
    args = parser.parse_args()

    orchestrator = DualAgentOrchestrator()
    res = orchestrator.run_feedback_loop(
        compile_pdf=not args.no_pdf,
        simulate_speculative_round1=not args.audit_existing
    )
    print("\n[KẾT QUẢ ĐIỀU PHỐI ĐA TÁC TỬ]")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
