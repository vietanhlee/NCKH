"""
=============================================================================
KỊCH BẢN ĐIỀU KHIỂN TẬP TRUNG TOÀN BỘ QUY TRÌNH TRÍCH XUẤT SỐ LIỆU THỰC TẾ
HCMC-TrafficSnap: Urban Traffic Camera Image Time-Series Dataset
Tác giả: Viet-Anh Le & Dr. Khanh Nguyen-Trong (PTIT)
Chuẩn sản xuất (Production-Ready) tuân thủ tiêu chuẩn Data in Brief (Elsevier)
=============================================================================
Hướng dẫn sử dụng:
  Chạy toàn bộ quy trình chỉ bằng 1 câu lệnh duy nhất:
    python run_all_metrics.py --image-dir <ĐƯỜNG_DẪN_KHO_ẢNH_THẬT>
  
  Hoặc chạy kiểm thử nhanh (Dry-run / Sample Mode) với tập ảnh mẫu có sẵn:
    python run_all_metrics.py --sample-mode
=============================================================================
"""

import os
import sys

# Đảm bảo mã hóa UTF-8 cho console Windows
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import logging
import argparse
import subprocess
from pathlib import Path

# Đảm bảo tương thích môi trường OpenMP/MKL và UTF-8 trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PipelineRunner")


def run_command_step(script_name: str, args_list: list) -> bool:
    """
    Thực thi 1 bước trong pipeline thông qua subprocess, theo dõi mã thoát lỗi.
    """
    cmd = [sys.executable, script_name] + args_list
    logger.info("=" * 70)
    logger.info("ĐANG CHẠY BƯỚC: %s", script_name)
    logger.info("Lệnh thực thi: %s", " ".join(cmd))
    logger.info("=" * 70)

    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), env=env)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        logger.error("BƯỚC %s THẤT BẠI với mã thoát: %d", script_name, result.returncode)
        return False

    logger.info("BƯỚC %s HOÀN THÀNH XUẤT SẮC trong %.2f giây.\n", script_name, elapsed)
    return True


def main():
    parser = argparse.ArgumentParser(description="Chạy tự động toàn bộ pipeline trích xuất số liệu HCMC-TrafficSnap")
    parser.add_argument("--image-dir", type=str, default="", help="Đường dẫn đến kho ảnh camera thực tế")
    parser.add_argument("--sample-mode", action="store_true", help="Chạy ở chế độ mẫu (sử dụng zenodo_bundle/sample_preview)")
    parser.add_argument("--sample-size", type=int, default=10000, help="Số lượng ảnh lấy mẫu phân tầng")
    parser.add_argument("--max-workers", type=int, default=4, help="Số tiến trình CPU xử lý song song")

    args = parser.parse_args()

    # Xác định thư mục ảnh đầu vào
    base_dir = Path(__file__).resolve().parent
    sample_preview_dir = base_dir.parent / "zenodo_bundle" / "sample_preview"
    metadata_dir = base_dir.parent / "zenodo_bundle" / "metadata"
    metadata_stations = metadata_dir / "stations_metadata.csv"
    if not metadata_stations.exists():
        metadata_stations = metadata_dir / "routes.csv"
    metadata_distance = metadata_dir / "road_network_distance.csv"
    output_dir = base_dir / "output"
    paper_tables_dir = base_dir.parent / "paper" / "tables"
    paper_figures_dir = base_dir.parent / "paper" / "figures"

    if args.sample_mode or not args.image_dir or not os.path.exists(args.image_dir):
        logger.info("Kích hoạt chế độ kiểm thử mẫu với thư mục: %s", sample_preview_dir)
        target_image_dir = str(sample_preview_dir)
    else:
        target_image_dir = args.image_dir
        logger.info("Sử dụng kho ảnh thực tế chỉ định: %s", target_image_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(paper_tables_dir, exist_ok=True)
    os.makedirs(paper_figures_dir, exist_ok=True)

    pipeline_start = time.time()

    # BƯỚC 1: Trích xuất thông số kho ảnh (02_extract_image_dataset_stats.py)
    step1_ok = run_command_step(
        "02_extract_image_dataset_stats.py",
        [
            "--image-dir", target_image_dir,
            "--output-dir", str(output_dir),
            "--max-workers", str(args.max_workers)
        ]
    )
    if not step1_ok:
        logger.error("Dừng pipeline do lỗi ở Bước 2.")
        sys.exit(1)

    # BƯỚC 2: Trích xuất trắc quang và chất lượng ảnh (03_extract_photometric_and_quality.py)
    step2_ok = run_command_step(
        "03_extract_photometric_and_quality.py",
        [
            "--input-dir", target_image_dir,
            "--output-dir", str(output_dir),
            "--sample-size", str(args.sample_size),
            "--max-workers", str(args.max_workers)
        ]
    )
    if not step2_ok:
        logger.error("Dừng pipeline do lỗi ở Bước 3.")
        sys.exit(1)

    # BƯỚC 3: Trích xuất thông số Topo Đồ thị OSM (04_extract_osm_graph_metrics.py)
    step3_ok = run_command_step(
        "04_extract_osm_graph_metrics.py",
        [
            "--stations-csv", str(metadata_stations),
            "--distance-csv", str(metadata_distance),
            "--output-dir", str(output_dir),
            "--cutoff-meters", "5000.0"
        ]
    )
    if not step3_ok:
        logger.error("Dừng pipeline do lỗi ở Bước 4.")
        sys.exit(1)

    # BƯỚC 4: Kiểm định định lượng PII (05_extract_pii_audit_metrics.py)
    step4_ok = run_command_step(
        "05_extract_pii_audit_metrics.py",
        [
            "--input-dir", target_image_dir,
            "--output-dir", str(output_dir),
            "--audit-samples", "2000"
        ]
    )
    if not step4_ok:
        logger.error("Dừng pipeline do lỗi ở Bước 5.")
        sys.exit(1)

    # BƯỚC 5: Sinh bảng LaTeX và hình ảnh 300 DPI (06_generate_latex_tables_and_figures.py)
    step5_ok = run_command_step(
        "06_generate_latex_tables_and_figures.py",
        [
            "--output-dir", str(output_dir),
            "--paper-tables-dir", str(paper_tables_dir),
            "--paper-figures-dir", str(paper_figures_dir),
            "--stations-csv", str(metadata_stations),
            "--sample-preview-dir", str(sample_preview_dir)
        ]
    )
    if not step5_ok:
        logger.error("Dừng pipeline do lỗi ở Bước 6.")
        sys.exit(1)

    total_pipeline_time = time.time() - pipeline_start
    logger.info("=" * 70)
    logger.info("TỔNG KẾT: TOÀN BỘ QUY TRÌNH ĐÃ THỰC THI THÀNH CÔNG RỰC RỠ!")
    logger.info("Tổng thời gian thực thi: %.2f giây.", total_pipeline_time)
    logger.info("Các file kết quả JSON tại: %s", str(output_dir))
    logger.info("Các bảng LaTeX tại: %s", str(paper_tables_dir))
    logger.info("Các hình vẽ 300 DPI tại: %s", str(paper_figures_dir))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
