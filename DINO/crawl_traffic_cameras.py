"""
=============================================================================
 Production Traffic Camera Crawler (IC4SD-Traffic-HCM)
 Thu thập dữ liệu hình ảnh thời gian thực từ 608 Camera Giao thông TP.HCM
 Nguồn: https://giaothong.hochiminhcity.gov.vn
=============================================================================
Tính năng chuẩn Production:
  - Connection Pooling & Keep-Alive với requests.Session + urllib3 Retry.
  - Lọc trực tiếp ảnh rác / placeholder "IMAGE NOT AVAILABLE" (284x177, 6-6.8 KB).
  - Atomic File Write (ghi tạm rồi đổi tên) chống file hỏng do ngắt mạng.
  - Drift-free Timer: Tự động tính bù thời gian quét để chu kỳ chạy chuẩn xác.
  - Logging chi tiết theo múi giờ Việt Nam (Asia/Ho_Chi_Minh, UTC+7).
  - Hỗ trợ CLI argparse linh hoạt (chạy đơn kỳ hoặc vòng lặp 24/7).
=============================================================================
"""

import os
import sys
import time
import csv
import argparse
from datetime import datetime, timezone, timedelta
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import numpy as np
import cv2

# Cấu hình UTF-8 cho terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Múi giờ chuẩn Việt Nam (UTC+7)
VN_TZ = timezone(timedelta(hours=7))


def create_resilient_session(pool_size: int = 50, max_retries: int = 3) -> requests.Session:
    """
    Tạo session HTTP có connection pooling và cơ chế tự động thử lại khi lỗi mạng.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=retry_strategy,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Connection": "keep-alive",
    })
    return session


def load_camera_list(csv_path: str) -> List[Dict[str, str]]:
    """
    Đọc danh sách camera từ file CSV.
    Cấu trúc mong đợi: [stt, name, cam_id, ...]
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file danh sách camera tại: {csv_path}")

    cameras = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row_idx, row in enumerate(reader, start=2):
            if len(row) >= 3:
                stt = row[0].strip()
                cam_id = row[2].strip()
                if stt and cam_id:
                    cameras.append({"stt": stt, "cam_id": cam_id})
    return cameras


def is_corrupted_or_placeholder(content: bytes, img_cv: Optional[np.ndarray]) -> bool:
    """
    Kiểm tra xem ảnh tải về có phải là ảnh placeholder 'IMAGE NOT AVAILABLE'
    hoặc ảnh hỏng/rỗng hay không:
      - Dung lượng [6.0, 6.8] KB và độ phân giải đúng 284x177.
      - Hoặc ảnh không thể decode (img_cv is None).
      - Hoặc kích thước ảnh quá bé (H < 50 hoặc W < 50).
    """
    size_kb = len(content) / 1024.0
    if img_cv is None:
        return True

    h, w = img_cv.shape[:2]
    # Lọc ảnh thông báo lỗi đặc thù của cổng giao thông TP.HCM
    if 5.8 <= size_kb <= 7.0 and w == 284 and h == 177:
        return True

    if h < 50 or w < 50:
        return True

    return False


def fetch_and_save_camera(
    cam: Dict[str, str],
    output_dir: str,
    session: requests.Session,
    timeout: float = 12.0,
) -> Tuple[str, bool, str]:
    """
    Tải và lưu ảnh cho một camera với cơ chế Atomic Write an toàn.

    Returns:
        (cam_stt, success, message)
    """
    stt = cam["stt"]
    cam_id = cam["cam_id"]
    image_url = f"https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={cam_id}"

    timestamp = int(time.time())
    now_vn = datetime.now(VN_TZ)
    time_str = now_vn.strftime("%d/%m/%Y %H:%M:%S")

    try:
        response = session.get(image_url, timeout=timeout)
        if response.status_code != 200:
            return stt, False, f"HTTP {response.status_code}"

        content = response.content
        if not content:
            return stt, False, "Dữ liệu trả về rỗng"

        # Decode ảnh trong bộ nhớ RAM
        img_np = np.frombuffer(content, dtype=np.uint8)
        img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Kiểm tra ảnh rác / placeholder
        if is_corrupted_or_placeholder(content, img_cv):
            return stt, False, "Ảnh placeholder lỗi (284x177)"

        # Tên file chuẩn: {stt}_{timestamp}.jpg
        final_filename = f"{stt}_{timestamp}.jpg"
        final_path = os.path.join(output_dir, final_filename)
        temp_path = os.path.join(output_dir, f".{final_filename}.tmp")

        # Atomic write: Ghi ra file tạm rồi đổi tên để tránh file bị đứt gãy nếu mất điện / crash
        cv2.imwrite(temp_path, img_cv)
        os.replace(temp_path, final_path)

        return stt, True, f"[{time_str}] Đã lưu {final_filename} ({img_cv.shape[1]}x{img_cv.shape[0]})"

    except Exception as e:
        return stt, False, f"Lỗi exception: {str(e)}"


def run_crawl_cycle(
    cameras: List[Dict[str, str]],
    output_dir: str,
    workers: int = 40,
    timeout: float = 12.0,
) -> Dict[str, int]:
    """
    Thực hiện 1 chu kỳ quét toàn bộ danh sách camera.
    """
    os.makedirs(output_dir, exist_ok=True)
    session = create_resilient_session(pool_size=workers)

    success_count = 0
    fail_count = 0
    placeholder_count = 0

    cycle_start = time.time()
    now_start = datetime.now(VN_TZ).strftime("%H:%M:%S")
    print(f"🚀 [{now_start}] Bắt đầu chu kỳ quét {len(cameras)} camera (Workers={workers})...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_and_save_camera, cam, output_dir, session, timeout): cam
            for cam in cameras
        }

        for future in as_completed(futures):
            stt, ok, msg = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                if "placeholder" in msg:
                    placeholder_count += 1

    elapsed = time.time() - cycle_start
    print(f"✅ Hoàn tất chu kỳ: {success_count}/{len(cameras)} thành công, "
          f"{fail_count} thất bại ({placeholder_count} placeholder) trong {elapsed:.1f}s.")

    return {
        "total": len(cameras),
        "success": success_count,
        "failed": fail_count,
        "placeholder": placeholder_count,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Production Traffic Camera Crawler (TP.HCM)")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=r"D:\DATN_transport-network-model\camera_data_608Cam.csv",
        help="Đường dẫn file CSV chứa danh sách camera (cột: stt, name, cam_id)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Thư mục lưu trữ ảnh origin đầu ra",
    )
    parser.add_argument(
        "--interval_sec",
        type=int,
        default=240,
        help="Khoảng cách giữa các chu kỳ quét (mặc định 240 giây = 4 phút)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=40,
        help="Số luồng tải ảnh song song",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="Timeout cho mỗi HTTP request (giây)",
    )
    parser.add_argument(
        "--single_run",
        action="store_true",
        help="Chỉ chạy đúng 1 chu kỳ rồi thoát (dành cho Cron job / Testing)",
    )
    args = parser.parse_args()

    print("=================================================================")
    print(" 📹 CRAWLER HÌNH ẢNH CAMERA GIAO THÔNG TP.HCM - PRODUCTION")
    print("=================================================================")
    print(f"📁 CSV Camera    : {args.csv_path}")
    print(f"📁 Thư mục lưu   : {args.output_dir}")
    print(f"⏱️  Chu kỳ quét   : {args.interval_sec}s | Workers: {args.workers}")
    print(f"🌏 Múi giờ log   : UTC+7 (Asia/Ho_Chi_Minh)")
    print("=================================================================")

    if not os.path.isfile(args.csv_path):
        print(f"⚠️  Không tìm thấy file: {args.csv_path}")
        print("💡 Đang kiểm tra đường dẫn thay thế trong thư mục dự án...")
        fallback_csv = os.path.join(os.path.dirname(__file__), "..", "camera_data_608Cam.csv")
        if os.path.isfile(fallback_csv):
            args.csv_path = os.path.abspath(fallback_csv)
            print(f"-> Sử dụng file thay thế: {args.csv_path}")
        else:
            print("❌ Không có file danh sách camera. Hãy chỉ định qua tham số --csv_path.")
            sys.exit(1)

    camera_list = load_camera_list(args.csv_path)
    print(f"[*] Đã tải danh sách {len(camera_list)} camera hợp lệ.")

    cycle_idx = 1
    while True:
        cycle_start_time = time.time()
        print(f"\n--- [CHU KỲ #{cycle_idx}] ---")
        run_crawl_cycle(camera_list, args.output_dir, workers=args.workers, timeout=args.timeout)

        if args.single_run:
            print("[*] Đã hoàn thành chế độ single_run. Kết thúc.")
            break

        # Drift-free sleep: Trừ hao thời gian đã dùng để quét camera
        elapsed = time.time() - cycle_start_time
        sleep_needed = max(0.0, float(args.interval_sec) - elapsed)
        next_time = datetime.now(VN_TZ) + timedelta(seconds=sleep_needed)
        print(f"💤 Nghỉ {sleep_needed:.1f}s. Chu kỳ tiếp theo bắt đầu lúc: {next_time.strftime('%H:%M:%S')}")
        time.sleep(sleep_needed)
        cycle_idx += 1


if __name__ == "__main__":
    main()
