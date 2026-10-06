#!/usr/bin/env python3
"""
01_crawl_or_sample_collector.py
===============================
Mục đích: Bộ thu thập dữ liệu ảnh camera giao thông thực tế từ Cổng Thông tin
          Giao thông TP.HCM (giaothong.hochiminhcity.gov.vn).

Tính năng production:
  1. Hỗ trợ đa luồng (ThreadPoolExecutor) với persistent HTTP Keep-Alive session.
  2. Đo đạc độ trễ mạng và độ lệch thời gian thực tế của camera (Delta t_lag).
  3. Ghi file nguyên tử (Atomic write: ghi file .tmp rồi os.replace) chống hỏng tệp khi mất nguồn.
  4. Tự động nhận diện và loại bỏ ảnh lỗi placeholder của cổng giao thông (284x177 px).
  5. Xuất nhật ký thu thập (crawl_latency_log.csv) phục vụ báo cáo độ trễ thực tế.

Cách sử dụng:
  python 01_crawl_or_sample_collector.py --routes_csv ../zenodo_bundle/metadata/routes.csv \
                                         --output_dir /path/to/raw_images \
                                         --cycles 1 --interval_seconds 300
"""

import os
import sys
import time
import argparse
import csv
import io
import urllib3
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# Tắt cảnh báo SSL không an toàn (nếu có chứng chỉ cổng công cộng tự ký)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Múi giờ chuẩn Việt Nam (ICT = UTC+7)
VN_TZ = timezone(timedelta(hours=7))


def load_camera_endpoints(camera_csv_path: str) -> List[Dict[str, Any]]:
    """Đọc danh sách các trạm camera từ file camera_data_608Cam.csv (hoặc routes.csv)."""
    if not os.path.exists(camera_csv_path):
        # Fallback to local or bundle alternatives
        parent_dir = os.path.dirname(camera_csv_path)
        for alt in ["camera_data_608Cam.csv", "../camera_data_608Cam.csv", "../zenodo_bundle/metadata/camera_data_608Cam.csv"]:
            candidate = os.path.normpath(os.path.join(parent_dir, alt))
            if os.path.exists(candidate):
                camera_csv_path = candidate
                break

    if not os.path.exists(camera_csv_path):
        raise FileNotFoundError(f"Không tìm thấy file danh mục camera tại: {camera_csv_path}")

    cameras = []
    with open(camera_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Chấp nhận cả stt hoặc station_id
                stt_val = row.get("stt") or row.get("station_id")
                if not stt_val:
                    continue
                station_id = int(stt_val)
                name = row.get("Location") or row.get("station_name") or f"Station {station_id}"
                cam_id = row.get("CamID", "")
                lat = float(row.get("latitude", 0.0))
                lng = float(row.get("longitude", 0.0))
                elev = float(row.get("camera_elevation_m", 8.0))

                cameras.append({
                    "station_id": station_id,
                    "name": name,
                    "cam_id": cam_id,
                    "lat": lat,
                    "lng": lng,
                    "elevation": elev
                })
            except (ValueError, KeyError):
                continue

    print(f"[Collector] Đã nạp thành công {len(cameras)} trạm camera hợp lệ từ {os.path.basename(camera_csv_path)}.")
    return cameras


def fetch_and_save_snapshot(
    session: requests.Session,
    camera: Dict[str, Any],
    output_dir: str,
    timeout: int = 12
) -> Dict[str, Any]:
    """
    Thu thập 1 ảnh từ cổng giao thông, đo độ trễ và lưu trữ nguyên tử.
    """
    station_id = camera["station_id"]
    # Endpoint chính thức của Cổng Giao thông TP.HCM
    url = f"https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={station_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Connection": "keep-alive",
    }

    start_time = time.time()
    unix_ts = int(start_time)
    record = {
        "station_id": station_id,
        "unix_timestamp": unix_ts,
        "local_datetime": datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "latency_sec": 0.0,
        "status_code": 0,
        "file_size_bytes": 0,
        "width": 0,
        "height": 0,
        "is_valid": False,
        "discard_reason": ""
    }

    try:
        response = session.get(url, headers=headers, timeout=timeout, verify=False)
        record["status_code"] = response.status_code
        elapsed = time.time() - start_time
        record["latency_sec"] = round(elapsed, 3)

        if response.status_code == 200 and response.content:
            data = response.content
            size_bytes = len(data)
            record["file_size_bytes"] = size_bytes

            # Kiểm tra định dạng ảnh bằng PIL
            try:
                with Image.open(io.BytesIO(data)) as img:
                    w, h = img.size
                    record["width"] = w
                    record["height"] = h

                    # 1. Phát hiện thẻ lỗi placeholder (kích thước 284x177 px)
                    if w == 284 and h == 177:
                        record["discard_reason"] = "municipal_error_card_284x177"
                        return record

                    # 2. Phát hiện kích thước quá nhỏ hoặc hỏng
                    if w < 200 or h < 150:
                        record["discard_reason"] = "sub_resolution_degraded"
                        return record

            except Exception as e:
                record["discard_reason"] = f"invalid_image_stream_{e}"
                return record

            # Ghi file nguyên tử (Atomic serialization)
            final_filename = f"{station_id}_{unix_ts}.jpg"
            final_path = os.path.join(output_dir, final_filename)
            tmp_path = os.path.join(output_dir, f".tmp_{final_filename}")

            with open(tmp_path, "wb") as f_out:
                f_out.write(data)

            # Atomic rename (thay thế an toàn trên hệ điều hành)
            os.replace(tmp_path, final_path)
            record["is_valid"] = True
        else:
            record["discard_reason"] = f"http_{response.status_code}"

    except requests.exceptions.Timeout:
        record["discard_reason"] = "http_timeout"
    except Exception as exc:
        record["discard_reason"] = f"network_error_{exc}"

    return record


def run_collection_cycle(
    cameras: List[Dict[str, Any]],
    output_dir: str,
    max_workers: int = 32,
    log_csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """Thực hiện một chu kỳ cào cho toàn bộ danh sách camera."""
    os.makedirs(output_dir, exist_ok=True)
    cycle_start = time.time()
    print(f"\n[Cycle Start] Bắt đầu thu thập {len(cameras)} camera lúc {datetime.now(VN_TZ).strftime('%H:%M:%S')}...")

    results = []
    # Sử dụng Session với Connection Pooling
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers, max_retries=2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_save_snapshot, session, cam, output_dir): cam for cam in cameras}
        for future in as_completed(futures):
            results.append(future.result())

    session.close()

    total = len(results)
    valid_count = sum(1 for r in results if r["is_valid"])
    discard_cards = sum(1 for r in results if r["discard_reason"] == "municipal_error_card_284x177")
    avg_latency = sum(r["latency_sec"] for r in results) / total if total > 0 else 0.0

    print(f"[Cycle End] Thu thập xong: {valid_count}/{total} ảnh hợp lệ "
          f"({valid_count/total*100:.1f}%). Thẻ lỗi loại bỏ: {discard_cards}. "
          f"Độ trễ TB: {avg_latency:.2f}s. Thời gian chu kỳ: {time.time()-cycle_start:.1f}s.")

    # Ghi log nếu yêu cầu
    if log_csv_path:
        file_exists = os.path.exists(log_csv_path)
        with open(log_csv_path, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)

    return {
        "total": total,
        "valid": valid_count,
        "discarded_error_cards": discard_cards,
        "avg_latency_sec": avg_latency,
        "cycle_duration_sec": time.time() - cycle_start
    }


def main():
    parser = argparse.ArgumentParser(description="Bộ thu thập ảnh camera thực tế HCMC-TrafficSnap.")
    parser.add_argument("--routes_csv", "--camera_csv", type=str, default="../zenodo_bundle/metadata/routes.csv",
                        help="Đường dẫn đến file routes.csv chứa 608 trạm.")
    parser.add_argument("--output_dir", type=str, default="data/raw_snapshots",
                        help="Thư mục lưu trữ ảnh cào được.")
    parser.add_argument("--log_csv", type=str, default="data/crawl_latency_log.csv",
                        help="Tệp CSV ghi nhật ký độ trễ và trạng thái.")
    parser.add_argument("--cycles", type=int, default=1,
                        help="Số chu kỳ chạy (1 để test mẫu, hoặc nhiều chu kỳ để chạy liên tục).")
    parser.add_argument("--interval_seconds", type=int, default=300,
                        help="Khoảng cách giữa các chu kỳ (mặc định: 300s = 5 phút).")
    parser.add_argument("--workers", type=int, default=24,
                        help="Số luồng đồng thời (mặc định: 24).")

    args = parser.parse_args()
    cameras = load_camera_endpoints(args.camera_csv)

    os.makedirs(os.path.dirname(os.path.abspath(args.log_csv)), exist_ok=True)

    for cycle in range(1, args.cycles + 1):
        print(f"\n=== CHU KỲ {cycle}/{args.cycles} ===")
        summary = run_collection_cycle(cameras, args.output_dir, max_workers=args.workers, log_csv_path=args.log_csv)

        if cycle < args.cycles:
            sleep_time = max(0, args.interval_seconds - summary["cycle_duration_sec"])
            print(f"[Timer] Nghỉ {sleep_time:.1f} giây trước chu kỳ tiếp theo...")
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
