"""
=============================================================================
 Production Background Generator (IC4SD-Traffic-HCM)
 Tự động tạo ảnh nền tĩnh (Background) từ tập ảnh camera giao thông
 Phương pháp: Lọc ảnh rác -> Gom nhóm (Tuyến x Khung giờ) -> Median Filter
=============================================================================
Cải tiến chuẩn Production:
  - Tự động nhận diện múi giờ Việt Nam (UTC+7) chuẩn chỉ trên mọi hệ điều hành.
  - Phân biệt an toàn tuyệt đối giữa Unix Timestamp (10/13 số) và Chuỗi ngày giờ (14 số).
  - Loại bỏ hoàn toàn nguy cơ văng lỗi OverflowError / OSError trên Windows & Linux.
  - Xóa ảnh lỗi và ảnh placeholder 'IMAGE NOT AVAILABLE' (284x177) đa tiến trình.
  - Chuẩn hóa ánh sáng (Illumination Normalization qua không gian màu HSV kênh V).
  - Tự động nhóm theo slot giờ linh hoạt (1h, 2h, 3h...) và tính toán Median đa lõi CPU.
=============================================================================
"""

import os
import sys
import glob
import re
import argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Tuple, List, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

# Cấu hình UTF-8 cho terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Khai báo tường minh múi giờ Việt Nam (UTC+7)
TZ_VN = timezone(timedelta(hours=7))


def check_and_delete_corrupted(file_path: str) -> bool:
    """
    Kiểm tra và xoá trực tiếp ảnh placeholder 'IMAGE NOT AVAILABLE' hoặc ảnh hỏng:
      - Dung lượng [5.8, 7.0] KB và độ phân giải đúng 284x177 -> Xoá.
      - Nếu ảnh bị hỏng không đọc được bằng cv2.imread() -> Xoá.
      - Nếu kích thước ảnh quá bé (H < 50 hoặc W < 50) -> Xoá.

    Returns:
        True nếu file bị xoá, False nếu file hợp lệ.
    """
    try:
        file_size_bytes = os.path.getsize(file_path)
        file_size_kb = file_size_bytes / 1024.0

        if 5.8 <= file_size_kb <= 7.0:
            img = cv2.imread(file_path)
            if img is None:
                os.remove(file_path)
                return True

            h, w = img.shape[:2]
            if w == 284 and h == 177:
                os.remove(file_path)
                return True

        # Kiểm tra đọc ảnh hợp lệ
        img = cv2.imread(file_path)
        if img is None:
            os.remove(file_path)
            return True

        h, w = img.shape[:2]
        if h < 50 or w < 50:
            os.remove(file_path)
            return True

        return False
    except Exception:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass
        return True


def parse_image_info(filename: str) -> Tuple[str, Optional[datetime]]:
    """
    Phân tích tên file: <route_id>_<timestamp_or_datetime>.<ext>
    Đồng bộ chuẩn 100% với crawler và TrafficPairMatcher:
      1. Chuỗi ngày giờ có ký tự phân tách: %Y%m%d_%H%M%S, %Y-%m-%d_%H-%M-%S...
      2. Chuỗi ngày giờ 14 chữ số: %Y%m%d%H%M%S (tránh bị nhầm thành unix timestamp)
      3. Chuỗi ngày giờ 12 chữ số: %Y%m%d%H%M
      4. Unix timestamp mili-giây: 13 chữ số (chia 1000 và quy đổi UTC+7)
      5. Unix timestamp giây: 9 đến 11 chữ số (quy đổi UTC+7)

    Returns:
        (route_id, datetime_obj_or_None)
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split("_", 1)
    if len(parts) < 2:
        # Fallback: Chỉ có ID camera ở đầu
        m = re.match(r"^(\d+)", basename)
        return m.group(1) if m else "unknown_route", None

    raw_route = parts[0]
    ts_str = parts[1].strip()

    # Chuẩn hóa route_id (ví dụ: route_1 -> 1, 01 -> 1)
    m_r = re.search(r"(\d+)", raw_route)
    route_id = str(int(m_r.group(1))) if m_r else raw_route

    # 1. ƯU TIÊN: Chuỗi ngày giờ có ký tự phân tách (_, -, :, khoảng trắng)
    if any(sep in ts_str for sep in ["-", "_", ":", " "]):
        date_formats_sep = [
            "%Y%m%d_%H%M%S",
            "%Y-%m-%d_%H-%M-%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y%m%d_%H%M",
        ]
        for fmt in date_formats_sep:
            try:
                # Chuỗi ngày giờ camera vốn đã là giờ địa phương Việt Nam
                return route_id, datetime.strptime(ts_str, fmt)
            except ValueError:
                pass

    # 2. XỬ LÝ CHUỖI THUẦN SỐ (isdigit) PHÂN BIỆT THEO ĐỘ DÀI
    if ts_str.isdigit():
        length = len(ts_str)

        # 2.1. Chuỗi ngày giờ YYYYMMDDHHMMSS (14 chữ số, ví dụ 20240510143000)
        if length == 14:
            try:
                return route_id, datetime.strptime(ts_str, "%Y%m%d%H%M%S")
            except ValueError:
                pass

        # 2.2. Chuỗi ngày giờ YYYYMMDDHHMM (12 chữ số)
        elif length == 12:
            try:
                return route_id, datetime.strptime(ts_str, "%Y%m%d%H%M")
            except ValueError:
                pass

        # 2.3. Unix Timestamp mili-giây (13 chữ số, ví dụ 1715324400000)
        elif length == 13:
            try:
                ts_sec = int(ts_str) / 1000.0
                return route_id, datetime.fromtimestamp(ts_sec, tz=TZ_VN)
            except (ValueError, OSError, OverflowError):
                pass

        # 2.4. Unix Timestamp giây (chuẩn 10 chữ số, từ 9 đến 11 chữ số)
        elif 9 <= length <= 11:
            try:
                ts_sec = float(ts_str)
                return route_id, datetime.fromtimestamp(ts_sec, tz=TZ_VN)
            except (ValueError, OSError, OverflowError):
                pass

    # 3. FALLBACK: Tìm chuỗi số timestamp bất kỳ còn lại
    m_ts = re.search(r"(\d{10,13})", ts_str)
    if m_ts:
        try:
            val = int(m_ts.group(1))
            raw_sec = val / 1000.0 if len(m_ts.group(1)) >= 13 else float(val)
            return route_id, datetime.fromtimestamp(raw_sec, tz=TZ_VN)
        except (ValueError, OSError, OverflowError):
            pass

    return route_id, None


def normalize_illumination(img: np.ndarray, target_mean_v: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Chuẩn hóa ánh sáng dựa trên kênh V (Value) trong không gian màu HSV.
    Giúp các frame thu nhận trong cùng 1 slot giờ có mức sáng tương đồng
    trước khi lấy median, hạn chế vệt sáng mờ do đèn xe hoặc mây trôi.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    mean_v = float(np.mean(hsv[:, :, 2]))

    if target_mean_v is not None and mean_v > 1e-3:
        scale = target_mean_v / mean_v
        scale = float(np.clip(scale, 0.7, 1.3))
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), mean_v


def process_single_slot(args: Tuple) -> Tuple[str, Union[int, str], bool, str]:
    """
    Xử lý tính toán background cho một cặp (route_id, slot_id).
    Phương pháp: Lấy mẫu đều -> Chuẩn hóa sáng HSV -> Lấy Median theo trục thời gian.
    """
    route_id, slot, paths, output_dir, max_samples = args

    if len(paths) < 3:
        return route_id, slot, False, f"Tuyến {route_id} - Slot {slot}h: Chỉ có {len(paths)} ảnh, không đủ điều kiện (cần >= 3)"

    # Lấy mẫu đại diện nếu số lượng ảnh quá lớn
    if len(paths) > max_samples:
        indices = np.linspace(0, len(paths) - 1, max_samples, dtype=int)
        sampled_paths = [paths[i] for i in indices]
    else:
        sampled_paths = paths

    frames = []
    v_means = []
    temp_imgs = []
    target_shape = None

    for p in sampled_paths:
        img = cv2.imread(p)
        if img is None:
            continue

        h, w = img.shape[:2]
        if target_shape is None:
            target_shape = (h, w)
        elif (h, w) != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

        temp_imgs.append(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_means.append(float(np.mean(hsv[:, :, 2])))

    if len(temp_imgs) < 3:
        return route_id, slot, False, f"Tuyến {route_id} - Slot {slot}h: Không đủ ảnh hợp lệ sau khi đọc"

    reference_mean_v = float(np.median(v_means))

    # Chuẩn hóa ánh sáng từng frame theo median reference
    for img in temp_imgs:
        norm_img, _ = normalize_illumination(img, target_mean_v=reference_mean_v)
        frames.append(norm_img)

    # Tính toán ảnh nền tĩnh bằng Temporal Median Filter
    stack = np.stack(frames, axis=0)
    background = np.median(stack, axis=0).astype(np.uint8)

    # Tạo thư mục cho từng tuyến: traffic_backgrounds/route_{route_id}/
    route_dir = os.path.join(output_dir, f"route_{route_id}")
    os.makedirs(route_dir, exist_ok=True)

    out_name = f"background_slot_{slot:02d}h.jpg" if isinstance(slot, int) else "background_unknown.jpg"
    out_path = os.path.join(route_dir, out_name)
    cv2.imwrite(out_path, background)

    return route_id, slot, True, f"Tuyến {route_id} - Slot {slot}h -> {out_name} ({len(frames)} frames)"


def process_traffic_dataset(
    input_dir: str,
    output_dir: str,
    time_interval_hours: int = 1,
    max_samples_per_slot: int = 30,
    max_workers: Optional[int] = None,
):
    """
    Quy trình tiền xử lý và sinh Background tự động:
      Bước 1: Quét và xoá vĩnh viễn ảnh lỗi/placeholder song song.
      Bước 2: Phân tích timestamp múi giờ UTC+7 và gom nhóm theo (tuyến, slot giờ).
      Bước 3: Tính toán Median Background đa tiến trình và lưu vào cấu trúc chuẩn.
    """
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    all_files = []
    for ext in supported_exts:
        all_files.extend(glob.glob(os.path.join(input_dir, ext)))

    print("=================================================================")
    print(" 🛠️  HỆ THỐNG SINH BACKGROUND TỰ ĐỘNG (TEMPORAL MEDIAN) - PRODUCTION")
    print("=================================================================")
    print(f"📁 Thư mục ảnh gốc (Origin) : {input_dir}")
    print(f"📁 Thư mục kết quả (BG)     : {output_dir}")
    print(f"⏱️  Khoảng cách slot giờ     : {time_interval_hours} tiếng/slot")
    print(f"🖼️  Số mẫu tối đa mỗi slot   : {max_samples_per_slot} frames")
    print(f"🌏 Múi giờ xử lý             : UTC+7 (Asia/Ho_Chi_Minh)")
    print(f"[*] Tìm thấy tổng cộng: {len(all_files)} file ảnh.")
    print("=================================================================")

    if not all_files:
        print("[!] Không tìm thấy file ảnh nào trong thư mục đầu vào.")
        return

    # BƯỚC 1: LỌC VÀ XOÁ ẢNH RÁC / PLACEHOLDER
    valid_files = []
    deleted_count = 0
    cpu_cores = os.cpu_count() or 4
    workers = max_workers or max(1, cpu_cores - 2)

    print(f"[*] [Bước 1/3] Đang quét và xoá ảnh rác trên {workers} workers CPU...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(check_and_delete_corrupted, all_files, chunksize=100),
            total=len(all_files),
            desc="Quét & xoá file lỗi",
        ))

    for file_path, was_deleted in zip(all_files, results):
        if was_deleted:
            deleted_count += 1
        else:
            valid_files.append(file_path)

    print(f"-> Đã loại bỏ vĩnh viễn : {deleted_count} file rác.")
    print(f"-> Số ảnh hợp lệ còn lại: {len(valid_files)} file.")

    if not valid_files:
        print("[!] Không còn ảnh hợp lệ nào sau khi lọc.")
        return

    # BƯỚC 2: GOM NHÓM THEO (TUYẾN ĐƯỜNG, SLOT GIỜ UTC+7)
    route_time_slots = defaultdict(list)
    routes_found = set()

    print("[*] [Bước 2/3] Đang phân tích timestamp và gom nhóm theo khung giờ...")
    for f in valid_files:
        route_id, dt = parse_image_info(f)
        routes_found.add(route_id)
        if dt is not None:
            # Slot giờ được chia theo time_interval_hours
            slot_id = (dt.hour // time_interval_hours) * time_interval_hours
            route_time_slots[(route_id, slot_id)].append(f)
        else:
            route_time_slots[(route_id, "unknown")].append(f)

    print(f"[*] Phát hiện {len(routes_found)} tuyến đường riêng biệt.")
    print(f"[*] Tổng số nhóm cần tính toán (Tuyến x Khung giờ): {len(route_time_slots)}")

    # BƯỚC 3: TÍNH TOÁN BACKGROUND SONG SONG
    slot_args = [
        (route_id, slot, paths, output_dir, max_samples_per_slot)
        for (route_id, slot), paths in sorted(
            route_time_slots.items(),
            key=lambda x: (str(x[0][0]), str(x[0][1]))
        )
    ]

    print(f"[*] [Bước 3/3] Đang tổng hợp Median Background trên {min(len(slot_args), workers)} workers...")
    success_slots = 0
    skipped_slots = 0

    with ProcessPoolExecutor(max_workers=min(len(slot_args), workers)) as executor:
        futures = {executor.submit(process_single_slot, arg): arg[:2] for arg in slot_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Tính Median Background"):
            route_id, slot, success, msg = future.result()
            if success:
                success_slots += 1
            else:
                skipped_slots += 1

    print("=================================================================")
    print(f"🎉 HOÀN THÀNH TẠO ẢNH NỀN!")
    print(f"   - Số slot nền thành công : {success_slots}")
    print(f"   - Số slot bị bỏ qua      : {skipped_slots} (không đủ ảnh)")
    print(f"   - Thư mục lưu kết quả    : {output_dir}")
    print("=================================================================")


def main():
    parser = argparse.ArgumentParser(description="Production Traffic Background Generator (Temporal Median)")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./output",
        help="Thư mục chứa ảnh origin cần tạo background",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./traffic_backgrounds",
        help="Thư mục lưu trữ background phân theo tuyến",
    )
    parser.add_argument(
        "--time_interval_hours",
        type=int,
        default=1,
        help="Khoảng cách khung giờ cho mỗi ảnh background (mặc định 1h/ảnh)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=30,
        help="Số lượng ảnh tối đa lấy mẫu để tính median cho mỗi slot",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Số lõi CPU xử lý song song (mặc định CPU - 2)",
    )
    args = parser.parse_args()

    process_traffic_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        time_interval_hours=args.time_interval_hours,
        max_samples_per_slot=args.max_samples,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
