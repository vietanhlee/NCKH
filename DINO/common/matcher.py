"""
=============================================================================
 Common Utility: TrafficPairMatcher
 Module đối sánh thông minh giữa ảnh Background tĩnh và ảnh Origin phương tiện
 Hỗ trợ cấu trúc lưu trữ thực tế của IC4SD-Traffic-HCM và các hệ thống CCTV đô thị
=============================================================================
"""

import os
import re
import glob
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


class TrafficPairMatcher:
    """
    Quản lý việc ánh xạ và ghép cặp (pairing) giữa ảnh phương tiện (Origin)
    và ảnh nền tĩnh (Background) tương ứng từ cùng vị trí quan sát của camera.

    Cấu trúc mặc định:
      - Origin: {origin_dir}/{stt}_{timestamp}.jpg (Ví dụ: '1_1755698811.jpg')
      - Background: {bg_dir}/route_{stt}/background_slot_{hour:02d}h.jpg
                    (Ví dụ: 'traffic_backgrounds/route_1/background_slot_00h.jpg')

    Hỗ trợ:
      - Tự động phân tích STT (Route ID / Camera ID) và Timestamp.
      - Chuyển đổi timestamp sang khung giờ tương ứng (theo Timezone Việt Nam UTC+7).
      - Tự động tìm kiếm slot giờ gần nhất nếu camera thiếu slot đúng giờ.
      - Hỗ trợ đa dạng chiến lược: 'route_hourly', 'camera_id', 'same_name', 'subfolder'.
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        bg_dir: str,
        origin_dir: str,
        match_strategy: str = "route_hourly",
        timezone_offset_hours: int = 7,
        camera_regex: str = r"^(\d+)_",
        strict_hour_match: bool = False,
    ):
        """
        Khởi tạo đối tượng Matcher.

        Args:
            bg_dir: Thư mục gốc chứa ảnh background.
            origin_dir: Thư mục chứa ảnh origin cần ghép cặp.
            match_strategy: Chiến lược đối sánh ('route_hourly', 'camera_id', 'same_name', 'subfolder').
            timezone_offset_hours: Múi giờ UTC offset (mặc định 7 cho múi giờ Việt Nam).
            camera_regex: Biểu thức chính quy trích xuất STT/Camera ID từ tên file.
            strict_hour_match: Nếu True, chỉ chấp nhận đúng slot giờ; nếu False, cho phép fallback slot gần nhất.
        """
        self.bg_dir = os.path.abspath(bg_dir) if bg_dir else ""
        self.origin_dir = os.path.abspath(origin_dir) if origin_dir else ""
        self.match_strategy = match_strategy
        self.tz = timezone(timedelta(hours=timezone_offset_hours))
        self.camera_regex = re.compile(camera_regex)
        self.strict_hour_match = strict_hour_match

        # Chỉ mục ảnh background: {route_id: {hour: path_to_bg_image}}
        self.bg_index: Dict[str, Dict[int, str]] = {}
        self.indexed = False

    def build_background_index(self) -> Dict[str, Dict[int, str]]:
        """
        Quét và xây dựng chỉ mục (Index) nhanh cho toàn bộ thư mục background.
        Cho phép tra cứu $O(1)$ đường dẫn ảnh nền theo (route_id, hour).
        """
        self.bg_index.clear()
        if not self.bg_dir or not os.path.isdir(self.bg_dir):
            return self.bg_index

        # Quét tất cả các file ảnh trong thư mục background
        for root, _, files in os.walk(self.bg_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.SUPPORTED_EXTS:
                    continue

                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, self.bg_dir)
                parts = Path(rel_path).parts

                # 1. Trích xuất route_id hoàn toàn động từ cấu trúc thư mục hoặc tên file
                route_id = None
                # Tìm trong tên thư mục cha (ví dụ: route_1, route_25, route_1050, stt_999, cam_A1)
                for part in parts[:-1]:
                    m_route = re.search(r"(?:route|stt|cam|camera)[_-]?([a-zA-Z0-9]+)", part, re.IGNORECASE)
                    if m_route:
                        val = m_route.group(1)
                        route_id = str(int(val)) if val.isdigit() else val
                        break

                # Nếu thư mục không có tiền tố route_, kiểm tra tên folder đầu tiên
                if route_id is None and len(parts) >= 2:
                    val = parts[0]
                    route_id = str(int(val)) if val.isdigit() else val

                # Nếu vẫn chưa có, kiểm tra phần định danh ở đầu tên file
                if route_id is None:
                    m_fnum = re.search(r"^([a-zA-Z0-9]+)", fname)
                    if m_fnum:
                        val = m_fnum.group(1)
                        route_id = str(int(val)) if val.isdigit() else val

                if route_id is None:
                    continue

                # 2. Trích xuất khung giờ (slot hour) từ tên file background
                # Ví dụ: background_slot_00h.jpg, slot_14h.jpg, bg_07.jpg
                m_hour = re.search(r"(?:slot[_-]?|bg[_-]?|h)?(\d{1,2})h?", fname, re.IGNORECASE)
                hour = -1
                if m_hour:
                    try:
                        extracted = int(m_hour.group(1))
                        if 0 <= extracted <= 23:
                            hour = extracted
                    except ValueError:
                        hour = -1

                if route_id not in self.bg_index:
                    self.bg_index[route_id] = {}

                # Ưu tiên lưu slot giờ hợp lệ, nếu không xác định giờ thì gán slot default (-1)
                self.bg_index[route_id][hour] = full_path

        self.indexed = True
        return self.bg_index

    def find_best_background(self, route_id: str, hour: int) -> Optional[Tuple[str, int]]:
        """
        Tìm kiếm ảnh background phù hợp nhất cho một tuyến đường và khung giờ.

        Args:
            route_id: ID tuyến đường / Camera STT (dạng chuỗi chuẩn hóa số nguyên).
            hour: Giờ quan sát (0 đến 23).

        Returns:
            Tuple (đường_dẫn_ảnh, giờ_thực_tế_của_slot) hoặc None nếu không tìm thấy.
        """
        if not self.indexed:
            self.build_background_index()

        route_key = str(int(route_id)) if route_id.isdigit() else route_id
        if route_key not in self.bg_index:
            return None

        available_slots = self.bg_index[route_key]
        if not available_slots:
            return None

        # 1. Khớp chính xác giờ
        if hour in available_slots:
            return available_slots[hour], hour

        # Nếu kích hoạt strict match, không tìm slot thay thế
        if self.strict_hour_match:
            return None

        # 2. Tìm slot giờ gần nhất theo khoảng cách chu kỳ ngày-đêm 24 giờ
        # (Ví dụ: 23h và 0h chỉ cách nhau 1 tiếng, thay vì khoảng cách thẳng 23 tiếng)
        valid_hours = [h for h in available_slots.keys() if 0 <= h <= 23]
        if valid_hours:
            def circular_dist(h1: int, h2: int) -> int:
                d = abs(h1 - h2)
                return min(d, 24 - d)

            best_hour = min(valid_hours, key=lambda h: circular_dist(h, hour))
            return available_slots[best_hour], best_hour

        # 3. Fallback lấy slot mặc định (-1) hoặc bất kỳ slot nào sẵn có
        if -1 in available_slots:
            return available_slots[-1], -1

        any_slot = next(iter(available_slots.values()))
        return any_slot, -1

    def parse_origin_filename(self, filename: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """
        Phân tích cú pháp tên file origin để lấy (route_id, timestamp, hour).
        Đồng bộ chuẩn 100% với hàm parse_image_info của script tạo background:
          - Tách tên: <route_id>_<timestamp_hoặc_chuỗi_ngày_giờ>.<ext>
          - Hỗ trợ Unix timestamp 10 chữ số (giây) và 13 chữ số (mili-giây).
          - Hỗ trợ các định dạng chuỗi ngày giờ: %Y%m%d%H%M%S, %Y%m%d_%H%M%S, %Y-%m-%d_%H-%M-%S.
          - Ánh xạ chính xác theo múi giờ chỉ định (mặc định UTC+7 cho Việt Nam).
        """
        basename = os.path.splitext(os.path.basename(filename))[0]
        parts = basename.split("_", 1)
        if len(parts) < 2:
            # Fallback nếu tên file không có dấu gạch dưới: chỉ tìm ID số ở đầu
            m_cam = self.camera_regex.match(basename)
            if m_cam:
                return str(int(m_cam.group(1))), None, None
            return None, None, None

        raw_route = parts[0]
        ts_str = parts[1]

        # Chuẩn hóa route_id (ví dụ: '1' -> '1', 'route_1' -> '1')
        m_r = re.search(r"(\d+)", raw_route)
        route_id = str(int(m_r.group(1))) if m_r else raw_route

        # 1. Trường hợp chuỗi ngày giờ có ký tự phân tách (_, -, :, khoảng trắng)
        if any(sep in ts_str for sep in ["-", "_", ":", " "]):
            date_formats_sep = [
                "%Y%m%d_%H%M%S",
                "%Y-%m-%d_%H-%M-%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y%m%d_%H%M",
            ]
            for fmt in date_formats_sep:
                try:
                    dt = datetime.strptime(ts_str, fmt)
                    return route_id, None, dt.hour
                except ValueError:
                    pass

        # 2. Trường hợp chuỗi thuần số (isdigit): Phân biệt theo độ dài chính xác
        if ts_str.isdigit():
            length = len(ts_str)
            # Dạng YYYYMMDDHHMMSS (14 chữ số)
            if length == 14:
                try:
                    dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
                    return route_id, None, dt.hour
                except ValueError:
                    pass
            # Dạng YYYYMMDDHHMM (12 chữ số)
            elif length == 12:
                try:
                    dt = datetime.strptime(ts_str, "%Y%m%d%H%M")
                    return route_id, None, dt.hour
                except ValueError:
                    pass
            # Dạng Unix timestamp mili-giây (13 chữ số)
            elif length == 13:
                try:
                    raw_sec = int(ts_str) / 1000.0
                    dt = datetime.fromtimestamp(raw_sec, tz=self.tz)
                    return route_id, int(raw_sec), dt.hour
                except (ValueError, OSError, OverflowError):
                    pass
            # Dạng Unix timestamp giây (9 đến 11 chữ số, chuẩn 10 chữ số)
            elif 9 <= length <= 11:
                try:
                    raw_sec = float(ts_str)
                    dt = datetime.fromtimestamp(raw_sec, tz=self.tz)
                    return route_id, int(raw_sec), dt.hour
                except (ValueError, OSError, OverflowError):
                    pass

        # 3. Fallback regex trích xuất timestamp số bất kỳ còn lại
        m_ts = re.search(r"(\d{10,13})", ts_str)
        if m_ts:
            try:
                val = int(m_ts.group(1))
                is_ms = len(m_ts.group(1)) >= 13
                raw_sec = val / 1000.0 if is_ms else float(val)
                dt = datetime.fromtimestamp(raw_sec, tz=self.tz)
                return route_id, int(raw_sec), dt.hour
            except (ValueError, OSError, OverflowError):
                pass

        return route_id, None, None

    def discover_pairs(self, max_pairs: Optional[int] = None) -> List[Dict[str, Union[str, int]]]:
        """
        Quét và sinh danh sách toàn bộ các cặp (Background, Origin) khớp nhau.

        Returns:
            List[Dict] với cấu trúc:
              - 'origin_path': đường dẫn tuyệt đối ảnh origin
              - 'bg_path': đường dẫn tuyệt đối ảnh background
              - 'route_id': ID tuyến đường / STT
              - 'origin_hour': giờ của ảnh origin
              - 'bg_hour': giờ thực tế của slot background được chọn
              - 'origin_name': tên file origin
        """
        if not self.indexed:
            self.build_background_index()

        if not self.origin_dir or not os.path.isdir(self.origin_dir):
            return []

        # Liệt kê tất cả file ảnh origin
        origin_files = []
        for root, _, files in os.walk(self.origin_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in self.SUPPORTED_EXTS:
                    origin_files.append(os.path.join(root, f))

        origin_files.sort()
        pairs: List[Dict[str, Union[str, int]]] = []

        if self.match_strategy == "route_hourly":
            for op in origin_files:
                # 0. Kiểm tra an toàn: Lọc file rỗng hoặc placeholder 'IMAGE NOT AVAILABLE' (284x177)
                try:
                    fsize = os.path.getsize(op)
                    if fsize < 1024:  # File quá nhỏ / rỗng
                        continue
                    if 5800 <= fsize <= 7200:
                        # Kiểm tra nhanh kích thước ảnh
                        from PIL import Image as _PILImg
                        with _PILImg.open(op) as _tmp_img:
                            if _tmp_img.size == (284, 177):
                                continue
                except Exception:
                    continue

                fname = os.path.basename(op)
                route_id, timestamp, hour = self.parse_origin_filename(fname)
                if route_id is None:
                    continue

                search_hour = hour if hour is not None else 12  # Giờ mặc định trưa nếu không parse được
                match_res = self.find_best_background(route_id, search_hour)
                if match_res is None:
                    continue

                bg_path, bg_hour = match_res
                pairs.append({
                    "origin_path": op,
                    "bg_path": bg_path,
                    "route_id": route_id,
                    "origin_hour": search_hour,
                    "bg_hour": bg_hour,
                    "origin_name": fname,
                })

                if max_pairs and len(pairs) >= max_pairs:
                    break

        elif self.match_strategy == "same_name":
            # Ghép cặp file trùng tên chính xác
            bg_name_map = {}
            for slots in self.bg_index.values():
                for bp in slots.values():
                    bg_name_map[os.path.basename(bp)] = bp

            for op in origin_files:
                fname = os.path.basename(op)
                if fname in bg_name_map:
                    pairs.append({
                        "origin_path": op,
                        "bg_path": bg_name_map[fname],
                        "route_id": os.path.splitext(fname)[0],
                        "origin_hour": -1,
                        "bg_hour": -1,
                        "origin_name": fname,
                    })
                if max_pairs and len(pairs) >= max_pairs:
                    break

        else:
            # Fallback theo camera ID thông thường
            for op in origin_files:
                fname = os.path.basename(op)
                route_id, _, _ = self.parse_origin_filename(fname)
                if route_id and route_id in self.bg_index:
                    any_bg = next(iter(self.bg_index[route_id].values()))
                    pairs.append({
                        "origin_path": op,
                        "bg_path": any_bg,
                        "route_id": route_id,
                        "origin_hour": -1,
                        "bg_hour": -1,
                        "origin_name": fname,
                    })
                if max_pairs and len(pairs) >= max_pairs:
                    break

        return pairs

    def to_dataframe(self, max_pairs: Optional[int] = None) -> pd.DataFrame:
        """Xuất danh sách các cặp ảnh dưới dạng pandas DataFrame."""
        pairs = self.discover_pairs(max_pairs=max_pairs)
        return pd.DataFrame(pairs)
