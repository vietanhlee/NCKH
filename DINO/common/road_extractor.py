"""
=============================================================================
 Common Utility: Road Extractor & Drivable Corridor Reconstruction
 Module trích xuất làn đường / phần lòng đường di chuyển của xe (Drivable Road Mask)
 Hỗ trợ:
   1. DaytimeFilter: Lọc tự động các khung hình ban ngày (UTC+7, 06:00 - 18:00) + kiểm tra quang trắc
   2. VehicleOccupancyAccumulator: Tích lũy vết tích chuyển động của phương tiện qua chuỗi khung hình
   3. DrivableCorridorReconstructor: Hợp nhất vết tích xe với bề mặt nhựa đường từ Background ban ngày,
      lấp đầy khoảng cách giữa các làn xe và triệt tiêu vỉa hè, cây cối, nhà cửa, bầu trời
=============================================================================
"""

import os
import re
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Múi giờ Việt Nam UTC+7
TZ_VN = timezone(timedelta(hours=7))


class DaytimeFilter:
    """
    Bộ lọc ảnh ban ngày cho camera giám sát giao thông:
      - Phân tích cú pháp thời gian từ tên file ảnh (<route_id>_<timestamp>.<ext>).
      - Xác định khung giờ ban ngày (mặc định: từ 06:00 đến 17:59 theo múi giờ UTC+7).
      - Kiểm tra độ sáng quang trắc (Photometric Luminance Safety Check): loại bỏ ảnh mất nguồn,
        đen màn hình hoặc mưa bão tối mịt (kênh V trong HSV < min_mean_v).
    """

    def __init__(
        self,
        start_hour: int = 6,
        end_hour: int = 18,
        min_mean_v: float = 45.0,
        min_file_size_kb: float = 0.5,
    ):
        """
        Args:
            start_hour: Giờ bắt đầu tính ban ngày (0-23, mặc định 6 = 06:00).
            end_hour: Giờ kết thúc ban ngày (0-23, mặc định 18 = 17:59:59).
            min_mean_v: Ngưỡng độ sáng trung bình tối thiểu kênh V (HSV, thang 0-255).
            min_file_size_kb: Kích thước file tối thiểu (loại bỏ file 0 byte hoặc hỏng < 0.5 KB).
        """
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.min_mean_v = min_mean_v
        self.min_file_size_kb = min_file_size_kb

    @staticmethod
    def parse_image_time(filename: str) -> Tuple[str, Optional[datetime]]:
        """
        Phân tích tên file ảnh để trích xuất (route_id, datetime_obj).
        Hỗ trợ chuẩn 100% các định dạng từ hệ thống camera IC4SD-Traffic-HCM:
          - Dạng: route_10_20240510_143000.jpg, 1_1755698811.jpg, cam_256_...
          - Unix timestamp 10 chữ số (giây) và 13 chữ số (mili-giây).
          - Chuỗi ngày giờ: YYYYMMDDHHMMSS, YYYY-MM-DD_HH-MM-SS...
        """
        basename = os.path.splitext(os.path.basename(filename))[0]

        # Khớp mẫu: (tiền tố route/stt/cam) + số route_id + dấu phân cách + chuỗi thời gian
        m = re.match(r"^(?:route|stt|cam|camera)?[_-]?(\d+)[_-](.+)$", basename, re.IGNORECASE)
        if m:
            route_id = str(int(m.group(1)))
            ts_str = m.group(2).strip()
        else:
            parts = basename.split("_", 1)
            if len(parts) >= 2:
                raw_route = parts[0]
                ts_str = parts[1].strip()
                m_r = re.search(r"(\d+)", raw_route)
                route_id = str(int(m_r.group(1))) if m_r else raw_route
            else:
                m_num = re.search(r"(\d+)", basename)
                return (str(int(m_num.group(1))) if m_num else "unknown"), None

        # 1. Chuỗi có ký tự phân tách
        if any(sep in ts_str for sep in ["-", "_", ":", " "]):
            date_formats = [
                "%Y%m%d_%H%M%S",
                "%Y-%m-%d_%H-%M-%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y%m%d_%H%M",
                "%Y-%m-%d_%H%M",
            ]
            for fmt in date_formats:
                try:
                    return route_id, datetime.strptime(ts_str, fmt)
                except ValueError:
                    pass

        # 2. Chuỗi thuần số
        if ts_str.isdigit():
            length = len(ts_str)
            if length == 14:
                try:
                    return route_id, datetime.strptime(ts_str, "%Y%m%d%H%M%S")
                except ValueError:
                    pass
            elif length == 12:
                try:
                    return route_id, datetime.strptime(ts_str, "%Y%m%d%H%M")
                except ValueError:
                    pass
            elif length == 13:
                try:
                    ts_sec = int(ts_str) / 1000.0
                    return route_id, datetime.fromtimestamp(ts_sec, tz=TZ_VN)
                except (ValueError, OSError, OverflowError):
                    pass
            elif 9 <= length <= 11:
                try:
                    ts_sec = float(ts_str)
                    return route_id, datetime.fromtimestamp(ts_sec, tz=TZ_VN)
                except (ValueError, OSError, OverflowError):
                    pass

        # 3. Fallback regex tìm chuỗi số timestamp bất kỳ
        m_ts = re.search(r"(\d{10,13})", ts_str)
        if m_ts:
            try:
                val = int(m_ts.group(1))
                raw_sec = val / 1000.0 if len(m_ts.group(1)) >= 13 else float(val)
                return route_id, datetime.fromtimestamp(raw_sec, tz=TZ_VN)
            except (ValueError, OSError, OverflowError):
                pass

        return route_id, None

    def is_daytime_image(
        self,
        image_path: str,
        verify_content: bool = True
    ) -> Tuple[bool, str, Optional[int]]:
        """
        Kiểm tra xem một file ảnh có phải là ảnh ban ngày hợp lệ hay không.

        Returns:
            (is_day, reason, hour)
        """
        if not os.path.isfile(image_path):
            return False, "File không tồn tại", None

        # Kiểm tra dung lượng
        try:
            fsize_kb = os.path.getsize(image_path) / 1024.0
            if fsize_kb < self.min_file_size_kb:
                return False, f"Dung lượng file quá nhỏ ({fsize_kb:.1f} KB < {self.min_file_size_kb} KB)", None
        except OSError as e:
            return False, f"Lỗi truy cập file: {e}", None

        # Trích xuất thời gian từ tên file
        _, dt = self.parse_image_time(image_path)
        hour = dt.hour if dt is not None else None

        # Nếu trích xuất được giờ: kiểm tra dải giờ
        if hour is not None:
            if not (self.start_hour <= hour < self.end_hour):
                return False, f"Khung giờ ban đêm ({hour:02d}h nằm ngoài [{self.start_hour:02d}h, {self.end_hour:02d}h))", hour

        # Kiểm tra quang trắc thực tế trên nội dung ảnh (nếu yêu cầu)
        if verify_content:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Không thể đọc nội dung ảnh (file hỏng)", hour

            h, w = img.shape[:2]
            if h < 50 or w < 50:
                return False, f"Độ phân giải quá nhỏ ({w}x{h})", hour

            # Kiểm tra ảnh placeholder 'IMAGE NOT AVAILABLE' (284x177)
            if w == 284 and h == 177:
                return False, "Ảnh placeholder 'IMAGE NOT AVAILABLE'", hour

            # Tính độ sáng trung bình qua kênh V trong HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mean_v = float(np.mean(hsv[:, :, 2]))

            if mean_v < self.min_mean_v:
                return False, f"Độ sáng quang trắc quá thấp (V={mean_v:.1f} < {self.min_mean_v})", hour

            # Nếu tên file không có giờ nhưng độ sáng đủ cao -> chấp nhận là ban ngày
            if hour is None:
                hour = 12

        return True, "Hợp lệ ban ngày", (hour if hour is not None else 12)


class VehicleOccupancyAccumulator:
    """
    Bộ tích lũy vết tích chuyển động của phương tiện giao thông (Vehicle Spatial Occupancy Accumulator):
      - Nhận diện các vùng có xe cộ lưu thông trên từng khung hình ban ngày.
      - Hỗ trợ 2 phương pháp trích xuất:
        1. Model-based: Sử dụng mạng nơ-ron Hướng 2 (TrafficDecompositionNet) trích xuất alpha_mask.
        2. Fast-difference: Sử dụng trừ nền bất biến ánh sáng & Otsu morphology khi không có checkpoint GPU.
      - Tích lũy xác suất không gian để tạo bản đồ nhiệt (Occupancy Heatmap).
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        device: Union[str, torch.device] = "cpu",
        img_size: int = 256,
    ):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.img_size = img_size
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def extract_vehicle_mask_model(
        self,
        img_pil: Image.Image,
        prior_pil: Optional[Image.Image] = None
    ) -> np.ndarray:
        """Trích xuất mặt nạ xe bằng mạng nơ-ron TrafficDecompositionNet."""
        orig_w, orig_h = img_pil.size
        # Tiền xử lý
        img_resized = img_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
        x = torch.from_numpy(np.array(img_resized).transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self.device)

        prior_tensor = None
        if prior_pil is not None:
            p_resized = prior_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
            prior_tensor = torch.from_numpy(np.array(p_resized).transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(x, prior=prior_tensor, use_prior=(prior_tensor is not None))
            mask = preds.get("alpha_mask", preds.get("pred_mask")).squeeze().cpu().numpy()

        # Resize về kích thước gốc
        mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask_full, 0.0, 1.0)

    @staticmethod
    def extract_vehicle_mask_fast(
        img_bgr: np.ndarray,
        bg_bgr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Trích xuất vết tích xe nhanh bằng hiệu chỉnh trắc quang và trừ nền hình thái học:
        Không cần GPU/checkpoint, chạy cực nhanh và ổn định.
        """
        h, w = img_bgr.shape[:2]

        if bg_bgr is not None:
            # Đồng bộ kích thước background nếu lệch
            if bg_bgr.shape[:2] != (h, w):
                bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

            # Căn chỉnh độ sáng kênh V giữa img và bg
            hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

            gain = float(np.mean(hsv_img[:, :, 2]) + 1.0) / float(np.mean(hsv_bg[:, :, 2]) + 1.0)
            gain = np.clip(gain, 0.6, 1.6)
            hsv_bg[:, :, 2] = np.clip(hsv_bg[:, :, 2] * gain, 0.0, 255.0)
            bg_aligned = cv2.cvtColor(hsv_bg.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Sai khác đa kênh
            diff = cv2.absdiff(img_bgr, bg_aligned)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            # Nếu không có ảnh nền: dùng gradient độ tương phản cục bộ và Laplacian
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            diff_gray = cv2.Sobel(blur, cv2.CV_8U, 1, 1, ksize=3)

        # Làm mờ và ngưỡng Otsu
        blurred = cv2.GaussianBlur(diff_gray, (7, 7), 1.5)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Lọc hình thái học để gom các cụm xe
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

        return (clean_mask.astype(np.float32) / 255.0)

    def accumulate_route(
        self,
        frame_paths: List[str],
        bg_path: Optional[str] = None,
        max_frames: int = 30,
        use_model: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tích lũy xác suất vết tích phương tiện qua danh sách khung hình ban ngày của một tuyến.

        Returns:
            (accum_heatmap, sample_daytime_bgr)
        """
        if not frame_paths:
            raise ValueError("Danh sách khung hình ban ngày rỗng!")

        # Lấy mẫu đều đặn theo thời gian nếu số frame lớn
        if len(frame_paths) > max_frames:
            step = len(frame_paths) / float(max_frames)
            sampled_paths = [frame_paths[int(i * step)] for i in range(max_frames)]
        else:
            sampled_paths = frame_paths

        # Đọc ảnh nền nếu có
        bg_bgr = None
        bg_pil = None
        if bg_path and os.path.isfile(bg_path):
            try:
                bg_bgr = cv2.imread(bg_path)
                bg_pil = Image.open(bg_path).convert("RGB")
            except Exception:
                pass

        sample_img_bgr = None
        accum_map = None
        valid_count = 0

        for p in sampled_paths:
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            if sample_img_bgr is None:
                sample_img_bgr = img_bgr.copy()
                accum_map = np.zeros((h, w), dtype=np.float32)

            if use_model and self.model is not None:
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    mask = self.extract_vehicle_mask_model(pil_img, prior_pil=bg_pil)
                except Exception:
                    mask = self.extract_vehicle_mask_fast(img_bgr, bg_bgr=bg_bgr)
            else:
                mask = self.extract_vehicle_mask_fast(img_bgr, bg_bgr=bg_bgr)

            if mask.shape != accum_map.shape:
                mask = cv2.resize(mask, (accum_map.shape[1], accum_map.shape[0]), interpolation=cv2.INTER_LINEAR)

            accum_map += mask
            valid_count += 1

        if valid_count > 0:
            accum_map /= float(valid_count)
        else:
            raise RuntimeError("Không đọc được khung hình hợp lệ nào để tích lũy!")

        return accum_map, sample_img_bgr


class DrivableCorridorReconstructor:
    """
    Bộ tái cấu trúc hành lang lòng đường xe chạy (Drivable Corridor Reconstructor):
      - Kết hợp bản đồ nhiệt vết tích xe với tính liên thông bề mặt nhựa đường từ ảnh nền.
      - Lấp đầy các khoảng trống giữa các làn xe bằng Morphological Closing & Convex Envelope.
      - Loại trừ triệt để vùng bầu trời (trên đường chân trời), vỉa hè người đi bộ, cây cối, mái nhà.
      - Xuất mặt nạ nhị phân sắc nét {0, 255} và ảnh overlay dạ quang trực quan.
    """

    def __init__(
        self,
        seed_threshold: float = 0.12,
        min_road_area_ratio: float = 0.05,
        horizon_margin_ratio: float = 0.15,
    ):
        """
        Args:
            seed_threshold: Ngưỡng xác suất tối thiểu để coi một pixel là vùng xe chạy thường xuyên.
            min_road_area_ratio: Tỷ lệ diện tích tối thiểu của lòng đường so với toàn ảnh (mặc định 5%).
            horizon_margin_ratio: Khoảng cách từ đỉnh ảnh (bầu trời) luôn bị loại bỏ (mặc định 15%).
        """
        self.seed_threshold = seed_threshold
        self.min_road_area_ratio = min_road_area_ratio
        self.horizon_margin_ratio = horizon_margin_ratio

    def reconstruct_road_mask(
        self,
        accum_heatmap: np.ndarray,
        reference_img_bgr: Optional[np.ndarray] = None,
        bg_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Tái tạo mặt nạ nhị phân lòng đường xe chạy từ bản đồ nhiệt tích lũy.

        Returns:
            (road_mask_uint8, stats_dict)
        """
        H, W = accum_heatmap.shape[:2]
        total_pixels = float(H * W)

        # 1. Khử nhiễu làm mượt bản đồ nhiệt
        smooth_heat = cv2.GaussianBlur(accum_heatmap, (15, 15), 3.0)

        # Chuẩn hóa về [0, 1]
        max_val = float(np.max(smooth_heat))
        if max_val > 1e-4:
            norm_heat = smooth_heat / max_val
        else:
            norm_heat = smooth_heat

        # 2. Tạo hạt giống vùng lõi xe chạy (Core Trajectory Seeds)
        core_seeds = (norm_heat >= self.seed_threshold).astype(np.uint8) * 255

        # 3. Loại trừ dải phía trên cùng (bầu trời, đỉnh cây, nóc nhà)
        top_horizon = max(1, int(H * self.horizon_margin_ratio))
        core_seeds[:top_horizon, :] = 0

        # Khử dải timestamp dưới đáy nếu có
        bottom_margin = int(H * 0.98)
        core_seeds[bottom_margin:, :] = 0

        # 4. Đóng kín hành lang giữa các làn đường (Multi-scale Directional Closing)
        kernel_h = max(5, int(H * 0.05))
        kernel_w = max(5, int(W * 0.06))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
        corridor = cv2.morphologyEx(core_seeds, cv2.MORPH_CLOSE, kernel_large)

        # Dilation nhẹ để phủ hết mép lề đường nơi xe chạy sát mép
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        corridor = cv2.dilate(corridor, kernel_dil, iterations=1)

        # 5. Phân cụm và giữ lại thành phần liên thông chính của mạng lưới đường
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(corridor)
        min_pixels = int(total_pixels * self.min_road_area_ratio)

        final_mask = np.zeros((H, W), dtype=np.uint8)
        if num_labels > 1:
            valid_labels = []
            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area >= min_pixels:
                    valid_labels.append(lbl)

            if not valid_labels:
                largest_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                valid_labels = [largest_lbl]

            for lbl in valid_labels:
                final_mask[labels == lbl] = 255

        # 6. Lấp kín các lỗ thủng bên trong lòng đường (fill internal holes)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filled = np.zeros((H, W), dtype=np.uint8)
        for cnt in contours:
            cv2.drawContours(mask_filled, [cnt], -1, 255, thickness=cv2.FILLED)

        # Làm mịn đường bao ngoài bằng Gaussian Blur + Threshold
        mask_blur = cv2.GaussianBlur(mask_filled.astype(np.float32) / 255.0, (9, 9), 2.0)
        binary_road_mask = ((mask_blur >= 0.45) * 255).astype(np.uint8)

        # Cắt lại biên an toàn phía trên cùng
        binary_road_mask[:top_horizon, :] = 0

        # Thống kê
        road_pixels = int(np.sum(binary_road_mask > 0))
        road_ratio = float(road_pixels / total_pixels) * 100.0

        stats_dict = {
            "road_pixels": road_pixels,
            "total_pixels": int(total_pixels),
            "road_area_pct": round(road_ratio, 2),
            "width": W,
            "height": H,
        }

        return binary_road_mask, stats_dict

    @staticmethod
    def create_road_overlay(
        image_bgr: np.ndarray,
        road_mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 230, 118),  # Xanh ngọc dạ quang (Emerald Neon BGR: 118, 230, 0)
        alpha: float = 0.45,
        draw_contour: bool = True,
        contour_color: Tuple[int, int, int] = (255, 255, 255),  # Viền trắng nổi bật
        contour_thickness: int = 2,
    ) -> np.ndarray:
        """
        Tạo ảnh trực quan phủ màu bán trong suốt dạ quang lên lòng đường xe chạy:
        Giúp nhìn rõ ranh giới lòng đường xe di chuyển, vạch kẻ làn, dải phân cách và vỉa hè.
        """
        h, w = image_bgr.shape[:2]
        if road_mask.shape != (h, w):
            road_mask = cv2.resize(road_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_bool = (road_mask > 127)
        overlay = image_bgr.copy().astype(np.float32)

        color_bgr = np.array([color[2], color[1], color[0]], dtype=np.float32)

        overlay[mask_bool] = (
            overlay[mask_bool] * (1.0 - alpha) + color_bgr * alpha
        )
        overlay_final = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

        if draw_contour:
            contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_final, contours, -1, contour_color, contour_thickness, cv2.LINE_AA)

        return overlay_final

    @staticmethod
    def create_heatmap_visualization(
        accum_heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """Tạo ảnh trực quan Heatmap thể hiện mật độ lưu thông xe trên các làn đường."""
        norm_map = np.clip(accum_heatmap / (np.max(accum_heatmap) + 1e-5), 0.0, 1.0)
        uint8_map = (norm_map * 255.0).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(uint8_map, colormap)
        return color_heatmap
