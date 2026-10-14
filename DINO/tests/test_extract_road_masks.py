"""
=============================================================================
 Test Suite: Road Mask Extractor (test_extract_road_masks.py)
 Kiểm thử tự động toàn diện module trích xuất lòng đường xe chạy:
   1. DaytimeFilter: Nhận diện múi giờ UTC+7, phân loại ngày/đêm và kiểm tra quang trắc.
   2. VehicleOccupancyAccumulator: Tích lũy bản đồ nhiệt từ chuỗi ảnh.
   3. DrivableCorridorReconstructor: Sinh mặt nạ nhị phân, lấp đầy làn xe, tạo ảnh overlay.
   4. End-to-End CLI Smoke Test: Kiểm thử quy trình chạy đầy đủ trên dữ liệu mô phỏng.
=============================================================================
"""

import os
import sys
import shutil
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
from PIL import Image

# Đảm bảo mã hóa UTF-8 cho Windows Terminal và chống xung đột OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn project
dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dino_dir not in sys.path:
    sys.path.insert(0, dino_dir)

from common.road_extractor import DaytimeFilter, VehicleOccupancyAccumulator, DrivableCorridorReconstructor


class TestRoadMaskExtractor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_road_mask_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_daytime_filter_time_and_luminance(self):
        """Kiểm tra DaytimeFilter phân loại chính xác ngày/đêm và lọc quang trắc."""
        day_filter = DaytimeFilter(start_hour=6, end_hour=18, min_mean_v=45.0)

        # 1. Kiểm tra parse thời gian
        # 14h30 chiều ban ngày
        r_id, dt1 = day_filter.parse_image_time("route_10_20240510_143000.jpg")
        self.assertEqual(r_id, "10")
        self.assertIsNotNone(dt1)
        self.assertEqual(dt1.hour, 14)

        # 02h15 đêm
        r_id, dt2 = day_filter.parse_image_time("256_20240510_021500.jpg")
        self.assertEqual(r_id, "256")
        self.assertEqual(dt2.hour, 2)

        # 2. Tạo ảnh ban ngày sáng rõ (V cao) và ảnh đêm tối om (V thấp)
        day_img_path = os.path.join(self.temp_dir, "1_20240510_100000.jpg")
        bright_arr = np.full((128, 128, 3), 160, dtype=np.uint8)  # Sáng rõ
        cv2.imwrite(day_img_path, bright_arr)

        night_img_path = os.path.join(self.temp_dir, "1_20240510_230000.jpg")
        cv2.imwrite(night_img_path, bright_arr)  # Giờ đêm 23h

        dark_day_img_path = os.path.join(self.temp_dir, "1_20240510_120000.jpg")
        dark_arr = np.full((128, 128, 3), 15, dtype=np.uint8)  # Đen kịt V < 45
        cv2.imwrite(dark_day_img_path, dark_arr)

        # Đánh giá
        is_day1, reason1, h1 = day_filter.is_daytime_image(day_img_path)
        self.assertTrue(is_day1, f"Kỳ vọng hợp lệ ban ngày: {reason1}")
        self.assertEqual(h1, 10)

        is_day2, reason2, h2 = day_filter.is_daytime_image(night_img_path)
        self.assertFalse(is_day2, "Kỳ vọng bị từ chối vì giờ ban đêm")
        self.assertIn("ban đêm", reason2.lower())

        is_day3, reason3, h3 = day_filter.is_daytime_image(dark_day_img_path)
        self.assertFalse(is_day3, "Kỳ vọng bị từ chối vì quá tối đen")
        self.assertIn("quang trắc", reason3.lower())

    def test_02_vehicle_occupancy_accumulator_and_fast_diff(self):
        """Kiểm tra tích lũy xác suất chuyển động xe qua các khung hình."""
        accumulator = VehicleOccupancyAccumulator(model=None)

        # Tạo background chuẩn và 3 frame có xe ở các vị trí lòng đường khác nhau
        H, W = 120, 160
        bg_arr = np.full((H, W, 3), 120, dtype=np.uint8)
        bg_path = os.path.join(self.temp_dir, "bg.jpg")
        cv2.imwrite(bg_path, bg_arr)

        frame_paths = []
        for i in range(3):
            f_arr = bg_arr.copy()
            # Đặt "xe" ở nửa dưới (lòng đường)
            y_start = 50 + i * 15
            f_arr[y_start:y_start + 20, 40:120] = 240
            f_path = os.path.join(self.temp_dir, f"1_20240510_{10 + i:02d}0000.jpg")
            cv2.imwrite(f_path, f_arr)
            frame_paths.append(f_path)

        accum_heat, sample_bgr = accumulator.accumulate_route(
            frame_paths=frame_paths,
            bg_path=bg_path,
            max_frames=10,
            use_model=False,
        )

        self.assertEqual(accum_heat.shape, (H, W))
        self.assertEqual(sample_bgr.shape, (H, W, 3))
        # Vùng có xe phải có nhiệt độ cao hơn vùng góc trên (bầu trời)
        self.assertGreater(float(np.mean(accum_heat[50:90, 40:120])), float(np.mean(accum_heat[:30, :])))

    def test_03_drivable_corridor_reconstructor(self):
        """Kiểm tra tái tạo mặt nạ lòng đường và ảnh phủ màu dạ quang."""
        reconstructor = DrivableCorridorReconstructor(
            seed_threshold=0.1,
            min_road_area_ratio=0.05,
            horizon_margin_ratio=0.15,
        )

        H, W = 150, 200
        mock_heat = np.zeros((H, W), dtype=np.float32)
        # Mô phỏng lòng đường ở nửa dưới (y từ 40 đến 140, x từ 20 đến 180)
        mock_heat[40:140, 20:180] = 0.5
        mock_heat[10:30, :] = 0.8  # Bầu trời/đèn đường giả (phải bị horizon_margin cắt bỏ)

        mock_img = np.full((H, W, 3), 130, dtype=np.uint8)

        road_mask, stats = reconstructor.reconstruct_road_mask(mock_heat, reference_img_bgr=mock_img)

        self.assertEqual(road_mask.shape, (H, W))
        self.assertEqual(road_mask.dtype, np.uint8)
        self.assertTrue(((road_mask == 0) | (road_mask == 255)).all())

        # Dải trên cùng phải sạch bóng 0 (loại trừ bầu trời)
        top_cut = int(H * 0.15)
        self.assertEqual(int(np.sum(road_mask[:top_cut, :])), 0)

        # Lòng đường phải được nhận diện
        self.assertGreater(stats["road_area_pct"], 10.0)

        # Kiểm tra tạo ảnh overlay
        overlay = reconstructor.create_road_overlay(mock_img, road_mask)
        self.assertEqual(overlay.shape, (H, W, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_04_end_to_end_cli_pipeline(self):
        """Kiểm thử toàn diện pipeline extract_road_masks.py qua CLI."""
        origin_dir = os.path.join(self.temp_dir, "origin")
        bg_dir = os.path.join(self.temp_dir, "backgrounds")
        output_dir = os.path.join(self.temp_dir, "road_masks_out")
        os.makedirs(origin_dir, exist_ok=True)
        os.makedirs(os.path.join(bg_dir, "route_1"), exist_ok=True)
        os.makedirs(os.path.join(bg_dir, "route_2"), exist_ok=True)

        # Tạo background mẫu cho route 1 và 2
        bg1 = np.full((120, 160, 3), 110, dtype=np.uint8)
        cv2.imwrite(os.path.join(bg_dir, "route_1", "background_slot_12h.jpg"), bg1)
        bg2 = np.full((120, 160, 3), 115, dtype=np.uint8)
        cv2.imwrite(os.path.join(bg_dir, "route_2", "background_slot_12h.jpg"), bg2)

        # Tạo các frame ban ngày cho route 1 (3 frames)
        for hour in [8, 12, 16]:
            img = bg1.copy()
            img[50:100, 30:130] = 230  # Xe chạy
            cv2.imwrite(os.path.join(origin_dir, f"1_20240510_{hour:02d}0000.jpg"), img)

        # Tạo các frame ban ngày cho route 2 (2 frames)
        for hour in [9, 14]:
            img = bg2.copy()
            img[40:90, 20:140] = 220
            cv2.imwrite(os.path.join(origin_dir, f"2_20240510_{hour:02d}0000.jpg"), img)

        # Tạo 1 frame ban đêm (phải bị bỏ qua)
        night_img = np.full((120, 160, 3), 20, dtype=np.uint8)
        cv2.imwrite(os.path.join(origin_dir, "1_20240510_010000.jpg"), night_img)

        # Chạy CLI qua subprocess
        import subprocess
        cmd = [
            sys.executable,
            os.path.join(dino_dir, "extract_road_masks.py"),
            "--origin_dir", origin_dir,
            "--bg_dir", bg_dir,
            "--output_dir", output_dir,
            "--method", "fast",
            "--min_daytime_frames", "2",
            "--max_frames_per_route", "5",
            "--device", "cpu",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(res.returncode, 0, f"Lỗi chạy extract_road_masks.py: {res.stderr}\nOutput: {res.stdout}")

        # Kiểm tra các tệp kết quả xuất ra
        mask_r1 = os.path.join(output_dir, "route_1_road_mask.png")
        overlay_r1 = os.path.join(output_dir, "route_1_road_overlay.jpg")
        heat_r1 = os.path.join(output_dir, "route_1_heatmap.jpg")

        mask_r2 = os.path.join(output_dir, "route_2_road_mask.png")
        summary_json = os.path.join(output_dir, "road_masks_summary.json")
        summary_csv = os.path.join(output_dir, "road_masks_summary.csv")

        self.assertTrue(os.path.isfile(mask_r1), "Thiếu file route_1_road_mask.png")
        self.assertTrue(os.path.isfile(overlay_r1), "Thiếu file route_1_road_overlay.jpg")
        self.assertTrue(os.path.isfile(heat_r1), "Thiếu file route_1_heatmap.jpg")
        self.assertTrue(os.path.isfile(mask_r2), "Thiếu file route_2_road_mask.png")
        self.assertTrue(os.path.isfile(summary_json), "Thiếu file summary JSON")
        self.assertTrue(os.path.isfile(summary_csv), "Thiếu file summary CSV")

        # Kiểm tra nội dung mask
        m1 = cv2.imread(mask_r1, cv2.IMREAD_GRAYSCALE)
        self.assertEqual(m1.shape, (120, 160))
        self.assertGreater(int(np.sum(m1 > 0)), 0)


if __name__ == "__main__":
    unittest.main()
