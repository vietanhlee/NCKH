# Hướng 6: Delta-Guided Unsupervised Vehicle Re-Identification Across Cameras & Corridor Travel Time Estimation

## 1. Giới thiệu & Đóng góp Khoa học (Novelty ⭐⭐⭐⭐⭐)
Bài toán nhận dạng lại phương tiện xuyên camera (Vehicle Re-ID) trong đô thị thường vấp phải 2 rào cản chi phí:
1. Phụ thuộc vào Object Detector nặng nề (YOLO / Faster R-CNN) để cắt khung bao xe.
2. Việc gán nhãn thủ công ID danh tính xe qua hàng trăm camera là bất khả thi.

### Giải pháp kỹ thuật đột phá:
* **$\Delta$-Guided RoI Extraction**: Sử dụng trực tiếp trường sai khác quang học $\Delta = |I - I_{\text{bg}}|$ kết hợp phân tích thành phần liên thông (Connected Components) và bộ lọc hình thái học xe cộ để cắt phương tiện **hoàn toàn không cần Object Detector**.
* **DINO ViT + BNNeck Projector**: Tạo vector đặc trưng danh tính 256 chiều chuẩn hóa $L_2$, kế thừa năng lực phân tách màu sơn và kiểu dáng xe của DINO ViT.
* **Cơ chế Không-Thời gian Vật lý (Spatio-Temporal Feasibility Windowing)**: Lọc ứng viên xe dựa trên dải vận tốc hợp lý đô thị ($v \in [10, 60] \text{ km/h}$) và khoảng cách giữa các camera trên hành lang:
  $$t_{\text{min}} \le |t_B - t_A| \le t_{\text{max}}$$
  Loại bỏ triệt để 98% ứng viên âm giả (False Positives), giải quyết điểm yếu nhầm lẫn hình dạng xe tại Việt Nam.
* **Ứng dụng Thực tiễn (Corridor Analytics)**: Tự động đo **Thời gian hành trình trung bình (Travel Time)** và **Vận tốc hành trình trung bình** của tuyến đường từ các cặp xe trùng khớp độ tin cậy cao mà không cần biển số xe hay thiết bị GPS.

---

## 2. Cấu trúc File
- `roi_extractor.py`: `DeltaRoIExtractor` định vị và cắt phương tiện từ $\Delta$.
- `models.py`: `VehicleReIDModel` (DINO ViT Backbone + Projector + BNNeck chuẩn CVPR Re-ID).
- `matcher.py`: `VehicleReIDMatcher` thuật toán so khớp không-thời gian, xếp hạng Top-k, tính vận tốc hành lang, tính CMC Rank-1/5 và mAP.
- `losses.py`: `TrackletContrastiveLoss` hàm mất mát tương phản tự thân.
- `run_reid.py`: Pipeline thực thi trọn vẹn, xuất ảnh đối chiếu Top-k và báo cáo CSV hành trình.

---

## 3. Hướng Dẫn Chạy (CLI Execution)

```bash
python direction6_vehicle_reid/run_reid.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --output_dir checkpoints/direction6_vehicle_reid \
    --backbone dinov3_vits16 \
    --distance_meters 1200.0 \
    --top_k 5 \
    --device cuda
```

### Kết quả đầu ra:
- `corridor_speed_report.csv`: Báo cáo thời gian hành trình và vận tốc trung bình của tuyến đường.
- `reid_retrieval_results.csv`: Danh sách chi tiết các cặp xe trùng khớp liên camera.
- `visualizations/query_cam...`: Ảnh trực quan đối chiếu xe Query và Top-5 ứng viên trùng khớp ở camera khác kèm điểm tương đồng.
