# Hướng 6: Delta-Guided Unsupervised Vehicle Re-Identification Across Cameras

## 1. Giới thiệu & Đóng góp Khoa học (Novelty ⭐⭐⭐⭐)
Bài toán nhận dạng lại phương tiện liên camera (Vehicle Re-Identification - Re-ID) trong mạng lưới camera giám sát đô thị (608 camera tại TP.HCM) đối mặt với hai rào cản lớn:
1. Phải chạy bộ dò vật thể (Object Detector như YOLO/Faster R-CNN) rất nặng nề trên từng khung hình để cắt vùng xe.
2. Chi phí gán nhãn danh tính xe (Identity ID) thủ công qua các camera khác nhau là bất khả thi ở quy mô lớn.

**Giải pháp đề xuất**:
1. **Delta-Guided RoI Extraction**: Sử dụng trực tiếp trường sai khác quang học $\Delta = \|I_{\text{origin}} - I_{\text{bg}}\|$ để cắt tự động các phương tiện chuyển động mà **không cần bộ dò Object Detector**.
2. **DINO ViT + BNNeck Feature Extractor**: Trích xuất vector đặc trưng $L_2$-normalized 256 chiều có tính bất biến mạnh mẽ với ánh sáng và góc quay camera.
3. **Unsupervised Tracklet Mining**: Khai thác chuỗi quan sát liên tục từ cùng camera làm cặp dương tự thân, kết hợp hàm mất mát tương phản để tối ưu hóa không gian biểu diễn danh tính xe.
4. **Truy vấn so khớp Cosine & Đánh giá CMC / mAP**: Đánh giá chuẩn học thuật quốc tế phục vụ các hội nghị lớn (CVPR / ICCV / IEEE T-ITS).

---

## 2. Cấu trúc Thư mục
- `roi_extractor.py`: `DeltaRoIExtractor` định vị và cắt phương tiện tự động từ bản đồ $\Delta$.
- `models.py`: `VehicleReIDModel` (ViT Backbone + BNNeck + $L_2$ Normalization).
- `losses.py`: `TrackletContrastiveLoss` (Hàm mất mát tương phản trên chuỗi tracklet).
- `matcher.py`: `VehicleReIDMatcher` (So khớp cosine, truy vấn Top-$k$, tính CMC Rank-1, Rank-5 và mAP).
- `run_reid.py`: Pipeline thực thi toàn bộ luồng từ trích xuất xe, tạo Gallery, đến tìm kiếm và lưu ảnh trực quan hóa.

---

## 3. Hướng Dẫn Chạy (CLI Execution)

```bash
python direction6_vehicle_reid/run_reid.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --output_dir checkpoints/direction6_vehicle_reid \
    --backbone dinov3_vits16 \
    --top_k 5 \
    --min_area 500 \
    --device cuda
```

### Kết quả đầu ra:
- `reid_retrieval_results.csv`: Danh sách các lượt truy vấn xe kèm điểm tương đồng và camera tương ứng.
- `visualizations/query_camX_vehY.png`: Ảnh hiển thị phương tiện Query bên trái và Top-5 phương tiện tương đồng nhất tìm thấy trong mạng lưới camera bên phải.
