# Hướng 5: Self-Supervised Road Surface Condition Estimation

## 1. Giới thiệu & Đóng góp Khoa học (Novelty ⭐⭐⭐⭐)
Mỗi camera giám sát giao thông đô thị ghi nhận chuỗi ảnh nền 24 giờ $\{I_{\text{bg}}^{h=0}, \dots, I_{\text{bg}}^{h=23}\}$. Thay vì chỉ sử dụng ảnh nền để trừ xe, chuỗi ảnh này chứa đựng các biến đổi tự nhiên của môi trường:
- Tình trạng thời tiết: Mặt đường khô ráo vs. mặt đường ẩm ướt / đọng nước sau mưa (thể hiện qua các phản chiếu gương - Specular Reflections).
- Chu kỳ chiếu sáng: Ánh sáng ban ngày, ánh hoàng hôn và hệ thống đèn đường ban đêm.
- Tình trạng hạ tầng: Mức độ nứt nẻ, gồ ghề, ổ gà hoặc xuống cấp của bề mặt nhựa đường.

**Giải pháp đề xuất**:
1. **Self-Supervised Surface Representation**: Dùng DINO ViT CLS token để trích xuất vector đặc trưng kết cấu mặt đường $\mathbf{z}_{\text{road}}$.
2. **Multi-Attribute Heads**: Dự đoán đồng thời chỉ số đọng nước (Wetness Index), lớp chiếu sáng (Day/Dusk/Night) và chỉ số suy giảm kết cấu đường (Degradation Index).
3. **Phân tích toàn đô thị & Trực quan hóa Gom cụm**: Gom cụm PCA 2D và sinh báo cáo tự động tình trạng hạ tầng cho toàn bộ mạng lưới camera.

---

## 2. Cấu trúc Thư mục
- `dataset.py`: `RoadSurfaceDataset` quét toàn bộ ảnh nền 24h, tính toán các chỉ số quang học cơ sở (độ chói, tỷ lệ phản chiếu gương, độ nhám gradient).
- `models.py`: `RoadConditionClassifier` (ViT Backbone + Wetness, Illumination & Degradation Heads).
- `losses.py`: `SurfaceConsistencyLoss` (Hàm mất mát đa mục tiêu kết hợp mỏ neo vật lý).
- `train.py`: Pipeline huấn luyện đa thuộc tính mặt đường với Multi-GPU & AMP.
- `eval.py`: Pipeline đánh giá toàn đô thị, vẽ biểu đồ gom cụm PCA và xuất báo cáo CSV.

---

## 3. Hướng Dẫn Chạy (CLI Execution)

```bash
# Huấn luyện mô hình:
python direction5_road_condition/train.py \
    --bg_dir traffic_backgrounds \
    --save_dir checkpoints/direction5_road_condition \
    --backbone dinov3_vits16 \
    --epochs 15 \
    --device cuda

# Đánh giá & xuất báo cáo toàn đô thị:
python direction5_road_condition/eval.py \
    --bg_dir traffic_backgrounds \
    --output_dir checkpoints/direction5_road_condition \
    --backbone dinov3_vits16 \
    --device cuda
```

### Kết quả đầu ra:
- `road_condition_clustering.png`: Biểu đồ phân bố PCA 2D của không gian mặt đường theo chiếu sáng và độ ẩm ướt.
- `road_surface_citywide_report.csv`: Báo cáo chi tiết từng camera: camera nào đang bị ngập ướt, điều kiện sáng, và mức độ hư hại mặt đường.
