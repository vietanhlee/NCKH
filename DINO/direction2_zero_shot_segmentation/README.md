# Hướng 2: Zero-Shot Vehicle Semantic Segmentation

## 1. Giới thiệu & Đóng góp Khoa học (Novelty ⭐⭐⭐)
Việc gán nhãn phân đoạn từng điểm ảnh (Pixel-level Semantic Segmentation) cho các luồng camera giao thông hỗn hợp (xe máy, ô tô, xe buýt san sát nhau) vô cùng tốn kém và đòi hỏi hàng trăm giờ lao động thủ công.

**Giải pháp đột phá không cần nhãn người (Zero-Shot):**
1. **Background Subtraction (Vật lý)**: Bắt trọn vẹn điểm ảnh chuyển động với độ phân giải cao nhưng dễ bị dính bóng đổ mặt đường.
2. **DINOv3 Emergent PCA (Ngữ nghĩa)**: Khai thác tính chất ngữ nghĩa không gian của Foundation Model — thành phần chính thứ nhất (PC1) tự động tách cấu trúc vật thể mà không bị ảnh hưởng bởi bóng đổ.
3. **Adaptive Fusion Engine & Edge Snapping**: Hợp nhất có trọng số giữa $\Delta$-mask và PCA PC1, sử dụng Guided Filter bám sát viền xe máy và thân ô tô.
4. **Lightweight Student Segmentor**: Huấn luyện một đầu giải mã nhẹ (Deconv Decoder) trên tập pseudo-labels này để chạy suy luận thời gian thực (>60 FPS) mà không cần ảnh background lúc triển khai thực tế.

---

## 2. Cấu trúc File trong Thư mục
- `pca_extractor.py`: `DINOPCAExtractor` (Trích xuất patch tokens từ DINOv3, chạy PCA 3 chiều sinh PC1/PC2/PC3).
- `fusion.py`: `MaskFusionEngine` (Khử bóng đổ, Guided Filter bám biên xe sắc nét).
- `segmentor.py`: `VehicleSegmentor`, `LightweightSegDecoder`, `DiceLoss`.
- `run_segmentation.py`: Pipeline CLI hoàn chỉnh tự động quét cặp ảnh, trích xuất pseudo-masks và xuất biểu đồ khoa học 300 DPI (PNG & PDF).

---

## 3. Hướng Dẫn Chạy Chi Tiết (CLI Execution)

### Kịch bản 1: Chạy thử trích xuất 10 ảnh mẫu & Xuất biểu đồ bài báo
Dùng để kiểm tra trực quan chất lượng đường viền mask trên một vài camera:
```bash
python DINO/direction2_zero_shot_segmentation/run_segmentation.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --output_dir checkpoints/direction2_zero_shot_seg/demo \
    --max_samples 10 \
    --max_vis 10 \
    --device cuda
```
*Đầu ra*: Mở thư mục `checkpoints/direction2_zero_shot_seg/demo/visualizations/` để xem ảnh so sánh 6 ô (Background, Origin, Delta, PCA components, Fused Mask, Overlay).

---

### Kịch bản 2: Chạy hàng loạt trích xuất Pseudo-Labels toàn bộ Dataset
Dùng để tự động tạo bộ nhãn phân đoạn cho toàn bộ hàng nghìn ảnh CCTV:
```bash
python DINO/direction2_zero_shot_segmentation/run_segmentation.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --output_dir checkpoints/direction2_zero_shot_seg/full_run \
    --max_samples 5000 \
    --max_vis 50 \
    --device cuda
```

---

## 4. Cấu Trúc Kết Quả Đầu Ra (Output Structure)

```
checkpoints/direction2_zero_shot_seg/
├── pseudo_masks/                       # Thư mục chứa Mask nhị phân (PNG uint8 0/255)
│   ├── 1_1755698811_mask.png           # Kích thước trùng khớp với ảnh gốc
│   └── ...
└── visualizations/                     # Biểu đồ khoa học chất lượng cao
    ├── 1_1755698811_segmentation.png   # Độ phân giải 300 DPI phục vụ chèn vào bài báo
    ├── 1_1755698811_segmentation.pdf   # Định dạng Vector PDF cho Overleaf / LaTeX
    └── ...
```

---

## 5. Bảng Giải Thích Tham Số (Arguments)

| Tham Số | Mặc Định | Ý Nghĩa Kỹ Thuật |
|:---|:---:|:---|
| `--bg_dir` | `traffic_backgrounds` | Thư mục chứa ảnh background theo tuyến đường (`route_stt/background_slot_XXh.jpg`). |
| `--origin_dir` | `output` | Thư mục chứa ảnh có phương tiện (`{stt}_{timestamp}.jpg`). |
| `--match_strategy` | `route_hourly` | Chiến lược ghép cặp theo giờ và tuyến đường. |
| `--backbone` | `dinov3_vits16` | Foundation Vision Backbone (`dinov3_vits16`, `dinov2_vits14`). |
| `--img_size` | `224` | Độ phân giải đầu vào trích xuất đặc trưng của ViT. |
| `--max_samples` | `100` | Số lượng cặp ảnh tối đa cần xử lý (None để chạy toàn bộ). |
| `--max_vis` | `10` | Số lượng ảnh mẫu xuất biểu đồ trực quan 6 ô (tránh tốn bộ nhớ đĩa). |
| `--device` | `cuda` | Thiết bị tính toán (`cuda` hoặc `cpu`). |
