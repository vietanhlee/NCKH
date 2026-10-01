# Hướng 5: Temporal Contrastive Learning for Traffic Density Estimation

## 1. Giới thiệu & Đóng góp Khoa học (Novelty ⭐⭐⭐⭐)
Hầu hết các nghiên cứu thị giác giao thông chỉ xử lý khung hình độc lập (Frame-by-Frame). Trong hệ thống camera giám sát đô thị:
- Camera CCTV cung cấp luồng quan sát **chuỗi thời gian liên tục** $\{I^{t_1}, I^{t_2}, \dots, I^{t_T}\}$.
- Tốc độ di chuyển và mật độ dòng xe được mã hóa trực tiếp qua tốc độ biến thiên của trường sai khác quang học $\Delta(u, v, t)$.

**Phương pháp đề xuất**:
1. **Temporal Contrastive Framework**: Nhóm các khung hình theo cửa sổ thời gian $K$. Hai cửa sổ liền kề nhau $(W_a, W_b^+)$ từ cùng một camera tạo thành cặp dương tự nhiên.
2. **Đồng mã hóa Không gian - Thời gian**:
   - Từng khung hình được nén qua DINO ViT CLS token + bản đồ sai khác $\Delta$.
   - Module **Temporal Self-Attention** tổng hợp động học thời gian qua $K$ frame thành vector trạng thái luồng xe $\mathbf{h} \in \mathbb{R}^{256}$.
3. **Ứng dụng**: Ước lượng mật độ dòng xe và phát hiện tắc nghẽn giao thông mà không cần nhãn thủ công hoặc chỉ cần một lượng rất nhỏ nhãn giám sát ít mẫu (Few-shot).

---

## 2. Cấu trúc File
- `dataset.py`: `TemporalTrafficDataset` gom nhóm camera theo timestamp, sinh cửa sổ thời gian $K$ và cặp dương liền kề.
- `models.py`: `TemporalTrafficEncoder` (ViT Backbone + DeltaSpatialEncoder + Temporal Self-Attention + Projection & Density Heads).
- `losses.py`: `TemporalContrastiveLoss` (InfoNCE loss cho tương phản thời gian + Smooth L1 cho giám sát mật độ).
- `train.py`: Pipeline huấn luyện hoàn chỉnh với AMP, Cosine Scheduler, checkpointing.

---

## 3. Hướng Dẫn Chạy (CLI Execution)

### Kịch bản 1: Huấn luyện nhanh trên dữ liệu mẫu
```bash
python direction5_temporal_density/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --window_size 4 \
    --batch_size 2 \
    --epochs 2 \
    --max_sequences 10 \
    --device cuda
```

### Kịch bản 2: Huấn luyện chính thức
```bash
python direction5_temporal_density/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --csv_file stage1_perception/counting_labels_5012.csv \
    --save_dir checkpoints/direction5_temporal_density \
    --backbone dinov3_vits16 \
    --window_size 4 \
    --batch_size 8 \
    --epochs 30 \
    --lr 3e-4 \
    --device cuda
```
