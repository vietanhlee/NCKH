# Hướng 3: Foreground-Enhanced Vehicle Counting (Stage 1 Upgrade)

## 1. Giới thiệu & Đóng góp Khoa học (Novelty ⭐⭐)
Nâng cấp trực tiếp bài toán ước lượng lưu lượng phương tiện của Stage 1 trong bài báo khoa học hiện tại (`counting_labels_5012.csv`):
- **Vấn đề cốt lõi**: Các mô hình chuẩn (EfficientNet-B5, ResNet-50, DINOv3 Baseline) chỉ nhận đầu vào là ảnh **3 kênh RGB**. Góc camera CCTV đường phố có rất nhiều vùng thừa gây nhiễu thị giác (tán cây rung rinh trong gió, mây trời, nhà dân 2 bên đường).
- **Giải pháp đề xuất**:
  1. Mở rộng tầng Patch Embedding của DINOv3 ViT từ 3 kênh lên **4 kênh (RGB + $\Delta$)**:
     - Kênh thứ 4 là bản đồ sai khác $\Delta$ từ background subtraction, đóng vai trò như một **Spatial Prior** (tiền nghiệm không gian) dẫn hướng mô hình dồn 100% tài nguyên chú ý vào luồng giao thông.
     - 3 kênh đầu kế thừa 100% trọng số pretrained của Meta AI, kênh thứ 4 khởi tạo bằng trung bình cộng 3 kênh để đảm bảo tính ổn định đạo hàm ngay từ epoch đầu tiên.
  2. Giao thức đánh giá chuẩn Q1/Scopus:
     - **Spatial Disjoint Camera Split**: Nghiêm cấm rò rỉ dữ liệu camera giữa tập train và test.
     - **Few-Shot Counting Benchmark**: Đánh giá độ bền bỉ khi chỉ có 5%, 10%, 20% dữ liệu nhãn.
  3. Tự động sinh bảng kết quả định dạng **$\text{\LaTeX}$ chuẩn tạp chí IEEE / Elsevier EAAI**.

---

## 2. Cấu trúc File trong Thư mục
- `dataset.py`: `FGCountingDataset` (Nạp CSV nhãn, tìm background theo slot giờ, tính $\Delta$, ghép tensor 4 kênh).
- `models.py`: `DINOv3FGCountingModel`, `adapt_patch_embed_to_4ch`, `RegressionHead` 2-layer MLP (kích hoạt ReLU đảm bảo số lượng xe $\ge 0$).
- `train.py`: Pipeline huấn luyện có giám sát với Smooth L1 Loss, Early Stopping, tự động phân phối đa GPU.
- `evaluate.py`: Chạy benchmark đối đầu trực diện giữa 3-ch Baseline và 4-ch FG-Enhanced, tự động xuất bảng $\text{\LaTeX}$.

---

## 3. Hướng Dẫn Chạy Chi Tiết (CLI Execution)

### Kịch bản 1: Chạy thử nghiệm nhanh (Quick Test)
Kiểm tra pipeline nạp CSV và backward trên 20 mẫu trong 2 epochs:
```bash
python DINO/direction3_foreground_enhanced_counting/train.py \
    --csv_file stage1_perception/counting_labels_5012.csv \
    --origin_dir output \
    --bg_dir traffic_backgrounds \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --mode 4channel \
    --epochs 2 \
    --batch_size 4 \
    --save_dir checkpoints/direction3_fg_counting/test_run \
    --device cuda
```

---

### Kịch bản 2: Huấn luyện đầy đủ trên Đa GPU (Kaggle 2x T4 / Server Lab)
Hệ thống **tự động phát hiện toàn bộ số GPU**, chia đều batch qua tất cả các GPU:
```bash
python DINO/direction3_foreground_enhanced_counting/train.py \
    --csv_file stage1_perception/counting_labels_5012.csv \
    --origin_dir output \
    --bg_dir traffic_backgrounds \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --mode 4channel \
    --epochs 20 \
    --batch_size 16 \
    --lr 2e-4 \
    --img_size 224 \
    --few_shot_ratio 1.0 \
    --save_dir checkpoints/direction3_fg_counting \
    --device cuda
```

---

### Kịch bản 3: Thí nghiệm Few-Shot Benchmark (Khi khan hiếm nhãn)
Thử nghiệm huấn luyện mô hình khi chỉ có **10% hoặc 20% nhãn** để chứng minh kênh $\Delta$ giúp mô hình học nhanh hơn baseline:
```bash
# Huấn luyện chỉ với 10% dữ liệu nhãn (Few-shot 10%):
python DINO/direction3_foreground_enhanced_counting/train.py \
    --csv_file stage1_perception/counting_labels_5012.csv \
    --origin_dir output \
    --bg_dir traffic_backgrounds \
    --mode 4channel \
    --few_shot_ratio 0.10 \
    --epochs 15 \
    --save_dir checkpoints/direction3_fg_counting/fewshot_10pct \
    --device cuda
```

---

### Kịch bản 4: Chạy Đánh Giá So Sánh Đối Đầu & Xuất Bảng LaTeX
Chạy script kiểm thử đối đầu trên tập Test Disjoint và sinh mã $\text{\LaTeX}$ để chèn trực tiếp vào Overleaf/Paper:
```bash
python DINO/direction3_foreground_enhanced_counting/evaluate.py \
    --csv_file stage1_perception/counting_labels_5012.csv \
    --origin_dir output \
    --bg_dir traffic_backgrounds \
    --weights checkpoints/direction3_fg_counting/best_counting_model_4channel.pth \
    --mode 4channel \
    --output_dir checkpoints/direction3_fg_counting/eval \
    --device cuda
```
*Kết quả đầu ra*: Mở file `checkpoints/direction3_fg_counting/eval/benchmark_comparison_table.tex` để lấy mã nguồn bảng LaTeX.

---

## 4. Bảng Giải Thích Tham Số (Arguments)

| Tham Số | Mặc Định | Ý Nghĩa Kỹ Thuật |
|:---|:---:|:---|
| `--csv_file` | `counting_labels_5012.csv` | File CSV chứa nhãn ground-truth (`filename, xe_may, o_to, tong`). |
| `--mode` | `4channel` | Chế độ mô hình: `4channel` (RGB + $\Delta$) hoặc `spatial_attention` (Attention Prior). |
| `--few_shot_ratio` | `1.0` | Tỷ lệ nhãn huấn luyện ($0.05$: 5%, $0.10$: 10%, $0.20$: 20%, $1.0$: 100%). |
| `--freeze_backbone` | `False` | Nếu True: Chỉ huấn luyện Regression Head (Linear Probe). Nếu False: Fine-tune toàn bộ mạng. |
| `--batch_size` | `16` | Batch size trên mỗi GPU (tự động nhân với số lượng GPU). |
| `--lr` | `2e-4` | Tốc độ học cực đại của Optimizer AdamW. |
