# Hướng 1: BG-Guided DINO (Background-Guided Self-Supervised Learning)

## 1. Giới thiệu & Đóng góp Khoa học (Novelty ⭐⭐⭐⭐)
Trong các phương pháp tự giám sát (SSL) truyền thống như Meta DINOv2/v3 hay MAE, cơ chế che mờ là **Random Patch Masking**. Đối với camera giao thông cố định, hơn 70% khung hình là mặt đường và nhà cửa tĩnh. Việc che ngẫu nhiên khiến mô hình lãng phí phần lớn năng lực tính toán để tái tạo lại mặt đường vô nghĩa.

**Đóng góp cốt lõi:**
- **Foreground-Aware Masking (FAM)**: Khai thác bản đồ sai khác vật lý $\Delta = |I_{\text{origin}} - I_{\text{bg}}|$ để điều hướng xác suất che mờ tập trung vào các patch chứa xe cộ:
  $$w_p = \text{Mean}(\Delta[\text{patch}_p]), \quad P_{\text{mask}}(p) = \alpha \frac{w_p}{\max(w)} + (1-\alpha) \frac{1}{N}$$
- Ép Student ViT phải suy luận chi tiết cấu trúc xe, chủng loại xe và mật độ giao thông từ ngữ cảnh không gian xung quanh.
- Sử dụng cơ chế Teacher Centering & Sharpening chống sụp đổ biểu diễn (Mode Collapse).

---

## 2. Cấu trúc File trong Thư mục
- `dataset.py`: `BGGuidedDINODataset`, `MultiCropBGGuidedAugmentation` (sinh 2 Global views 224x224 + 4 Local views 96x96 và Patch Masks).
- `models.py`: `BGGuidedDINOModel`, `DINOHead` (Student, Teacher EMA, Prototype Projection Head với L2-normalization).
- `losses.py`: `BGGuidedDINOLoss` (Cross-Entropy chưng cất tự thân với Teacher Centering và Sharpening).
- `train.py`: Quy trình huấn luyện chuẩn Production (Tự động đa GPU, Cosine LR, EMA momentum, AMP mixed precision, checkpointing).

---

## 3. Hướng Dẫn Chạy Chi Tiết (CLI Execution)

### Kịch bản 1: Chạy thử nghiệm nhanh (Quick Test / Smoke Test)
Dùng để kiểm tra dữ liệu và pipeline chạy thông suốt (chỉ chạy 30 mẫu, 2 epochs):
```bash
python DINO/direction1_bg_guided_dino/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --epochs 2 \
    --batch_size 4 \
    --max_samples 30 \
    --save_dir checkpoints/direction1_bg_dino/test_run \
    --device cuda
```
*(Nếu máy không có GPU, thêm `--device cpu`)*.

---

### Kịch bản 2: Huấn luyện đầy đủ trên Đa GPU (Kaggle 2x T4 / Server Lab)
Hệ thống **tự động phát hiện toàn bộ số lượng GPU**, bọc `nn.DataParallel`, tự động nhân đôi/nhân bốn tổng Batch Size và scale Learning Rate:
```bash
python DINO/direction1_bg_guided_dino/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --epochs 20 \
    --batch_size 16 \
    --lr 2e-4 \
    --alpha_fg 0.75 \
    --mask_ratio 0.5 \
    --local_crops 4 \
    --save_dir checkpoints/direction1_bg_dino \
    --save_every 5 \
    --device cuda
```
*Lưu ý:* `--batch_size 16` là batch size trên **mỗi GPU**. Khi chạy trên Kaggle 2x T4, DataLoader sẽ tự động xử lý tổng batch size là $32$.

---

### Kịch bản 3: Huấn luyện trên Đơn GPU (Local RTX 3060/3090/4090)
```bash
python DINO/direction1_bg_guided_dino/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --match_strategy route_hourly \
    --backbone dinov2_vits14 \
    --epochs 15 \
    --batch_size 16 \
    --lr 2e-4 \
    --save_dir checkpoints/direction1_bg_dino \
    --device cuda
```

---

## 4. Bảng Giải Thích Tham Số (Arguments)

| Tham Số | Mặc Định | Ý Nghĩa Kỹ Thuật |
|:---|:---:|:---|
| `--bg_dir` | `traffic_backgrounds` | Thư mục chứa ảnh background (chứa subfolder `route_stt/background_slot_XXh.jpg`). |
| `--origin_dir` | `output` | Thư mục chứa ảnh origin cần ghép cặp (`{stt}_{timestamp}.jpg`). |
| `--match_strategy` | `route_hourly` | Chiến lược ghép cặp: `route_hourly` (ghép theo STT tuyến đường và slot giờ), `same_name`, `camera_id`. |
| `--backbone` | `dinov3_vits16` | Foundation backbone (`dinov3_vits16`, `dinov3_vitb16`, `dinov2_vits14`). |
| `--alpha_fg` | `0.75` | Mức độ thiên vị che vùng xe cộ ($0.0$: che ngẫu nhiên, $1.0$: chỉ che vùng có xe). Khuyến nghị $0.7 - 0.8$. |
| `--mask_ratio` | `0.5` | Tỷ lệ diện tích patch bị che mờ trong Student view (mặc định 50%). |
| `--batch_size` | `16` | Batch size trên mỗi GPU (hệ thống tự nhân với số lượng GPU). |
| `--lr` | `2e-4` | Tốc độ học cực đại của Optimizer AdamW (tự scale theo số GPU). |
| `--save_dir` | `checkpoints/...` | Thư mục lưu checkpoint mô hình (tự động dọn sạch tiền tố `module.`). |
| `--save_every` | `5` | Chu kỳ lưu checkpoint định kỳ (epochs). |
