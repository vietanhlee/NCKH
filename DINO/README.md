# DINO Traffic Suite: Khai Thác Cặp Ảnh Background–Origin Cho Thị Giác Giao Thông

Bộ công cụ nghiên cứu toàn diện khai thác tín hiệu tự giám sát vật lý từ cặp ảnh **Background (nền tĩnh)** và **Origin (có phương tiện)** từ hệ thống camera giám sát đô thị (IC4SD-Traffic-HCM).

```
DINO/
├── common/                                 # Module dùng chung (Matcher, Subtraction, Backbone Loader)
│   ├── matcher.py                          # Ghép cặp origin (stt_timestamp) và background (route_stt/slot_XXh)
│   ├── subtraction.py                      # Trừ nền LAB/HSV/RGB, Otsu, Morphology, Patch Probability
│   └── backbone_loader.py                  # Tải DINOv3, DINOv2, trích xuất CLS & Patch Tokens
│
├── direction1_bg_guided_dino/              # [HƯỚNG 1] BG-Guided DINO Continual SSL Pre-training
│   ├── dataset.py                          # Multi-Crop + Foreground-Aware Masking (FAM)
│   ├── models.py                           # Student, Teacher EMA, DINOHead Prototype Projection
│   ├── losses.py                           # Self-Distillation Loss với Centering & Sharpening
│   ├── train.py                            # CLI huấn luyện Continual SSL
│   └── README.md
│
├── direction2_zero_shot_segmentation/      # [HƯỚNG 2] Zero-Shot Vehicle Semantic Segmentation
│   ├── pca_extractor.py                    # DINOv3 PCA Patch Feature decomposition (PC1/2/3)
│   ├── fusion.py                           # Fusion Engine: Δ-Mask ∩ PCA + Guided Filtering
│   ├── segmentor.py                        # Lightweight Student SegHead (>60 FPS)
│   ├── run_segmentation.py                 # Pipeline sinh Pseudo-Masks & đồ thị bài báo 300 DPI
│   └── README.md
│
├── direction3_scene_decomposition/         # [HƯỚNG 3] Self-Supervised Scene Decomposition (Novelty ⭐⭐⭐⭐⭐)
│   ├── dataset.py                          # Dataset đồng bộ không gian Origin & Background
│   ├── models.py                           # TrafficDecompositionNet (Alpha Compositing)
│   ├── losses.py                           # Reconstruction + Background Supervision + Sparsity + TV Loss
│   ├── train.py                            # Pipeline huấn luyện phân rã cảnh
│   ├── infer.py                            # Inpainting tự động: Xóa xe, tách nền đường sạch
│   └── README.md
│
└── direction4_foreground_enhanced_counting/# [HƯỚNG 4] Foreground-Enhanced Counting (Stage 1 Upgrade)
    ├── dataset.py                          # Nạp CSV nhãn, tính Δ map, tạo tensor 4 kênh (RGB + Δ)
    ├── models.py                           # DINOv3 4-Channel Patch Embedding & MLP Regressor
    ├── train.py                            # Huấn luyện đếm xe chống rò rỉ dữ liệu (Disjoint Cameras)
    ├── evaluate.py                         # So sánh 3-ch vs 4-ch, tự động sinh bảng LaTeX bài báo
    └── README.md
```

## Bảng So Sánh Các Hướng Nghiên Cứu

| Hướng | Tên Nghiên Cứu | Thư Mục | Mục Tiêu Cốt Lõi | Venue Phù Hợp |
|:---|:---|:---|:---|:---|
| **1** | **BG-Guided DINO** | `direction1_bg_guided_dino/` | Foreground-Aware Masking ép ViT học biểu diễn xe cộ thay vì nền đường vô nghĩa | IEEE T-ITS, EAAI |
| **2** | **Zero-Shot Segmentation**| `direction2_zero_shot_segmentation/`| Tự sinh nhãn phân đoạn pixel không cần con người qua DINOv3 PCA + $\Delta$ Fusion | Pattern Recognition, CVPR Workshop |
| **3** | **Scene Decomposition** | `direction3_scene_decomposition/` | Tách lớp vật lý (Mặt đường sạch xe + Xe cộ + Alpha Mask) dùng ảnh nền thật giám sát | CVPR, ECCV, NeurIPS |
| **4** | **FG-Enhanced Counting**| `direction4_foreground_enhanced_counting/`| Mở rộng Patch Embed 4 kênh (RGB+$\Delta$) cải thiện trực tiếp MAE đếm xe Stage 1 | EAAI (Bổ sung vào bài báo hiện tại) |

---

## ⚡ Cơ Chế Tự Động Nhận Diện & Huấn Luyện Đa GPU (Multi-GPU Engine)

Tất cả các pipeline huấn luyện (`direction1`, `direction3`, `direction4`) đều được tích hợp module `common/gpu_utils.py` tự động tối ưu hóa phần cứng:
- **Tự động đếm GPU (`torch.cuda.device_count()`)**: Nếu môi trường có 2x T4 (Kaggle) hoặc 4x/8x GPU (Server), hệ thống sẽ **tự động bọc `nn.DataParallel` để phân phối tính toán song song trên TẤT CẢ các GPU cùng lúc** (không chia GPU riêng cho từng task).
- **Tự động mở rộng Batch Size (Linear Batch Scaling)**: $\text{Total Batch Size} = \text{batch\_size\_per\_gpu} \times N_{\text{gpus}}$.
- **Tự động điều chỉnh Tốc độ học (Linear LR Scaling Rule)**: $\text{Effective LR} = \text{base\_lr} \times N_{\text{gpus}}$.
- **Lưu Checkpoint an toàn**: Tự động giải phóng lớp bọc `module.` qua hàm `unwrap_model()`, giúp weights lưu ra tương thích hoàn toàn khi load lại ở môi trường 1 GPU hoặc CPU.
- **Fallback an toàn**: Chạy đơn GPU nếu chỉ có 1 GPU, hoặc chuyển sang CPU nếu không có CUDA.

---

## 🚀 BẢNG TỔNG HỢP LỆNH CHẠY NHANH TỪ A-Z (PIPELINE CHEAT SHEET)

### 0. Tiền xử lý & Chuẩn bị dữ liệu (Data Pipeline)

#### 0.1. Thu thập ảnh thời gian thực từ 608 Camera TP.HCM (Crawler Production)
```bash
# Chạy thu thập liên tục 24/7 (quét 608 camera mỗi 240 giây = 4 phút):
python DINO/crawl_traffic_cameras.py \
    --csv_path D:\DATN_transport-network-model\camera_data_608Cam.csv \
    --output_dir output \
    --interval_sec 240 \
    --workers 40

# Hoặc chạy kiểm tra 1 chu kỳ duy nhất (single run):
python DINO/crawl_traffic_cameras.py --single_run --workers 40
```

#### 0.2. Tự động lọc rác và tạo ảnh Background tĩnh (Temporal Median)
```bash
# Lọc ảnh lỗi 284x177 và tạo background theo slot giờ (chuẩn múi giờ UTC+7):
python DINO/generate_backgrounds.py \
    --input_dir output \
    --output_dir traffic_backgrounds \
    --time_interval_hours 1 \
    --max_samples 30 \
    --workers 8
```

---

### 1. Hướng 1: BG-Guided DINO Continual SSL
```bash
# Huấn luyện SSL (Tự nhận diện 1 GPU hoặc Multi-GPU):
python DINO/direction1_bg_guided_dino/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --epochs 20 \
    --batch_size 16 \
    --lr 2e-4 \
    --alpha_fg 0.75 \
    --save_dir checkpoints/direction1_bg_dino \
    --device cuda
```

### 2. Hướng 2: Zero-Shot Vehicle Semantic Segmentation
```bash
# Trích xuất Pseudo-Masks & Đồ thị báo cáo khoa học 300 DPI:
python DINO/direction2_zero_shot_segmentation/run_segmentation.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --output_dir checkpoints/direction2_zero_shot_seg \
    --max_samples 500 \
    --max_vis 20 \
    --device cuda
```

### 3. Hướng 3: Self-Supervised Scene Decomposition (Traffic-Decompose)
```bash
# a. Huấn luyện phân rã cảnh (Mặt đường, Xe, Alpha Mask):
python DINO/direction3_scene_decomposition/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --img_size 256 \
    --epochs 20 \
    --batch_size 8 \
    --lr 3e-4 \
    --save_dir checkpoints/direction3_scene_decomp \
    --device cuda

# b. Suy luận xóa xe tự động (Unsupervised Road Inpainting):
python DINO/direction3_scene_decomposition/infer.py \
    --weights checkpoints/direction3_scene_decomp/best_decomposition_model.pth \
    --input_path output \
    --output_dir checkpoints/direction3_scene_decomp/inferred \
    --device cuda
```

### 4. Hướng 4: Foreground-Enhanced Counting (Stage 1 Upgrade)
```bash
# a. Huấn luyện mô hình 4 kênh (RGB + Δ) với Spatial Disjoint Camera Split:
python DINO/direction4_foreground_enhanced_counting/train.py \
    --csv_file stage1_perception/counting_labels_5012.csv \
    --origin_dir output \
    --bg_dir traffic_backgrounds \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --mode 4channel \
    --epochs 20 \
    --batch_size 16 \
    --lr 2e-4 \
    --save_dir checkpoints/direction4_fg_counting \
    --device cuda

# b. Chạy đánh giá đối đầu và xuất bảng LaTeX chèn vào bài báo:
python DINO/direction4_foreground_enhanced_counting/evaluate.py \
    --csv_file stage1_perception/counting_labels_5012.csv \
    --origin_dir output \
    --bg_dir traffic_backgrounds \
    --weights checkpoints/direction4_fg_counting/best_counting_model_4channel.pth \
    --mode 4channel \
    --output_dir checkpoints/direction4_fg_counting/eval \
    --device cuda
```

---

## 🧪 Kiểm Thử Toàn Bộ Suite (Smoke Test Runner)
Chạy script kiểm thử tự động toàn diện với dữ liệu mô phỏng trong vòng 10 giây:
```bash
python DINO/test_all_directions.py
```
