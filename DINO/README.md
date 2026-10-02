# DINO Traffic Suite: Khai Thác Cặp Ảnh Background–Origin Cho Thị Giác Giao Thông

Bộ công cụ nghiên cứu toàn diện khai thác tín hiệu tự giám sát vật lý từ cặp ảnh **Background (nền tĩnh)** và **Origin (có phương tiện)** từ hệ thống camera giám sát đô thị (IC4SD-Traffic-HCM).

```
DINO/
├── common/                                 # Module dùng chung (Matcher, Subtraction, Backbone Loader, Multi-GPU)
│   ├── matcher.py                          # Ghép cặp origin (stt_timestamp) và background (route_stt/slot_XXh)
│   ├── subtraction.py                      # Trừ nền LAB/HSV/RGB, Otsu, Morphology, Patch Probability
│   ├── backbone_loader.py                  # Tải DINOv3, DINOv2, trích xuất CLS & Patch Tokens
│   └── gpu_utils.py                        # Quản lý DataParallel, auto-scaling, unwrap checkpoint
│
├── direction1_bg_guided_dino/              # [HƯỚNG 1] BG-Guided DINO Continual SSL Pre-training
│   ├── dataset.py                          # Multi-Crop + Foreground-Aware Masking (FAM)
│   ├── models.py                           # Student, Teacher EMA, DINOHead Prototype Projection
│   ├── losses.py                           # Self-Distillation Loss với Centering & Sharpening
│   ├── train.py                            # CLI huấn luyện Continual SSL
│   └── README.md
│
├── direction2_scene_decomposition/         # [HƯỚNG 2] Self-Supervised Scene Decomposition (Novelty ⭐⭐⭐⭐⭐)
│   ├── dataset.py                          # Dataset đồng bộ không gian Origin & Background
│   ├── models.py                           # TrafficDecompositionNet (Alpha Compositing)
│   ├── losses.py                           # Reconstruction + Background Supervision + Sparsity + TV Loss
│   ├── train.py                            # Pipeline huấn luyện phân rã cảnh
│   ├── infer.py                            # Inpainting tự động: Xóa xe, tách nền đường sạch
│   └── README.md
│
├── direction3_foreground_enhanced_counting/# [HƯỚNG 3] Foreground-Enhanced Counting (Stage 1 Upgrade)
│   ├── dataset.py                          # Nạp CSV nhãn, tính Δ map, tạo tensor 4 kênh (RGB + Δ)
│   ├── models.py                           # DINOv3 4-Channel Patch Embedding & MLP Regressor
│   ├── train.py                            # Huấn luyện đếm xe chống rò rỉ dữ liệu (Disjoint Cameras)
│   ├── evaluate.py                         # So sánh 3-ch vs 4-ch, tự động sinh bảng LaTeX bài báo
│   └── README.md
│
├── direction5_temporal_density/            # [HƯỚNG 5] Spatio-Temporal Density & HCM LoS Estimation
│   ├── dataset.py                          # Nạp chuỗi thời gian, mỏ neo vật lý ρ_phys, phân loại HCM LoS
│   ├── models.py                           # SpatioTemporalDensityModel (DINO + Delta-CNN + Bi-GRU)
│   ├── losses.py                           # TemporalDensityMultiTaskLoss (Smooth L1 + LoS CE + Smoothness)
│   ├── train.py                            # Huấn luyện đa nhiệm không-thời gian với AMP & Multi-GPU
│   └── README.md
│
├── direction6_vehicle_reid/                # [HƯỚNG 6] Corridor-Based Vehicle Re-ID Across Cameras
│   ├── roi_extractor.py                    # DeltaRoIExtractor tự động cắt xe không cần Object Detector
│   ├── models.py                           # VehicleReIDModel với GeM Pooling và BNNeck (256-D L2)
│   ├── losses.py                           # TrackletContrastiveLoss (học danh tính tự giám sát)
│   ├── matcher.py                          # VehicleReIDMatcher ràng buộc không-thời gian [v_min, v_max], tính km/h
│   ├── run_reid.py                         # Pipeline truy vấn phương tiện và đo thời gian hành trình
│   └── README.md
│
├── direction7_open_vocabulary/             # [HƯỚNG 7] Open-Vocabulary Traffic Scene Understanding
│   ├── proposal_engine.py                  # DeltaProposalEngine sinh đề xuất vùng đối tượng qua NMS
│   ├── text_prompts.py                     # Quản lý từ vựng văn bản mở tiếng Việt / tiếng Anh
│   ├── models.py                           # OpenVocabTrafficDetector chiếu DINO sang CLIP text space
│   ├── pipeline.py                         # Pipeline nhận diện từ vựng mở & vẽ bounding boxes
│   └── README.md
│
└── direction8_road_condition/              # [HƯỚNG 8] Self-Supervised Road Surface Condition Estimation
    ├── dataset.py                          # RoadSurfaceDataset nạp chuỗi ảnh nền 24h & chỉ số quang học
    ├── models.py                           # RoadConditionClassifier (Head: Wetness, Illumination, Degradation)
    ├── losses.py                           # SurfaceConsistencyLoss với mỏ neo vật lý
    ├── eval.py                             # Đánh giá, gom cụm PCA và xuất báo cáo hạ tầng đô thị
    └── README.md
```

## Bảng So Sánh 7 Hướng Nghiên Cứu

| Hướng | Tên Nghiên Cứu | Thư Mục | Cơ Chế Cốt Lõi | Venue Đề Xuất |
|:---|:---|:---|:---|:---|
| **H1** | **BG-Guided DINO Continual SSL** | `direction1_bg_guided_dino/` | Foreground-Aware Masking (FAM) ép ViT học biểu diễn xe cộ thay vì nền vô nghĩa | IEEE T-ITS, EAAI |
| **H2** | **Scene Decomposition Network** | `direction2_scene_decomposition/` | Alpha Compositing tự giám sát với mỏ neo nền thật (Road Inpainting) | CVPR, ECCV, NeurIPS |
| **H3** | **Foreground-Enhanced Counting** | `direction3_foreground_enhanced_counting/` | Mở rộng Patch Embed 4 kênh (RGB+$\Delta$) kết hợp Warm-Start | EAAI Journal, ITSC |
| **H5** | **Spatio-Temporal Density & HCM LoS** | `direction5_temporal_density/` | Hợp nhất DINO + $\Delta$-CNN + BiGRU với mỏ neo $\rho_{\text{phys}}$ dự đoán mật độ & LoS | IEEE T-ITS, CVPR |
| **H6** | **Corridor-Based Vehicle Re-ID** | `direction6_vehicle_reid/` | $\Delta$-RoI không cần detector, GeM+BNNeck, Ràng buộc không-thời gian đo tốc độ $km/h$ | IEEE T-ITS, TRB |
| **H7** | **Open-Vocabulary Scene Understanding** | `direction7_open_vocabulary/` | Delta proposals không phụ thuộc lớp kết hợp căn chỉnh DINO sang CLIP text | ECCV, WACV |
| **H8** | **Road Surface Condition Estimation** | `direction8_road_condition/` | Đánh giá đa thuộc tính mặt đường (đọng nước, chiếu sáng, hư hại) từ ảnh nền 24h | IEEE T-ITS, TRB |

---

## ⚡ Cơ Chế Tự Động Nhận Diện & Huấn Luyện Đa GPU (Multi-GPU Engine)

Tất cả các pipeline huấn luyện đều được tích hợp module `common/gpu_utils.py` tự động tối ưu hóa phần cứng:
- **Tự động đếm GPU (`torch.cuda.device_count()`)**: Nếu môi trường có 2x T4 (Kaggle) hoặc 4x/8x GPU (Server), hệ thống sẽ tự động bọc `nn.DataParallel` để phân phối tính toán song song trên TẤT CẢ các GPU cùng lúc.
- **Tự động mở rộng Batch Size (Linear Batch Scaling)**: $\text{Total Batch Size} = \text{batch\_size\_per\_gpu} \times N_{\text{gpus}}$.
- **Tự động điều chỉnh Tốc độ học (Linear LR Scaling Rule)**: $\text{Effective LR} = \text{base\_lr} \times N_{\text{gpus}}$.
- **Lưu Checkpoint an toàn**: Tự động giải phóng lớp bọc `module.` qua hàm `unwrap_model()`, giúp weights tương thích hoàn toàn khi load lại ở môi trường 1 GPU hoặc CPU.
- **Resume Training đầy đủ**: Tải lại trọn vẹn trạng thái huấn luyện cũ (`model`, `optimizer`, `scaler`, `epoch`, `best_metric`) qua cờ `--resume_checkpoint <path>` để tiếp tục train không bị gián đoạn.

---

## 🚀 BẢNG TỔNG HỢP LỆNH CHẠY NHANH (PIPELINE CHEAT SHEET)

### 1. Hướng 1: BG-Guided DINO Continual SSL
```bash
python DINO/direction1_bg_guided_dino/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --epochs 20 \
    --batch_size 16 \
    --alpha_fg 0.75 \
    --save_dir checkpoints/direction1_bg_dino \
    --device cuda
```

### 2. Hướng 2: Self-Supervised Scene Decomposition (Traffic-Decompose)
```bash
# Huấn luyện phân rã cảnh (Mặt đường, Xe, Alpha Mask):
python DINO/direction2_scene_decomposition/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --save_dir checkpoints/direction2_scene_decomp \
    --img_size 256 \
    --epochs 20 \
    --batch_size 8 \
    --device cuda

# Suy luận xóa xe tự động (Unsupervised Road Inpainting):
python DINO/direction2_scene_decomposition/infer.py \
    --weights checkpoints/direction2_scene_decomp/best_decomposition_model.pth \
    --input_path output \
    --output_dir checkpoints/direction2_scene_decomp/inferred \
    --device cuda
```

### 3. Hướng 3: Foreground-Enhanced Counting (Stage 1 Upgrade)
```bash
# Huấn luyện mô hình 4 kênh (RGB + Δ) với Spatial Disjoint Camera Split:
python DINO/direction3_foreground_enhanced_counting/train.py \
    --csv_file stage1_perception/counting_labels_5012.csv \
    --origin_dir output \
    --bg_dir traffic_backgrounds \
    --mode 4channel \
    --epochs 20 \
    --batch_size 16 \
    --save_dir checkpoints/direction3_fg_counting \
    --device cuda

# Chạy đánh giá đối đầu và xuất bảng LaTeX chèn vào bài báo:
python DINO/direction3_foreground_enhanced_counting/evaluate.py \
    --csv_file stage1_perception/counting_labels_5012.csv \
    --origin_dir output \
    --bg_dir traffic_backgrounds \
    --weights checkpoints/direction3_fg_counting/best_counting_model_4channel.pth \
    --output_dir checkpoints/direction3_fg_counting/eval \
    --device cuda
```

### 5. Hướng 5: Spatio-Temporal Density & HCM LoS Estimation
```bash
python DINO/direction5_temporal_density/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --seq_len 4 \
    --batch_size 8 \
    --epochs 30 \
    --save_dir checkpoints/direction5_temporal_density \
    --device cuda
```

### 6. Hướng 6: Corridor-Based Vehicle Re-ID & Travel Time Estimation
```bash
python DINO/direction6_vehicle_reid/run_reid.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --output_dir checkpoints/direction6_vehicle_reid \
    --top_k 5 \
    --min_speed 10.0 \
    --max_speed 60.0 \
    --min_area 500 \
    --device cuda
```

### 7. Hướng 7: Open-Vocabulary Traffic Scene Understanding
```bash
python DINO/direction7_open_vocabulary/pipeline.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --output_dir checkpoints/direction7_open_vocabulary \
    --language vi \
    --device cuda
```

### 8. Hướng 8: Self-Supervised Road Surface Condition Estimation
```bash
python DINO/direction8_road_condition/eval.py \
    --bg_dir traffic_backgrounds \
    --output_dir checkpoints/direction8_road_condition \
    --device cuda
```

---

## 🧪 Kiểm Thử Toàn Bộ 7 Hướng (Smoke Test Runner)
Chạy script kiểm thử tự động toàn diện với dữ liệu mô phỏng trong vòng 10 giây:
```bash
python DINO/test_all_directions.py
```
