# DINO Traffic Suite: Khai Thác Cặp Ảnh Background–Origin Cho Thị Giác Giao Thông

Bộ công cụ nghiên cứu toàn diện khai thác tín hiệu tự giám sát vật lý từ cặp ảnh **Background (nền tĩnh)** và **Origin (có phương tiện)** từ hệ thống camera giám sát đô thị (IC4SD-Traffic-HCM). Hệ thống gồm **8 hướng nghiên cứu trọng tâm** (Hướng 1 đến Hướng 8).

```
DINO/
├── common/                                 # Module dùng chung (Matcher, Subtraction, Backbone Loader, Multi-GPU, BDB, FCS, Reliability)
│   ├── matcher.py                          # Ghép cặp origin (stt_timestamp) và background (route_stt/slot_XXh)
│   ├── subtraction.py                      # Trừ nền LAB/HSV/RGB, Otsu, Morphology, Patch Probability
│   ├── backbone_loader.py                  # Tải DINOv3, DINOv2, trích xuất CLS & Patch Tokens, quản lý HF_TOKEN
│   ├── reliability.py                      # Ước lượng độ tin cậy vùng tĩnh r_i & Phase Correlation căn chỉnh camera
│   ├── degradation.py                      # Bộ suy thoái nền BDB (6 loại nhiễu x 5 mức độ nghiêm trọng)
│   ├── corrupt.py                          # Bộ hư hao khung hình FCS (8 loại hư hao ngoại cảnh thực tế)
│   └── gpu_utils.py                        # Quản lý DataParallel, auto-scaling, unwrap checkpoint
│
├── direction1_bg_guided_dino/              # [HƯỚNG 1] BG-Guided DINO Continual SSL Pre-training
│   ├── dataset.py                          # Multi-Crop + Foreground-Aware Masking (FAM)
│   ├── models.py                           # Student, Teacher EMA, DINOHead Prototype Projection
│   ├── losses.py                           # DINO [CLS] + iBOT [Patch] Multi-Crop Loss
│   ├── train.py                            # CLI huấn luyện Continual SSL
│   └── README.md
│
├── direction2_scene_decomposition/         # [HƯỚNG 2] Noise-Aware Scene Decomposition (Laplace σ Prior)
│   ├── dataset.py                          # Dataset đồng bộ không gian Origin & Background (K ngày khác nhau)
│   ├── models.py                           # Multi-scale ViT + DPT Head (M_alpha, F, B_hat, sigma)
│   ├── losses.py                           # Reconstruction + Laplace Prior + Cross-Day Shared Bg + Exclusion Loss
│   ├── train.py                            # Pipeline huấn luyện phân rã cảnh
│   ├── infer.py                            # Inpainting tự động: Xóa xe, tách nền đường sạch
│   └── README.md
│
├── direction3_foreground_enhanced_counting/# [HƯỚNG 3] Foreground-Enhanced Counting (Stage 1 Upgrade)
│   ├── dataset.py                          # Nạp CSV nhãn, tính Δ map, tạo tensor 4 kênh (RGB + Δ)
│   ├── models.py                           # DINOv3 4 cơ chế tiêm prior (Early, Late, FiLM, None) + Zero-init
│   ├── train.py                            # Huấn luyện đếm xe chống rò rỉ dữ liệu (Disjoint Cameras, Δ-Dropout)
│   ├── evaluate.py                         # So sánh đối đầu 4 cơ chế, tự động sinh bảng LaTeX bài báo
│   └── README.md
│
├── direction4_temporal_density/            # [HƯỚNG 4] Road-Space Occupancy & Causal Onset Forecasting
│   ├── dataset.py                          # Nạp chuỗi thời gian, mỏ neo strictly Road Mask ρ_proxy, phân loại LoS
│   ├── models.py                           # SpatioTemporalDensityModel (Bi-GRU Nowcasting & 1-way Causal GRU)
│   ├── losses.py                           # Huber Smoothness + Road Occupancy Loss + LoS Cross-Entropy
│   ├── train.py                            # Huấn luyện đa nhiệm không-thời gian với AMP & Multi-GPU
│   ├── eval.py                             # Đánh giá độc lập mô hình, trích xuất MAE và ma trận LoS
│   └── README.md
│
├── direction5_weak_supervision/            # [HƯỚNG 5] Context-Aware Markov Weak Supervision Label Aggregation
│   ├── lfs/                                # 5 nguồn nhãn yếu (Detector Box, Background Diff, Temporal, History, VLM)
│   ├── context.py                          # Bộ trích xuất 54 tổ hợp ngữ cảnh giao thông
│   ├── label_model.py                      # Context-Aware Markov Label Model (Forward-Backward Log-Sum-Exp EM)
│   ├── end_model.py                        # DINOv3 + Causal GRU End Model (Tự chủ 100% không cần background)
│   ├── evaluate.py                         # So sánh với Majority Vote, Dawid-Skene trên Gold Set
│   └── README.md
│
├── direction6_anomaly_detection/           # [HƯỚNG 6] Persistence-Aware Anomaly & Camera Fault Disentanglement
│   ├── features.py                         # Trích xuất patch token DINOv3 + Road-Aware Masking
│   ├── pooling.py                          # Temporal Feature Pooling (Median cửa sổ trượt W frames)
│   ├── bank.py                             # Coreset Normal Memory Bank (K-Center Greedy Selection nén 90%)
│   ├── score.py                            # Anomaly Scoring (k-NN L2 distance) & Patch Heatmap
│   ├── camera_fault.py                     # Phân tách Lỗi Camera (Static Anomaly) vs Sự cố Giao thông (Road Anomaly)
│   ├── events.py                           # Persistence Filtering & Quản lý vòng đời sự kiện
│   ├── synth_events.py                     # Bộ sinh sự cố tổng hợp có Ground Truth
│   ├── evaluate.py                         # Pipeline đánh giá định lượng (AUROC, Delay, False Alarms)
│   └── README.md
│
├── direction7_traffic_forecasting/         # [HƯỚNG 7] City-Scale Congestion Forecasting on Camera Graph
│   ├── graph.py                            # Xây dựng ma trận kề địa lý & Adaptive Adjacency Layer
│   ├── dataset.py                          # Lưới thời gian 5 phút, Missing mask, Chronological split
│   ├── models.py                           # ST-GraphWaveNet với Gated Dilated TCN & Multi-Task Heads
│   ├── losses.py                           # Masked MAE, Focal Onset Loss, Multi-task Loss
│   ├── evaluate.py                         # Đánh giá đa tầm h=15', 30', 60' & Khảo sát camera offline 10%-50%
│   └── README.md
│
└── direction8_bg_conditioning/             # [HƯỚNG 8] Generalization to Unseen Cameras via Trimmed Background Stats
    ├── descriptor.py                       # Trích xuất Trimmed Mean/Std Scene Descriptor z từ DINOv3
    ├── conditioning.py                     # FiLM (Zero-init), Scene Prompt Tokens, Cross-Attention
    ├── models.py                           # BackgroundConditionedModel + Bg-Dropout (p=0.25)
    ├── evaluate.py                         # Đánh giá Unseen Camera Generalization & BDB Stress Test
    └── README.md
```

---

## Bảng So Sánh 8 Hướng Nghiên Cứu Trọng Tâm

| Hướng | Tên Nghiên Cứu | Thư Mục | Cơ Chế Cốt Lõi | Venue Đề Xuất |
|:---|:---|:---|:---|:---|
| **H1** | **BG-Guided DINO Continual SSL** | `direction1_bg_guided_dino/` | Foreground-Aware Masking (FAM) ép ViT học biểu diễn xe cộ thay vì nền vô nghĩa | IEEE T-ITS, CVPR |
| **H2** | **Noise-Aware Scene Decomposition** | `direction2_scene_decomposition/` | Bóc tách cảnh tự giám sát với Laplace $\sigma$ Prior + Nền dùng chung đa ngày | IEEE TIP, Pattern Recognition |
| **H3** | **Foreground-Enhanced Counting** | `direction3_foreground_enhanced_counting/` | Mở rộng Patch Embed 4 kênh (RGB+$\Delta$) kết hợp Zero-init & $\Delta$-Dropout | IEEE T-ITS, EAAI |
| **H4** | **Road-Space Occupancy & Causal Onset** | `direction4_temporal_density/` | Định lượng chiếm dụng strictly trên Road Mask + Causal GRU cảnh báo sớm kẹt xe | TR-Part C, IEEE T-ITS |
| **H5** | **Context-Aware Weak Supervision** | `direction5_weak_supervision/` | Gộp nhãn yếu đa nguồn (5 LFs, 54 ngữ cảnh, Markov EM) & End Model độc lập | IEEE T-ITS, Information Fusion |
| **H6** | **Persistence Anomaly Detection** | `direction6_anomaly_detection/` | Temporal Feature Median Pooling + Phân tách lỗi camera vs sự cố đường | TR-Part C, Pattern Recognition |
| **H7** | **City-Scale Traffic Forecasting** | `direction7_traffic_forecasting/` | Mạng nơ-ron đồ thị ST-GraphWaveNet thích ứng với mất tín hiệu camera (Node Dropout) | TR-Part C, IEEE TKDE |
| **H8** | **Background Conditioning & Adaptation**| `direction8_bg_conditioning/` | Thích ứng camera chưa thấy qua Trimmed Descriptor $z$ và Zero-init FiLM Modulation | Pattern Recognition, EAAI |

---

## ⚡ Cơ Chế Tự Động Nhận Diện & Huấn Luyện Đa GPU (Multi-GPU Engine)

Tất cả các pipeline huấn luyện đều được tích hợp module `common/gpu_utils.py` tự động tối ưu hóa phần cứng:
- **Tự động đếm GPU (`torch.cuda.device_count()`)**: Nếu môi trường có 2x T4 (Kaggle) hoặc 4x/8x GPU (Server), hệ thống sẽ tự động bọc `nn.DataParallel` để phân phối tính toán song song trên TẤT CẢ các GPU cùng lúc.
- **Tự động mở rộng Batch Size (Linear Batch Scaling)**: $\text{Total Batch Size} = \text{batch\_size\_per\_gpu} \times N_{\text{gpus}}$.
- **Tự động điều chỉnh Tốc độ học (Linear LR Scaling Rule)**: $\text{Effective LR} = \text{base\_lr} \times N_{\text{gpus}}$.
- **Lưu Checkpoint an toàn**: Tự động giải phóng lớp bọc `module.` qua hàm `unwrap_model()`, giúp weights tương thích hoàn toàn khi load lại ở môi trường 1 GPU hoặc CPU.
- **Resume Training đầy đủ**: Tải lại trọn vẹn trạng thái huấn luyện cũ (`model`, `optimizer`, `scaler`, `epoch`, `best_metric`) qua cờ `--resume_checkpoint <path>` để tiếp tục train không bị gián đoạn.

---

## 🧪 Kiểm Thử Toàn Bộ 8 Hướng Trọng Tâm (Smoke Test Runner)
Chạy script kiểm thử tự động toàn diện với dữ liệu mô phỏng trong vòng 10 giây:
```bash
python DINO/test_all_directions.py
```
