# DINO Traffic Suite: Khai Thác Cặp Ảnh Background–Origin Cho Thị Giác Giao Thông

Bộ công cụ nghiên cứu toàn diện khai thác tín hiệu tự giám sát vật lý từ cặp ảnh **Background (nền tĩnh)** và **Origin (có phương tiện)** từ hệ thống camera giám sát đô thị (IC4SD-Traffic-HCM). Hệ thống bao gồm 4 hướng nghiên cứu cốt lõi được đánh số thứ tự liên tục, đi kèm hệ thống mô hình đối chuẩn (Baselines) và công trình dữ liệu quy mô thành phố:

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
├── direction1_new/                         # [HƯỚNG 1] Vehicle-Centric SSL Pretraining (Không cần ảnh nền)
│   ├── features.py                         # Trích xuất patch tokens DINOv3 đóng băng và chiếu PCA 64 chiều
│   ├── tam.py                              # Temporal Atypicality Map & PositionStats (K=4 cụm trạng thái)
│   ├── masking.py                          # Atypicality-Guided Masking (AGM) Gumbel Top-K
│   ├── srs.py                              # Static Region Swap giữa 2 frame khác ngày cùng camera
│   ├── losses.py                           # DINO [CLS] + iBOT [Patch] + KoLeo Differential Entropy Loss
│   ├── train.py                            # Pipeline huấn luyện tự giám sát không cần ảnh nền
│   └── README.md                           # Tài liệu kỹ thuật chi tiết Hướng 1
│
├── direction2_new/                         # [HƯỚNG 2] Prior-Free Traffic Scene Decomposition (Không cần ảnh nền)
│   ├── scene_fit.py                        # Giai đoạn 1: Khớp SceneBasis đa chiếu sáng (E0, Ej, ell) bằng Robust IRLS
│   ├── solve_ell.py                        # Solver giải tích Weighted Least Squares cho mã ánh sáng ell
│   ├── models.py                           # TrafficDecompositionNet (5 đầu ra: M, Fg, Bg, log_sigma, ell)
│   ├── losses.py                           # SceneDecompositionLossV2 (Laplace NLL, Sparsity, Total Variation)
│   ├── dataset.py                          # Nạp cặp ảnh và pseudo-backgrounds tự sinh
│   ├── train.py                            # Pipeline huấn luyện mạng phân rã sâu
│   ├── infer.py                            # Suy luận phân rã cảnh và xóa xe trên ảnh thực tế
│   └── README.md                           # Tài liệu kỹ thuật chi tiết Hướng 2
│
├── direction3_weak_supervision/            # [HƯỚNG 3] Context-Aware Weak Supervision (Gộp nhãn yếu đa nguồn)
│   ├── lfs/                                # Các hàm sinh nhãn yếu (Detector Area, Bg Diff, Temporal Diff, History, VLM)
│   ├── context.py                          # Phân loại 54 ngữ cảnh đô thị (Ánh sáng x Khung giờ x Cấp đường x Độ tin cậy r_i)
│   ├── label_model.py                      # Context-Aware Markov Label Model (Thuật toán EM + Forward-Backward log-sum-exp)
│   ├── end_model.py                        # Mô hình đích: DINOv3 ViT + Causal GRU (Tự chủ 100%, không cần LF hay nền)
│   ├── evaluate.py                         # Đánh giá đối chuẩn với Majority Vote và Dawid-Skene
│   └── README.md                           # Tài liệu kỹ thuật chi tiết Hướng 3
│
├── direction4_anomaly_detection/           # [HƯỚNG 4] Persistence Traffic Anomaly & Camera Fault Disentanglement
│   ├── features.py                         # Trích xuất patch token DINOv3 + Mặt nạ lòng đường Road-Aware Mask
│   ├── pooling.py                          # Temporal Median Pooling triệt tiêu xe cộ thoáng qua, giữ biến đổi kéo dài
│   ├── bank.py                             # Coreset Normal Memory Bank (Thuật toán K-Center Greedy nén 90%)
│   ├── score.py                            # Chấm điểm bất thường Top 5% khoảng cách Euclidean L2 tới Memory Bank
│   ├── camera_fault.py                     # Tách lỗi camera (s_static) và sự cố giao thông mặt đường (s_road)
│   ├── events.py                           # Persistence Filtering & Theo dõi vòng đời sự kiện (Ngưỡng phân vị 99.5%)
│   ├── synth_events.py                     # Trình giả lập sự cố ngập lụt, tai nạn dừng đỗ, rào chắn công trình
│   ├── evaluate.py                         # Đánh giá F1-score, False Alarm Rate, Lead Time
│   └── README.md                           # Tài liệu kỹ thuật chi tiết Hướng 4
│
├── direction1_bg_guided_dino/              # [BASELINE H1] BG-Guided DINO Continual SSL (Cần ảnh nền)
│   ├── dataset.py                          # Multi-Crop + Foreground-Aware Masking (FAM)
│   ├── models.py                           # Student, Teacher EMA, DINOHead Prototype Projection
│   ├── losses.py                           # DINO [CLS] + iBOT [Patch] Multi-Crop Loss
│   ├── train.py                            # CLI huấn luyện Continual SSL
│   └── README.md
│
├── direction2_scene_decomposition/         # [BASELINE H2] Noise-Aware Scene Decomposition (Cần background prior)
│   ├── dataset.py                          # Dataset đồng bộ không gian Origin & Background
│   ├── models.py                           # Multi-scale ViT + DPT Head (M_alpha, F, B_hat, sigma)
│   ├── losses.py                           # Reconstruction + Laplace Prior + Cross-Day Shared Bg
│   ├── train.py                            # Pipeline huấn luyện phân rã cảnh
│   ├── infer.py                            # Inpainting tự động: Xóa xe, tách nền đường sạch
│   └── README.md
│
├── direction_data_article/                 # [BÀI BÁO DỮ LIỆU Q1] IC4SD-TrafficSnap (Elsevier Data in Brief)
│   ├── paper/                              # Bản thảo bài báo LaTeX (main.tex), các bảng tables/ và hình figures/
│   ├── dual_agents/                        # Hệ thống 2 tác tử phản biện độc lập & kiểm toán dữ liệu thực nghiệm
│   ├── extract_real_metrics/               # Bộ script trích xuất số liệu thực tế từ 714,123 ảnh và đồ thị OSRM
│   ├── validation/                         # Bộ công cụ xác thực độc lập bảo mật PII và đồ thị
│   └── zenodo_bundle/                      # Gói phát hành mở theo chuẩn Zenodo
│
└── tests/                                  # Thư mục kiểm thử tự động (được cấu hình trong .gitignore)
    ├── test_direction1_new.py              # Unit tests Hướng 1 (TAM, SRS, AGM, Losses)
    ├── test_direction2_new.py              # Unit tests Hướng 2 (SceneBasis, solve_ell, LossV2)
    └── test_all_directions.py              # Test suite tích hợp kiểm thử tự động toàn bộ 4 hướng cốt lõi và baselines
```

---

## Bảng So Sánh Các Hướng Nghiên Cứu

| Hướng | Tên Nghiên Cứu | Thư Mục | Cơ Chế Cốt Lõi | Vai Trò Nền | Tạp Chí Mục Tiêu |
|:---|:---|:---|:---|:---|:---|
| **Hướng 1** ⭐ | **Vehicle-Centric SSL Pretraining** | `direction1_new/` | TAM (Temporal Atypicality Map) + AGM Masking + Hoán đổi vùng tĩnh SRS đa ngày | **Không cần nền** | IEEE TPAMI, CVPR |
| **Hướng 2** ⭐ | **Prior-Free Scene Decomposition** | `direction2_new/` | Phân rã cảnh không cần ảnh nền qua SceneBasis đa chiếu sáng + Mạng nơ-ron Laplace $\sigma$ | **Không cần nền** | CVPR, IEEE TIP |
| **Hướng 3** | **Context-Aware Weak Supervision** | `direction3_weak_supervision/` | Gộp nhãn yếu từ 5 LFs qua Markov Label Model (54 ngữ cảnh) + Huấn luyện End Model (DINOv3 + Causal GRU) | LF2 chênh lệch nền | IEEE T-ITS, NeurIPS |
| **Hướng 4** | **Persistence Traffic Anomaly Detection** | `direction4_anomaly_detection/` | Temporal Feature Pooling triệt tiêu xe chạy + Coreset Normal Bank + Phân tách lỗi camera vs Sự cố ngập lụt/tai nạn | Không cần nền mốc | IEEE T-ITS, TR-C |
| **Baseline H1** | **BG-Guided DINO Continual SSL** | `direction1_bg_guided_dino/` | Foreground-Aware Masking (FAM) ép ViT học biểu diễn xe cộ thay vì nền vô nghĩa | Tiền nghiệm che FAM | IEEE T-ITS |
| **Baseline H2** | **Noise-Aware Scene Decomposition** | `direction2_scene_decomposition/` | Phân rã cảnh có giám sát ảnh nền mốc (Cross-day Background Prior) | Laplace Prior mềm | Pattern Recognition |
| **Data Article** | **IC4SD-TrafficSnap Data Article** | `direction_data_article/` | Mô tả bộ dữ liệu 608 trạm camera, 714,123 ảnh, đồ thị OSRM 2,450 cạnh có hướng | Dữ liệu nền tảng | Elsevier Data in Brief |

---

## Tài Liệu Báo Cáo Khoa Học Toàn Diện

- **Bản báo cáo PDF chất lượng in ấn**: [`Bao_Cao_Khoa_Hoc_DINO_Traffic_Suite.pdf`](Bao_Cao_Khoa_Hoc_DINO_Traffic_Suite.pdf) (14 trang, biên dịch từ [`Bao_Cao_Khoa_Hoc_DINO_Traffic_Suite.tex`](Bao_Cao_Khoa_Hoc_DINO_Traffic_Suite.tex)).
- **Tài liệu Markdown khoa học**: [`BAO_CAO_KHOA_HOC_DINO_TRAFFIC_SUITE.md`](BAO_CAO_KHOA_HOC_DINO_TRAFFIC_SUITE.md).

---

## Tối Ưu Hóa Tính Toán Phân Tán Đa GPU (Multi-GPU Engine)

Tất cả các pipeline huấn luyện đều được tích hợp module `common/gpu_utils.py` tự động tối ưu hóa phần cứng:
- **Tự động nhận diện GPU (`torch.cuda.device_count()`)**: Tự động bọc `nn.DataParallel` để phân phối tính toán song song trên tất cả các GPU khả dụng.
- **Tự động mở rộng Batch Size (Linear Batch Scaling)**: $\text{Total Batch Size} = \text{batch\_size\_per\_gpu} \times N_{\text{gpus}}$.
- **Tự động điều chỉnh Tốc độ học (Linear LR Scaling Rule)**: $\text{Effective LR} = \text{base\_lr} \times N_{\text{gpus}}$.
- **Lưu Checkpoint an toàn**: Tự động giải phóng lớp bọc `module.` qua hàm `unwrap_model()`, giúp weights tương thích hoàn toàn khi nạp lại ở môi trường 1 GPU hoặc CPU.
- **Resume Training toàn vẹn**: Tải lại trọn vẹn trạng thái huấn luyện cũ (`model`, `optimizer`, `scaler`, `epoch`) qua cờ `--resume <path>` để tiếp tục huấn luyện không suy hao.

---

## Kiểm Thử Toàn Bộ Hệ Thống (Unit Tests & Verification Suite)

Chạy script kiểm thử tự động toàn diện với dữ liệu mô phỏng trong vòng 10 giây:
```bash
# Chạy bộ test tích hợp toàn diện các hướng cốt lõi (H1, H2, H3, H4, Baselines, Common)
python tests/test_all_directions.py

# Chạy unit test riêng biệt cho Hướng 1
python -m unittest tests/test_direction1_new.py

# Chạy unit test riêng biệt cho Hướng 2
python -m unittest tests/test_direction2_new.py
```
