# DINO Traffic Suite: Khai Thác Cặp Ảnh Background–Origin Cho Thị Giác Giao Thông

Bộ công cụ nghiên cứu toàn diện khai thác tín hiệu tự giám sát vật lý từ cặp ảnh **Background (nền tĩnh)** và **Origin (có phương tiện)** từ hệ thống camera giám sát đô thị (IC4SD-Traffic-HCM). Hệ thống tập trung vào các **hướng nghiên cứu trọng tâm mũi nhọn** tự học không cần ảnh nền mẫu sạch (Direction 1 New và Direction 2 New) cùng các baseline đối chuẩn và công trình dữ liệu quy mô thành phố.

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
├── direction1_new/                         # [HƯỚNG 1 MỚI - ĐỘT PHÁ] Vehicle-Centric SSL Pretraining (Không cần ảnh nền)
│   ├── features.py                         # Trích xuất patch tokens DINOv3 đóng băng và chiếu PCA 64 chiều
│   ├── tam.py                              # Temporal Atypicality Map & PositionStats (K=4 cụm trạng thái)
│   ├── masking.py                          # Atypicality-Guided Masking (AGM) Gumbel Top-K
│   ├── srs.py                              # Static Region Swap giữa 2 frame khác ngày cùng camera
│   ├── losses.py                           # DINO [CLS] + iBOT [Patch] + KoLeo Differential Entropy Loss
│   ├── train.py                            # Pipeline huấn luyện tự giám sát không cần ảnh nền
│   └── README.md                           # Tài liệu kỹ thuật chi tiết
│
├── direction2_new/                         # [HƯỚNG 2 MỚI - ĐỘT PHÁ] Prior-Free Traffic Scene Decomposition (Không cần ảnh nền)
│   ├── scene_fit.py                        # Giai đoạn 1: Khớp SceneBasis đa chiếu sáng (E0, Ej, ell) bằng Robust IRLS
│   ├── solve_ell.py                        # Solver giải tích Weighted Least Squares cho mã ánh sáng ell
│   ├── models.py                           # TrafficDecompositionNet (5 đầu ra: M, Fg, Bg, log_sigma, ell)
│   ├── losses.py                           # SceneDecompositionLossV2 (Laplace NLL, Sparsity, Total Variation)
│   ├── dataset.py                          # Nạp cặp ảnh và pseudo-backgrounds tự sinh
│   ├── train.py                            # Pipeline huấn luyện mạng phân rã sâu
│   ├── infer.py                            # Suy luận phân rã cảnh và xóa xe trên ảnh thực tế
│   └── README.md                           # Tài liệu kỹ thuật chi tiết
│
├── direction1_bg_guided_dino/              # [HƯỚNG 1 CŨ - BASELINE] BG-Guided DINO Continual SSL (Cần ảnh nền)
│   ├── dataset.py                          # Multi-Crop + Foreground-Aware Masking (FAM)
│   ├── models.py                           # Student, Teacher EMA, DINOHead Prototype Projection
│   ├── losses.py                           # DINO [CLS] + iBOT [Patch] Multi-Crop Loss
│   ├── train.py                            # CLI huấn luyện Continual SSL
│   └── README.md
│
├── direction2_scene_decomposition/         # [HƯỚNG 2 CŨ - BASELINE] Noise-Aware Scene Decomposition (Cần background prior)
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
    ├── test_direction1_new.py              # Unit tests Hướng 1 Mới (TAM, SRS, AGM, Losses)
    ├── test_direction2_new.py              # Unit tests Hướng 2 Mới (SceneBasis, solve_ell, LossV2)
    └── test_all_directions.py              # Test suite tích hợp kiểm thử tự động các hướng trọng tâm
```

---

## Bảng So Sánh Các Hướng Nghiên Cứu Trọng Tâm

| Hướng | Tên Nghiên Cứu | Thư Mục | Cơ Chế Cốt Lõi | Vai Trò Nền | Venue Đề Xuất |
|:---|:---|:---|:---|:---|:---|
| **H1 Mới** ⭐ | **Vehicle-Centric SSL (Không Cần Nền)** | `direction1_new/` | TAM (Temporal Atypicality Map) + AGM Masking + Hoán đổi vùng tĩnh SRS đa ngày | **Không cần nền** | IEEE T-PAMI, CVPR |
| **H2 Mới** ⭐ | **Prior-Free Scene Decomposition** | `direction2_new/` | Bóc tách cảnh không cần ảnh nền qua SceneBasis đa chiếu sáng + Mạng nơ-ron Laplace $\sigma$ | **Không cần nền** | CVPR, IEEE TIP |
| **H1 Cũ** | **BG-Guided DINO Continual SSL** | `direction1_bg_guided_dino/` | Foreground-Aware Masking (FAM) ép ViT học biểu diễn xe cộ thay vì nền vô nghĩa | Tiền nghiệm che FAM | IEEE T-ITS |
| **H2 Cũ** | **Noise-Aware Scene Decomposition** | `direction2_scene_decomposition/` | Bóc tách cảnh có giám sát ảnh nền mốc (Cross-day Background Prior) | Laplace Prior mềm | Pattern Recognition |
| **Data Article** | **IC4SD-TrafficSnap Data Article** | `direction_data_article/` | Mô tả bộ dữ liệu 608 trạm camera, 714,123 ảnh, đồ thị OSRM 2,450 cạnh có hướng | Dữ liệu nền tảng | Elsevier Data in Brief |

---

## ⚡ Cơ Chế Tự Động Nhận Diện & Huấn Luyện Đa GPU (Multi-GPU Engine)

Tất cả các pipeline huấn luyện đều được tích hợp module `common/gpu_utils.py` tự động tối ưu hóa phần cứng:
- **Tự động đếm GPU (`torch.cuda.device_count()`)**: Tự động bọc `nn.DataParallel` để phân phối tính toán song song trên TẤT CẢ các GPU cùng lúc.
- **Tự động mở rộng Batch Size (Linear Batch Scaling)**: $\text{Total Batch Size} = \text{batch\_size\_per\_gpu} \times N_{\text{gpus}}$.
- **Tự động điều chỉnh Tốc độ học (Linear LR Scaling Rule)**: $\text{Effective LR} = \text{base\_lr} \times N_{\text{gpus}}$.
- **Lưu Checkpoint an toàn**: Tự động giải phóng lớp bọc `module.` qua hàm `unwrap_model()`, giúp weights tương thích hoàn toàn khi load lại ở môi trường 1 GPU hoặc CPU.
- **Resume Training đầy đủ**: Tải lại trọn vẹn trạng thái huấn luyện cũ (`model`, `optimizer`, `scaler`, `epoch`, `best_metric`) qua cờ `--resume <path>` để tiếp tục train không bị gián đoạn.

---

## 🧪 Kiểm Thử Toàn Bộ Hệ Thống (Unit Tests & Verification Suite)

Chạy script kiểm thử tự động toàn diện với dữ liệu mô phỏng trong vòng 10 giây:
```bash
# Chạy bộ test tích hợp toàn diện các hướng cốt lõi
python tests/test_all_directions.py

# Chạy unit test riêng biệt cho Hướng 1 Mới
python -m unittest tests/test_direction1_new.py

# Chạy unit test riêng biệt cho Hướng 2 Mới
python -m unittest tests/test_direction2_new.py
```
