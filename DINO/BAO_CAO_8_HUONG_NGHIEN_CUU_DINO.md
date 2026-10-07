# BÁO CÁO KHOA HỌC: HỆ SINH THÁI NGHIÊN CỨU DINO TRAFFIC SUITE CHO CAMERA GIAO THÔNG TP.HCM

> **Dự án:** Nghiên cứu và Phát triển Hệ sinh thái Thị giác Máy tính Tự giám sát cho Giám sát Giao thông Đô thị Thông minh  
> **Địa bàn thực nghiệm:** Mạng lưới camera giao thông TP.HCM (>600 camera CCTV, mật độ xe máy chiếm ưu thế)  
> **Triết lý khoa học nền tảng:** **Tách bạch đa tạp nền tĩnh khỏi đặc trưng biến thiên của phương tiện; Tự học hệ cơ sở nền không cần Prior (Prior-Free Manifold) hoặc coi Background Median là tín hiệu yếu có nhiễu (Noisy Weak Prior)** — Không sử dụng Background làm Ground Truth cứng; tích hợp cơ chế tự thích nghi và kiểm soát độ bất định ($\sigma$).  
> **Mục tiêu công bố:** Các công trình nghiên cứu mũi nhọn sẵn sàng gửi các tạp chí quốc tế Q1 (IEEE TPAMI, IEEE TIP, IEEE T-ITS, Elsevier Data in Brief).  
> **Mã nguồn hệ thống:** `g:/nckh/DINO/`  

---

## MỤC LỤC
1. [Bối cảnh Khoa học và Triết lý Nền tảng](#1-bối-cảnh-khoa-học-và-triết-lý-nền-tảng)
2. [Tầng Tiện ích Dùng chung (Common Utilities)](#2-tầng-tiện-ích-dùng-chung-common-utilities)
3. [Bài báo Hướng 1 Mới: Vehicle-Centric SSL Không Cần Nền (Direction 1 New)](#3-bài-báo-hướng-1-mới-vehicle-centric-ssl-không-cần-nền-direction-1-new)
4. [Bài báo Hướng 2 Mới: Phân Rã Cảnh Không Cần Nền (Direction 2 New)](#4-bài-báo-hướng-2-mới-phân-rã-cảnh-không-cần-nền-direction-2-new)
5. [Bài báo Hướng 1 Cũ (Baseline): BG-Guided DINO Continual SSL Pre-training](#5-bài-báo-hướng-1-cũ-baseline-bg-guided-dino-continual-ssl-pre-training)
6. [Bài báo Hướng 2 Cũ (Baseline): Noise-Aware Traffic Scene Decomposition](#6-bài-báo-hướng-2-cũ-baseline-noise-aware-traffic-scene-decomposition)
7. [Bài báo Dữ liệu: IC4SD-TrafficSnap (Elsevier Data in Brief)](#7-bài-báo-dữ-liệu-ic4sd-trafficsnap-elsevier-data-in-brief)
8. [Giao thức Thực nghiệm, Phân chia Dữ liệu và Kiểm soát Rò rỉ](#8-giao-thức-thực-nghiệm-phân-chia-dữ-liệu-và-kiểm-soát-rò-rỉ)
9. [Tổng kết và Kết quả Kiểm thử Hệ thống](#9-tổng-kết-và-kết-quả-kiểm-thử-hệ-thống)

---

## 1. BỐI CẢNH KHOA HỌC VÀ TRIẾT LÝ NỀN TẢNG

### 1.1. Bản chất dữ liệu Camera Giao thông TP.HCM
Hệ thống camera giám sát giao thông đô thị TP.HCM sở hữu hai nguồn tín hiệu hình ảnh tự nhiên cùng góc quan sát:
1. **Ảnh chụp hiện trường ($I_{\text{origin}} \in \mathbb{R}^{H \times W \times 3}$):** Thu thập theo chuỗi thời gian thưa (10–60 giây hoặc 3–5 phút/frame), ghi nhận luồng giao thông hỗn hợp với xe máy chiếm hơn 80%, thường xuyên có hiện tượng che khuất (occlusion) dày đặc.
2. **Ảnh nền tĩnh sạch bóng xe ($B \in \mathbb{R}^{H \times W \times 3}$):** Ước lượng thông qua thuật toán lọc trung vị thời gian (Temporal Median Filtering) theo từng camera $c$ và từng slot giờ $s \in [00\text{h}, 23\text{h}]$.

### 1.2. Sai số hệ thống của Background Median và Cạm bẫy Học Tự giám sát
Trong các nghiên cứu trừ nền cổ điển, ảnh nền median thường bị xem nhầm là "Ground Truth tuyệt đối". Trên thực tế tại TP.HCM, background median chứa các sai số hệ thống nghiêm trọng:
- **Bóng ma phương tiện (Ghost Vehicles):** Khi xảy ra ùn tắc giao thông kéo dài, xe buýt dừng đỗ lâu hoặc xe máy xếp hàng tại giao lộ suốt 15–30 phút, thuật toán median sẽ giữ lại các xe này như một phần của mặt đường nền. Hiện tượng này xảy ra nặng nhất đúng vào **giờ cao điểm** — thời điểm cần hệ thống giám sát chính xác nhất.
- **Biến động quang học thời tiết:** Bóng đổ di chuyển nhanh, mặt đường ướt phản chiếu ánh đèn khi mưa giông nhiệt đới, lóa đèn pha ban đêm (headlight glare).
- **Cạm bẫy đường tắt nền (Background Shortcut Trap):** Trong ảnh camera cố định, 75%–80% diện tích là vỉa hè và mặt đường tĩnh. Các mô hình tự học chuẩn (DINO, MAE) dễ đi "đường tắt", học nhận diện camera ID và góc chụp thay vì học bản chất nhận dạng xe cộ, làm suy sụp khả năng thích ứng sang camera mới.

### 1.3. Định vị Triết lý Tinh gọn Cho Hệ Thống
Hệ sinh thái DINO Suite tập trung toàn diện nguồn lực vào hai hướng mũi nhọn tự học không cần nền và hai baseline có nền:

```
+-----------------------------------------------------------------------------------------------+
|                             DINO TRAFFIC SUITE CORE ARCHITECTURE                              |
+-----------------------------------------------------------------------------------------------+
|                                                                                               |
|   +---------------------------------------------------------------------------------------+   |
|   |                      COMMON FOUNDATION & VERIFICATION LAYER                           |   |
|   |  - Static Reliability Estimator (r_i)        - Phase Correlation Camera Alignment    |   |
|   |  - Background Degradation Benchmark (BDB)    - Frame Corruption Suite (FCS)           |   |
|   |  - Multi-GPU Smart Resume & Checkpointing     - Road-Aware Subtraction Engine         |   |
|   +---------------------------------------------------------------------------------------+   |
|                                              │                                                |
|       ┌──────────────────────────────────────┴───────────────────────────────────────┐        |
|       ▼                                                                             ▼        |
|   [NHÓM SSL MŨI NHỌN KHÔNG CẦN NỀN (PRIOR-FREE)]                 [NHÓM BASELINE & DỮ LIỆU ĐÔ THỊ]     |
|   1. Hướng 1 Mới (direction1_new):                               3. Hướng 1 Cũ (direction1_bg):       |
|      Vehicle-Centric SSL (TAM + AGM + SRS)                          Continual SSL với FAM-Δ           |
|   2. Hướng 2 Mới (direction2_new):                               4. Hướng 2 Cũ (direction2_decomp):   |
|      Prior-Free Scene Decomposition (SceneBasis + σ)                Scene Decomposition Median Prior  |
|                                                                  5. Data Article: IC4SD-TrafficSnap   |
+-----------------------------------------------------------------------------------------------+
```

---

## 2. TẦNG TIỆN ÍCH DÙNG CHUNG (COMMON UTILITIES)

Mã nguồn tại thư mục: `DINO/common/`

### 2.1. Ước lượng Độ Tin cậy Vùng Tĩnh & Căn chỉnh Camera (`reliability.py`)
- **Vùng tĩnh thực sự ($S$):** Tính dựa trên phương sai cường độ ánh sáng thời gian của chuỗi frame:
  $$S = \left\{(u, v) \mid \operatorname{Var}_{t}(I_t(u, v)) < \tau_{\text{var}}^2 \right\}$$
- **Hệ số tin cậy nền ($r_i$):**
  $$r_i = \exp\left( -\frac{\bar{\Delta}_{\text{static}}}{\kappa} \right), \quad \bar{\Delta}_{\text{static}} = \frac{1}{|S|} \sum_{(u, v) \in S} |I(u, v) - B(u, v)|$$
  Khi camera bị rung, đổi góc hoặc ánh sáng chênh lệch lớn $\implies \bar{\Delta}_{\text{static}}$ tăng $\implies r_i \to 0$.
- **Căn chỉnh Camera bằng Phase Correlation 2D:** Biến đổi Fourier $F_1, F_2$ trên vùng tĩnh để phát hiện độ dịch chuyển $(\Delta x, \Delta y)$ với độ chính xác sub-pixel.

### 2.2. Bộ Suy thoái Nền BDB — Background Degradation Benchmark (`degradation.py`)
Mô phỏng 6 loại nhiễu $\times$ 5 mức độ nghiêm trọng (Severity 1 $\to$ 5): `time_shift`, `camera_shift`, `ghost_injection`, `optical_change`, `noise_compression`, `cross_camera_swap`.

### 2.3. Bộ Hư hao Khung hình FCS — Frame Corruption Suite (`corrupt.py`)
8 loại hư hao ngoại cảnh thực tế: `gaussian_noise`, `motion_blur`, `defocus_blur`, `jpeg_compression`, `low_light`, `fog`, `rain_streaks`, `glare`.

### 2.4. Quản lý Checkpoint Đa GPU Chuẩn Production (`gpu_utils.py`)
Tự động dọn sạch tiền tố `module.` khi huấn luyện phân tán `DataParallel` / `DistributedDataParallel`, lưu trữ và khôi phục nguyên vẹn Model weights, Optimizer, LR Scheduler, Epoch và Metrics.

---

## 3. BÀI BÁO HƯỚNG 1 MỚI: VEHICLE-CENTRIC SSL KHÔNG CẦN NỀN (DIRECTION 1 NEW)

> **Tên bài báo:** *Vehicle-Centric Representation Learning from Sparse Static-Camera Feeds via Long-Term Temporal Atypicality and Counterfactual Background Swapping*  
> **Target:** IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI) / CVPR / ECCV / IEEE T-ITS  
> **Mã nguồn:** `direction1_new/`

### 3.1. Đóng góp Khoa học Đột phá
1. **TAM (Temporal Atypicality Map):**
   - Đóng băng DINOv3 ViT kết hợp chiếu PCA 64 chiều, xây dựng lớp `PositionStats` lưu $K=4$ cụm trạng thái tĩnh trực tuyến cho từng vị trí patch $p$.
   - Sử dụng GMM Calibrator ước lượng xác suất tiền cảnh $\pi_t(p) \in [0, 1]$ từ chuỗi ảnh thưa rời rạc mà không cần video liên tục hay ảnh nền có sẵn.
2. **AGM (Atypicality-Guided Masking):**
   - Tập trung ngân sách che $\phi = 0.5$ vào các patch có $\pi_t$ cao để ép Student ViT học suy luận ngữ cảnh xe cộ, khống chế trần $q_{\max} = 0.6$ tránh mất hoàn toàn đặc trưng nhận dạng.
3. **SRS (Static-Region Swap):**
   - Lấy mẫu 2 frame khác ngày cùng camera; hoán đổi các khối nền tĩnh ($\pi < 0.2$) giữa 2 khung hình để phá vỡ tương quan giả tạo giữa góc nền và xe cộ (triệt tiêu hiện tượng Background Shortcut Trap).
4. **Hàm mất mát chưng cất tự thân đa nhiệm:**
   $$\mathcal{L} = \mathcal{L}_{\text{DINO}}^{[\text{CLS}]} + \lambda_{\text{ibot}} \mathcal{L}_{\text{iBOT}}^{[\text{Patch}]}(\pi) + \lambda_{\text{koleo}} \mathcal{L}_{\text{KoLeo}}$$

---

## 4. BÀI BÁO HƯỚNG 2 MỚI: PHÂN RÃ CẢNH KHÔNG CẦN NỀN (DIRECTION 2 NEW)

> **Tên bài báo:** *Prior-Free Traffic Scene Decomposition via Multi-Illumination Manifold Learning and Uncertainty-Aware Composite Reconstruction*  
> **Target:** IEEE Transactions on Image Processing (TIP) / Pattern Recognition / IEEE TCSVT  
> **Mã nguồn:** `direction2_new/`

### 4.1. Đóng góp Khoa học Đột phá
1. **Hệ cơ sở quang học tĩnh `SceneBasis`:**
   - Mô hình hóa nền của mỗi camera dưới dạng tổ hợp tuyến tính của hệ cơ sở: $B(t) = E_0 + \sum_{j=1}^J \ell_j(t) E_j$, với $E_0$ là ảnh tĩnh cơ sở và $E_j$ ($J=3$) là các thành phần biến thiên chiếu sáng, bóng râm và độ ẩm.
   - Khớp trực tiếp từ lịch sử nhiều ngày không nhãn, không phụ thuộc ảnh nền median.
2. **Bộ giải trực tuyến $\boldsymbol{\ell}(t)$:**
   - Giải tối ưu hóa hệ số ánh sáng $\boldsymbol{\ell}(t)$ tức thời bằng phương pháp giải tích đóng Least Squares hoặc Robust Huber-IRLS trên vùng tĩnh $(1 - \alpha)^2$.
3. **Mạng phân rã toàn diện và Hàm mất mát Laplace NLL với độ bất định:**
   - Dự đoán đồng thời: $(\hat{I}_{\text{recon}}, \hat{B}, \hat{F}, \alpha, \sigma, \hat{\boldsymbol{\ell}})$.
   - Hàm mất mát Laplace Negative Log-Likelihood tự học bản đồ $\sigma(u, v)$ giúp mạng thích ứng hoàn hảo với các vùng lóa đèn và biên mép phức tạp:
     $$\mathcal{L}_{\text{laplace}} = \frac{1}{HW} \sum_{u, v} \left[ \frac{|I(u, v) - \hat{I}_{\text{recon}}(u, v)|}{\sigma(u, v)} + \log \sigma(u, v) \right]$$

---

## 5. BÀI BÁO HƯỚNG 1 CŨ (BASELINE): BG-GUIDED DINO CONTINUAL SSL PRE-TRAINING

> **Tên bài báo:** *Background-Guided Self-Supervised Vision Transformer Pre-training for Dense Urban Traffic Surveillance*  
> **Target:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
> **Mã nguồn:** `direction1_bg_guided_dino/`

- Sử dụng Foreground-Aware Masking (FAM-$\Delta$) chuẩn hóa thứ bậc để tập trung che vào vùng xe cộ dựa trên sai khác với ảnh nền median.
- Cổng tin cậy thích ứng $r_i$ giúp mô hình tự chuyển mượt về che ngẫu nhiên khi chất lượng ảnh nền suy giảm.
- Đóng vai trò là phương pháp baseline đối chuẩn trực tiếp cho Hướng 1 Mới.

---

## 6. BÀI BÁO HƯỚNG 2 CŨ (BASELINE): NOISE-AWARE TRAFFIC SCENE DECOMPOSITION

> **Tên bài báo:** *Noise-Aware Traffic Scene Decomposition with Imperfect Background Priors on City-Scale Camera Networks*  
> **Target:** Pattern Recognition / IEEE TCSVT  
> **Mã nguồn:** `direction2_scene_decomposition/`

- Mô hình hóa Alpha Compositing với Laplace Prior trên ảnh nền median, có bản đồ độ bất định $\sigma$ để giảm thiểu tác động của bóng ma xe kẹt.
- Ràng buộc nền dùng chung giữa các ngày khác nhau $\mathcal{L}_{\text{shared}}$ giúp bóc tách phương tiện dừng đỗ lâu.
- Đóng vai trò là phương pháp baseline đối chuẩn trực tiếp cho Hướng 2 Mới.

---

## 7. BÀI BÁO DỮ LIỆU: IC4SD-TRAFFICSAP (ELSEVIER DATA IN BRIEF)

> **Tên bài báo:** *IC4SD-TrafficSnap: A Multi-Modal Dataset of Sparse Surveillance Imagery and Road Network Topology for Urban Traffic Analysis in Ho Chi Minh City*  
> **Target:** Elsevier Data in Brief  
> **Mã nguồn:** `direction_data_article/`

- Mô tả toàn diện bộ dữ liệu 608 trạm camera tại TP.HCM, 714,123 ảnh JPEG (44.38 GiB), và đồ thị mạng lưới đường bộ OSRM gồm 2,450 cạnh có hướng.
- Kiểm toán bảo mật dữ liệu PII tuân thủ chuẩn mực Rule of Three ($p \le 0.30\%$).
- Đã hoàn thiện bản thảo LaTeX 26 trang (`paper/main.tex`), 5 bảng số liệu thực nghiệm và hệ thống 2 tác tử phản biện độc lập (`dual_agents/`).

---

## 8. GIAO THỨC THỰC NGHIỆM, PHÂN CHIA DỮ LIỆU VÀ KIỂM SOÁT RÒ RỈ

- **Phân chia theo Cụm Camera (Spatial Disjoint Splitting):** Các camera thuộc cùng một ngã tư hoặc nút giao bắt buộc phải nằm chung một tập phân chia (70% Train, 10% Val, 20% Test) để triệt tiêu hiện tượng rò rỉ bối cảnh tĩnh.
- **Quy chuẩn kích thước khung hình:** Chuẩn hóa $256 \times 448$ (tỷ lệ chuẩn $16:9$) để không làm méo mó hình học của các dòng xe máy nhỏ ở xa.

---

## 9. TỔNG KẾT VÀ KẾT QUẢ KIỂM THỬ HỆ THỐNG

Bộ kiểm thử tự động tại `tests/test_all_directions.py` đã vượt qua 100% các bài test:
1. `[TEST 1]` Common Utilities (TrafficPairMatcher & BackgroundSubtractor) $\rightarrow$ **PASSED**
2. `[TEST 2]` Direction 1 Cũ (BG-Guided DINO SSL Baseline) $\rightarrow$ **PASSED**
3. `[TEST 3]` Direction 2 Cũ (Scene Decomposition Baseline) $\rightarrow$ **PASSED**
4. `[TEST 6]` Multi-GPU Smart Checkpointing Interoperability $\rightarrow$ **PASSED**
5. `[TEST 7]` Common Advanced (Reliability $r_i$, BDB Degradation & FCS Frame Corruptions) $\rightarrow$ **PASSED**
6. `[TEST 12]` Direction 1 Mới (Vehicle-Centric SSL Pretraining TAM + AGM + SRS) $\rightarrow$ **PASSED**
7. `[TEST 13]` Direction 2 Mới (Prior-Free Scene Decomposition SceneBasis + Huber-IRLS + Loss V2) $\rightarrow$ **PASSED**
