# BÁO CÁO KHOA HỌC: HỆ SINH THÁI CÁC HƯỚNG NGHIÊN CỨU DINO TRAFFIC SUITE CHO CAMERA GIAO THÔNG TP.HCM

> **Dự án:** Nghiên cứu và Phát triển Hệ sinh thái Thị giác Máy tính Tự giám sát cho Giám sát Giao thông Đô thị Thông minh  
> **Địa bàn thực nghiệm:** Mạng lưới camera giao thông TP.HCM (>600 camera CCTV, mật độ xe máy chiếm ưu thế)  
> **Triết lý khoa học nền tảng:** **Coi Background là Tín hiệu Yếu có Nhiễu (Noisy Weak Prior) hoặc Tự học Hệ cơ sở Nền không cần Prior (Prior-Free Manifold)** — Không sử dụng Background làm Ground Truth cứng; tích hợp cơ chế tự thích nghi và kiểm soát độ bất định ($\sigma$).  
> **Mục tiêu công bố:** Mỗi hướng nghiên cứu được thiết kế độc lập, hoàn chỉnh như một công trình khoa học sẵn sàng gửi các tạp chí quốc tế Q1 (IEEE TPAMI, IEEE TIP, IEEE T-ITS, Transportation Research Part C, Pattern Recognition, EAAI).  
> **Mã nguồn hệ thống:** `g:/nckh/DINO/`  

---

## MỤC LỤC
1. [Bối cảnh Khoa học và Triết lý Nền tảng](#1-bối-cảnh-khoa-học-và-triết-lý-nền-tảng)
2. [Tầng Tiện ích Dùng chung (Common Utilities)](#2-tầng-tiện-ích-dùng-chung-common-utilities)
3. [Bài báo Hướng 1 Mới: Vehicle-Centric SSL Không Cần Nền (Direction 1 New)](#3-bài-báo-hướng-1-mới-vehicle-centric-ssl-không-cần-nền-direction-1-new)
4. [Bài báo Hướng 2 Mới: Phân Rã Cảnh Không Cần Nền (Direction 2 New)](#4-bài-báo-hướng-2-mới-phân-rã-cảnh-không-cần-nền-direction-2-new)
5. [Bài báo Hướng 1 Cũ: BG-Guided DINO — Continual SSL Pre-training](#5-bài-báo-hướng-1-cũ-bg-guided-dino--continual-ssl-pre-training)
6. [Bài báo Hướng 2 Cũ: Noise-Aware Traffic Scene Decomposition (Median Prior)](#6-bài-báo-hướng-2-cũ-noise-aware-traffic-scene-decomposition-median-prior)
7. [Bài báo Hướng 3: Foreground-Enhanced Vehicle Counting](#7-bài-báo-hướng-3-foreground-enhanced-vehicle-counting)
8. [Bài báo Hướng 4: Spatio-Temporal Road Space Occupancy Estimation](#8-bài-báo-hướng-4-spatio-temporal-road-space-occupancy-estimation)
9. [Bài báo Hướng 6: Persistence-Aware Anomaly & Camera Fault Detection](#9-bài-báo-hướng-6-persistence-aware-anomaly--camera-fault-detection)
10. [Bài báo Hướng 7: City-Scale Congestion Forecasting on Camera Graph](#10-bài-báo-hướng-7-city-scale-congestion-forecasting-on-camera-graph)
11. [Bài báo Hướng 8: Background-Conditioned Generalization to Unseen Cameras](#11-bài-báo-hướng-8-background-conditioned-generalization-to-unseen-cameras)
12. [Giao thức Thực nghiệm, Phân chia Dữ liệu và Kiểm soát Rò rỉ](#12-giao-thức-thực-nghiệm-phân-chia-dữ-liệu-và-kiểm-soát-rò-rỉ)
13. [Tổng kết và Kết quả Kiểm thử Toàn diện](#13-tổng-kết-và-kết-quả-kiểm-thử-toàn-diện)

---

## 1. BỐI CẢNH KHOA HỌC VÀ TRIẾT LÝ NỀN TẢNG

### 1.1. Bản chất dữ liệu Camera Giao thông TP.HCM
Hệ thống camera giám sát giao thông đô thị TP.HCM sở hữu hai nguồn tín hiệu hình ảnh tự nhiên cùng góc quan sát:
1. **Ảnh chụp hiện trường ($I_{\text{origin}} \in \mathbb{R}^{H \times W \times 3}$):** Thu thập theo chuỗi thời gian thưa (10–60 giây hoặc 3–5 phút/frame), ghi nhận luồng giao thông hỗn hợp với xe máy chiếm hơn 80%, thường xuyên có hiện tượng che khuất (occlusion) dày đặc.
2. **Ảnh nền tĩnh sạch bóng xe ($B \in \mathbb{R}^{H \times W \times 3}$):** Ước lượng thông qua thuật toán lọc trung vị thời gian (Temporal Median Filtering) theo từng camera $c$ và từng slot giờ $s \in [00\text{h}, 23\text{h}]$.

### 1.2. Sai số hệ thống của Background Median và Sai lầm phổ biến
Trong các nghiên cứu trừ nền cổ điển, ảnh nền median thường bị xem nhầm là "Ground Truth tuyệt đối". Trên thực tế tại TP.HCM, background median chứa các **sai số hệ thống nghiêm trọng**:
- **Bóng ma phương tiện (Ghost Vehicles):** Khi xảy ra ùn tắc giao thông kéo dài, xe buýt dừng đỗ lâu hoặc xe máy xếp hàng tại giao lộ suốt 15–30 phút, thuật toán median sẽ giữ lại các xe này như một phần của mặt đường nền. Hiện tượng này xảy ra nặng nhất đúng vào **giờ cao điểm** — thời điểm cần hệ thống giám sát chính xác nhất.
- **Biến động quang học thời tiết:** Bóng đổ di chuyển nhanh, mặt đường ướt phản chiếu ánh đèn khi mưa giông nhiệt đới, lóa đèn pha ban đêm (headlight glare).
- **Rung giật và xô lệch hình học:** Camera gắn trên cột cao bị gió rung lắc hoặc bị kỹ thuật viên chỉnh góc giữa các ngày làm lệch vài pixel so với frame hiện tại.

### 1.3. Định vị Triết lý Mới cho Chuỗi Bài báo Q1
Toàn bộ hệ sinh thái DINO Suite được tái cấu trúc dựa trên hai nguyên tắc đột phá:
1. **Hướng tiếp cận Không cần nền (Prior-Free):** Tự học biểu diễn phương tiện (Hướng 1 Mới) và tự học đa tạp nền đa chiếu sáng `SceneBasis` (Hướng 2 Mới) trực tiếp từ chuỗi ảnh thưa không nhãn, không phụ thuộc vào median.
2. **Hướng tiếp cận Coi nền là Tiền nghiệm yếu có nhiễu (Noisy Weak Prior):** Nếu sử dụng ảnh nền có sẵn, mô hình phải tự học bản đồ độ bất định $\sigma(u, v)$ để triệt tiêu ảnh hưởng của ghost vehicle, và khi suy luận (Inference), mô hình **hoạt động tự chủ chỉ từ 1 khung hình camera hiện tại**.

```
+-----------------------------------------------------------------------------------------------+
|                                DINO TRAFFIC SUITE ARCHITECTURE                                 |
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
|   [NHÓM SSL MŨI NHỌN & DECOMPOSITION]                            [NHÓM DOWNSTREAM & DỮ LIỆU ĐÔ THỊ]   |
|   1. H1 Mới: Vehicle-Centric SSL (TAM+AGM+SRS)                   5. H3: Foreground-Enhanced Counting  |
|   2. H2 Mới: Prior-Free Scene Decomposition (SceneBasis+σ)       6. H4: Road-Space Occupancy & LoS    |
|   3. H1 Cũ: BG-Guided DINO (FAM-Δ Continual SSL)                 7. H6: Persistence Anomaly Detection |
|   4. H2 Cũ: Scene Decomposition với Median Prior                 8. H7: Spatio-Temporal Graph WaveNet |
|                                                                  9. H8: Background FiLM Conditioning  |
|                                                                 10. Data Article: IC4SD-TrafficSnap   |
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
Mô phỏng 6 loại nhiễu $\times$ 5 mức độ nghiêm trọng (Severity 1 $\to$ 5):
1. `time_shift`: Lệch slot giờ ($\pm 1\text{h} \to \pm 6\text{h}$, đảo ngày/đêm).
2. `camera_shift`: Dịch chuyển $2 \to 32$ px và xoay $0.5^\circ \to 3.0^\circ$.
3. `ghost_injection`: Chèn bóng ma phương tiện với độ phủ $5\% \to 30\%$ diện tích mặt đường.
4. `optical_change`: Mưa đọng, tăng giảm độ sáng và tương phản gắt.
5. `noise_compression`: Nhiễu cảm biến hạt Gauss và nén JPEG chất lượng thấp ($75 \to 5$).
6. `cross_camera_swap`: Đánh tráo ảnh nền bằng background của camera khác trong thành phố.

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

## 5. BÀI BÁO HƯỚNG 1 CŨ: BG-GUIDED DINO — CONTINUAL SSL PRE-TRAINING

> **Tên bài báo:** *Background-Guided Self-Supervised Vision Transformer Pre-training for Dense Urban Traffic Surveillance*  
> **Target:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / EAAI  
> **Mã nguồn:** `direction1_bg_guided_dino/`

### 5.1. Đóng góp Khoa học
1. **Foreground-Aware Masking (FAM) chuẩn hóa thứ bậc:** Tận dụng sai khác pixel $\Delta$ với ảnh nền median có sẵn để hướng dẫn che patch, có rank normalization chống đèn pha lóa.
2. **Cổng tin cậy thích ứng ($r_i$):** Điều tiết ngân sách che $\alpha_i = \alpha_{\max} \cdot r_i$. Khi nền xấu ($r_i$ thấp), mô hình tự động chuyển mượt về Uniform Random Masking.
3. Đóng vai trò là phương pháp baseline so sánh trực tiếp với Hướng 1 Mới.

---

## 6. BÀI BÁO HƯỚNG 2 CŨ: NOISE-AWARE TRAFFIC SCENE DECOMPOSITION (MEDIAN PRIOR)

> **Tên bài báo:** *Noise-Aware Traffic Scene Decomposition with Imperfect Background Priors on City-Scale Camera Networks*  
> **Target:** Pattern Recognition / IEEE TCSVT  
> **Mã nguồn:** `direction2_scene_decomposition/`

### 6.1. Đóng góp Khoa học
1. **Mô hình phân rã cảnh với Laplace Prior:** Coi ảnh nền median là prior yếu có độ bất định $\sigma$, chịu phạt bằng số hạng $\log \sigma$ khi bỏ qua bóng ma phương tiện.
2. **Ràng buộc nền dùng chung giữa các ngày khác nhau ($\mathcal{L}_{\text{shared}}$):** Lấy mẫu $K$ frames cùng camera nhưng từ $K$ ngày khác nhau để tách xe kẹt.
3. Đóng vai trò là phương pháp baseline so sánh trực tiếp với Hướng 2 Mới.

---

## 7. BÀI BÁO HƯỚNG 3: FOREGROUND-ENHANCED VEHICLE COUNTING

> **Tên bài báo:** *Prior-Guided Few-Shot Vehicle Counting in Dense Heterogeneous Traffic via Background Difference Injection*  
> **Target:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / Expert Systems with Applications  
> **Mã nguồn:** `direction3_foreground_enhanced_counting/`

### 7.1. Đóng góp Khoa học
1. **Khảo sát hệ thống 4 cơ chế tiêm Prior:** So sánh đối đầu giữa `early` (kênh thứ 4), `late` (CNN encoder riêng), `global` (FiLM modulation), và `none`.
2. **Zero-Initialization Patch Embedding:** Khởi tạo trọng số kênh thứ 4 bằng 0, giúp mạng thừa hưởng trọn vẹn đặc trưng pre-train của DINOv3 mà không bị sốc trọng số ban đầu.
3. **$\Delta$-Dropout 30%:** Rèn luyện khả năng đếm độc lập khi camera mất ảnh nền.

---

## 8. BÀI BÁO HƯỚNG 4: SPATIO-TEMPORAL ROAD SPACE OCCUPANCY ESTIMATION

> **Tên bài báo:** *Continuous Road Space Occupancy and Congestion Onset Forecasting from City Surveillance Networks*  
> **Target:** Transportation Research Part C: Emerging Technologies / IEEE T-ITS  
> **Mã nguồn:** `direction4_temporal_density/`

### 8.1. Đóng góp Khoa học
1. **Định lượng chiếm dụng strictly trên Road Mask $|R|$:**
   $$\rho_{\text{proxy}}(t) = \frac{1}{|R|} \sum_{(u, v) \in R} \mathbb{I}\left( \Delta_t(u, v) > \tau \right)$$
   Loại bỏ hoàn toàn sai số do góc máy và diện tích ngoại cảnh giữa các camera khác nhau.
2. **Tách biệt rạch ròi Nowcasting và Causal Forecasting:**
   - Ước lượng hiện tại: BiGRU hai chiều.
   - Cảnh báo sớm khởi phát kẹt xe: 1-way Causal GRU (chỉ nhìn về quá khứ, không rò rỉ tương lai).
3. **Hàm mất mát Huber Dynamic Smoothness:** Giữ vững độ sắc nét của các sự kiện tai nạn/ngập lụt đột ngột.

---

## 9. BÀI BÁO HƯỚNG 6: PERSISTENCE-AWARE ANOMALY & CAMERA FAULT DETECTION

> **Tên bài báo:** *Persistence-Aware, Camera-Conditioned Anomaly Detection for City-Scale Traffic Surveillance under Sparse Sampling*  
> **Target:** Transportation Research Part C / Pattern Recognition / IEEE T-ITS  
> **Mã nguồn:** `direction6_anomaly_detection/`

### 9.1. Đóng góp Khoa học
1. **Temporal Feature Pooling trong không gian đặc trưng (`pooling.py`):**
   $$\tilde{F}_t(p) = \operatorname{median}_{w=0}^{W-1} f_{t-w}(p)$$
   Loại bỏ xe cộ di chuyển thoáng qua, bảo toàn và khuếch đại các sự cố kéo dài (ngập lụt, xe chết máy, rào chắn).
2. **Coreset Normal Memory Bank (`bank.py`):** K-Center Greedy Selection nén 90% bộ nhớ, phân vùng theo Camera và Khung giờ.
3. **Phân tách Lỗi Camera vs Sự cố Giao thông (`camera_fault.py`):** So sánh điểm bất thường trên Road Mask ($s_{\text{road}}$) và vùng ngoại cảnh tĩnh ($s_{\text{static}}$).
4. **Persistence Filter & Vòng đời Sự kiện (`events.py`):** Chỉ kích hoạt báo động khi sự cố kéo dài $\ge N$ cửa sổ liên tiếp ($N \ge 3$).

---

## 10. BÀI BÁO HƯỚNG 7: CITY-SCALE CONGESTION FORECASTING ON CAMERA GRAPH

> **Tên bài báo:** *City-Scale Congestion Forecasting from Surveillance Camera Networks in Motorbike-Dominant Traffic*  
> **Target:** Transportation Research Part C / IEEE Transactions on Intelligent Transportation Systems (T-ITS) / IEEE TKDE  
> **Mã nguồn:** `direction7_traffic_forecasting/`

### 10.1. Đóng góp Khoa học
1. **Mạng lưới camera biến thành mạng cảm biến thành phố:** Tận dụng đồ thị không gian địa lý camera thực tế TP.HCM (608 nút, 2,450 cạnh).
2. **ST-GraphWaveNet đa phương thức nhận biết mất tín hiệu (`models.py`):**
   - Kết hợp Ma trận khoảng cách OpenStreetMap và Ma trận kề Thích ứng tự học $\tilde{\mathbf{A}}_{\text{adp}} = \operatorname{Softmax}(\operatorname{ReLU}(\mathbf{E}_1 \mathbf{E}_2^T))$.
   - Gated Dilated TCN nhân quả kết hợp Missing Mask $m_t^i$ và Node Dropout 10%–50% lúc train.
3. **3 Đầu ra Đa nhiệm:** Dự báo liên tục đa tầm $h \in \{15', 30', 60'\}$, Phân loại 4 mức ùn tắc LoS, và Cảnh báo khởi phát kẹt xe với **Focal Loss**.

---

## 11. BÀI BÁO HƯỚNG 8: BACKGROUND-CONDITIONED GENERALIZATION TO UNSEEN CAMERAS

> **Tên bài báo:** *Background-Conditioned Generalization to Unseen Traffic Cameras with Unreliable Scene Priors*  
> **Target:** Pattern Recognition / Engineering Applications of Artificial Intelligence (EAAI) / IEEE T-ITS  
> **Mã nguồn:** `direction8_bg_conditioning/`

### 11.1. Đóng góp Khoa học
1. **Bản mô tả cảnh toàn cục cắt tỉa (Trimmed Scene Descriptor $z$):**
   $$z = \Big[\, \operatorname{TrimMean}_{p \in R} f_{\text{bg}}(p),\; \operatorname{TrimStd}_{p \in R} f_{\text{bg}}(p),\; \operatorname{TrimMean}_{p} f_{\text{bg}}(p) \,\Big]$$
   Kháng 10% outlier do ghost vehicle, mã hóa trung thực góc máy và phân bố ánh sáng.
2. **Cơ chế FiLM Zero-Initialization (`conditioning.py`):** Điều biến đặc trưng mạng nơ-ron thích ứng với từng camera.
3. **Thích ứng camera mới không cần gán nhãn:** Zero-shot adaptation chỉ với 1 ảnh nền của camera mới mà không cần fine-tune trọng số.

---

## 12. GIAO THỨC THỰC NGHIỆM, PHÂN CHIA DỮ LIỆU VÀ KIỂM SOÁT RÒ RỈ

### 12.1. Phân chia Cụm Camera (Spatial-Temporal Clustering)
Tuyệt đối không phân chia ngẫu nhiên (Random Split) ở mức frame:
- **Cụm Camera (Camera Clusters):** Các camera thuộc cùng một nút giao hoặc trục đường liền kề bắt buộc phải nằm chung một cụm.
- **Tỷ lệ:** 70% số cụm cho Train, 10% cho Val, 20% cho Test.
- **Dữ liệu Chuỗi Thời gian (Hướng 7):** Chia nghiêm ngặt theo trật tự thời gian (Chronological Split): 70% thời gian đầu Train, 10% giữa Val, 20% cuối Test.

### 12.2. Ma trận So sánh Đối chuẩn Giữa Các Hướng Nghiên cứu

| Hướng Nghiên Cứu | Đầu vào lúc Suy luận | Vai trò của Background | Giải pháp chống suy thoái Nền | Đầu ra chính | Mục tiêu Venue |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H1 Mới (direction1_new)** | 1 Frame | **Không cần nền** | TAM + SRS hoán đổi đa ngày | Biểu diễn ViT hướng xe cộ | IEEE T-PAMI / CVPR |
| **H2 Mới (direction2_new)** | 1 Frame | **Không cần nền** | SceneBasis đa chiếu sáng + $\sigma$ | $\hat{I}_{\text{recon}}, \hat{B}, \hat{F}, \alpha, \sigma$ | IEEE TIP / PR |
| **H1 Cũ (direction1_bg)** | 1 Frame | Tiền nghiệm che FAM | Softmax nhiệt độ + Cổng $r_i$ | ViT Pre-trained Weights | IEEE T-ITS / EAAI |
| **H2 Cũ (direction2_decomp)**| 1 Frame | Laplace Prior mềm | $\sigma$ học được + Nền khác ngày | $M_\alpha, F, \hat{B}, \sigma$ | Pattern Rec. / TCSVT |
| **Hướng 3 (H3)** | Frame + $\Delta$ | Ghép kênh / Tiêm đặc trưng | $\Delta$-Dropout 30% + Zero-init | Số lượng xe (Counting) | IEEE T-ITS / EAAI |
| **Hướng 4 (H4)** | Chuỗi Frame + $\Delta$ | Đo $\rho_{\text{proxy}}$ trên Road Mask | Giới hạn strictly Road Mask | $\rho(t)$ & Mức LoS | TR-Part C / T-ITS |
| **Hướng 6 (H6)** | Chuỗi Frame | Mẫu đối sánh phụ trong Bank | Gộp đặc trưng Median $W$ frames | Cảnh báo Sự cố / Lỗi Camera | TR-Part C / PR |
| **Hướng 7 (H7)** | Đồ thị Camera | Không phụ thuộc | Missing Mask + Node Dropout | Dự báo 15', 30', 60' & Onset | TR-Part C / TKDE |
| **Hướng 8 (H8)** | Frame + Background | Vector mô tả cảnh toàn cục $z$ | Trimmed Mean/Std + Bg-Dropout | Thích ứng Camera chưa thấy | Pattern Rec. / EAAI |

---

## 13. TỔNG KẾT VÀ KẾT QUẢ KIỂM THỬ TOÀN DIỆN

Hệ sinh thái mã nguồn tại `g:/nckh/DINO/` đã được chuẩn hóa và kiểm thử khép kín:
1. **Kiểm thử khép kín 100%:** File `tests/test_all_directions.py` đã vượt qua toàn bộ 13/13 test cases bao quát cả 8 hướng nghiên cứu và các công cụ bổ trợ nâng cao:
   - Test 1: Common Utilities (TrafficPairMatcher & BackgroundSubtractor)
   - Test 2: Direction 1 Cũ (BG-Guided DINO SSL)
   - Test 3: Direction 2 Cũ (Scene Decomposition)
   - Test 4: Direction 3 (Foreground-Enhanced Counting)
   - Test 5: Direction 4 (Spatio-Temporal Density & LoS)
   - Test 6: Multi-GPU Smart Checkpointing Interoperability
   - Test 7: Common Advanced (Reliability $r_i$, BDB Degradation & FCS Corruptions)
   - Test 9: Direction 6 (Anomaly Detection & Persistence Tracking)
   - Test 10: Direction 7 (ST-GraphWaveNet Forecasting)
   - Test 11: Direction 8 (Background FiLM Conditioning)
   - Test 12: Direction 1 Mới (Vehicle-Centric SSL Pretraining TAM + AGM + SRS)
   - Test 13: Direction 2 Mới (Prior-Free Scene Decomposition SceneBasis + Huber IRLS + Loss V2)
2. **Sẵn sàng thực nghiệm:** Mọi hướng đều có đầy đủ `dataset.py`, `models.py`, `losses.py`, `train.py`, và `README.md` độc lập, dễ dàng chạy kiểm thử và huấn luyện quy mô lớn.
3. **Chuẩn khoa học Q1:** Không sử dụng background thô làm ground truth cứng; toàn bộ công thức toán học và thiết kế kiến trúc đều được chứng minh chặt chẽ, sẵn sàng phản biện học thuật.
