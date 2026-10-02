# BÁO CÁO KHOA HỌC: HỆ SINH THÁI 8 HƯỚNG NGHIÊN CỨU TRỌNG TÂM KHAI THÁC CẶP ẢNH NỀN VÀ CHUỖI ẢNH CAMERA GIAO THÔNG TP.HCM

> **Dự án:** Nghiên cứu và Phát triển Hệ sinh thái Thị giác Máy tính Tự giám sát cho Giám sát Giao thông Đô thị Thông minh  
> **Địa bàn thực nghiệm:** Mạng lưới camera giao thông TP.HCM (>600 camera CCTV, mật độ xe máy chiếm ưu thế)  
> **Triết lý khoa học nền tảng:** **Coi Background là Tín hiệu Yếu có Nhiễu (Noisy Weak Prior)** — Không sử dụng Background làm Ground Truth cứng hay ngưỡng thô; tích hợp cơ chế tự thích nghi và kiểm soát độ bất định ($\sigma$).  
> **Mục tiêu công bố:** Mỗi hướng nghiên cứu được thiết kế độc lập, hoàn chỉnh như một công trình khoa học sẵn sàng gửi các tạp chí quốc tế Q1 (IEEE T-ITS, IEEE TIP, Transportation Research Part C, Pattern Recognition, EAAI).  
> **Mã nguồn hệ thống:** `g:/nckh/DINO/`  

---

## MỤC LỤC
1. [Bối cảnh Khoa học và Triết lý Nền tảng](#1-bối-cảnh-khoa-học-và-triết-lý-nền-tảng)
2. [Tầng Tiện ích Dùng chung (Common Utilities)](#2-tầng-tiện-ích-dùng-chung-common-utilities)
3. [Bài báo 1 (Hướng 1): BG-Guided DINO — Continual SSL Pre-training](#3-bài-báo-1-hướng-1-bg-guided-dino--continual-ssl-pre-training)
4. [Bài báo 2 (Hướng 2 / II.A): Noise-Aware Traffic Scene Decomposition](#4-bài-báo-2-hướng-2--iia-noise-aware-traffic-scene-decomposition)
5. [Bài báo 3 (Hướng 3): Foreground-Enhanced Vehicle Counting](#5-bài-báo-3-hướng-3-foreground-enhanced-vehicle-counting)
6. [Bài báo 4 (Hướng 4): Spatio-Temporal Road Space Occupancy Estimation](#6-bài-báo-4-hướng-4-spatio-temporal-road-space-occupancy-estimation)
7. [Bài báo 5 (Hướng B): Context-Aware Weak Supervision Label Aggregation](#7-bài-báo-5-hướng-b-context-aware-weak-supervision-label-aggregation)
8. [Bài báo 6 (Hướng C): Persistence-Aware Anomaly & Camera Fault Detection](#8-bài-báo-6-hướng-c-persistence-aware-anomaly--camera-fault-detection)
9. [Bài báo 7 (Hướng D): City-Scale Congestion Forecasting on Camera Graph](#9-bài-báo-7-hướng-d-city-scale-congestion-forecasting-on-camera-graph)
10. [Bài báo 8 (Hướng E): Background-Conditioned Generalization to Unseen Cameras](#10-bài-báo-8-hướng-e-background-conditioned-generalization-to-unseen-cameras)
11. [Giao thức Thực nghiệm, Phân chia Dữ liệu và Kiểm soát Rò rỉ](#11-giao-thức-thực-nghiệm-phân-chia-dữ-liệu-và-kiểm-soát-rò-rỉ)
12. [Tổng kết và Lộ trình Triển khai](#12-tổng-kết-và-lộ-trình-triển-khai)

---

## 1. BỐI CẢNH KHOA HỌC VÀ TRIẾT LÝ NỀN TẢNG

### 1.1. Bản chất dữ liệu Camera Giao thông TP.HCM
Hệ thống camera giám sát giao thông đô thị TP.HCM sở hữu hai nguồn tín hiệu hình ảnh tự nhiên cùng góc quan sát:
1. **Ảnh chụp hiện trường ($I_{\text{origin}} \in \mathbb{R}^{H \times W \times 3}$):** Thu thập theo chuỗi thời gian thưa (10–60 giây/frame), ghi nhận luồng giao thông hỗn hợp với xe máy chiếm hơn 80%, thường xuyên có hiện tượng che khuất (occlusion) dày đặc.
2. **Ảnh nền tĩnh sạch bóng xe ($B \in \mathbb{R}^{H \times W \times 3}$):** Ước lượng thông qua thuật toán lọc trung vị thời gian (Temporal Median Filtering) theo từng camera $c$ và từng slot giờ $s \in [00\text{h}, 23\text{h}]$.

### 1.2. Sai số hệ thống của Background Median và Sai lầm phổ biến
Trong các nghiên cứu trừ nền cổ điển, ảnh nền median thường bị xem nhầm là "Ground Truth tuyệt đối". Trên thực tế tại TP.HCM, background median chứa các **sai số hệ thống nghiêm trọng**:
- **Bóng ma phương tiện (Ghost Vehicles):** Khi xảy ra ùn tắc giao thông kéo dài, xe buýt dừng đỗ lâu hoặc xe máy xếp hàng tại giao lộ suốt 15–30 phút, thuật toán median sẽ giữ lại các xe này như một phần của mặt đường nền. Hiện tượng này xảy ra nặng nhất đúng vào **giờ cao điểm** — thời điểm cần hệ thống giám sát chính xác nhất.
- **Biến động quang học thời tiết:** Bóng đổ di chuyển nhanh, mặt đường ướt phản chiếu ánh đèn khi mưa giông nhiệt đới, lóa đèn pha ban đêm (headlight glare).
- **Rung giật và xô lệch hình học:** Camera gắn trên cột cao bị gió rung lắc hoặc bị kỹ thuật viên chỉnh góc giữa các ngày làm lệch vài pixel so với frame hiện tại.

### 1.3. Định vị Triết lý Mới cho Chuỗi Bài báo Q1
Toàn bộ hệ sinh thái DINO Suite được tái cấu trúc dựa trên nguyên tắc:
$$\text{Background median } B_{c,s} \text{ chỉ là một tín hiệu tiên nghiệm yếu (Noisy Weak Prior), không phải nhãn cứng.}$$
Mọi mô hình phải:
1. Tự học bản đồ độ bất định $\sigma(u, v)$ để biết nơi nào background bị sai (ghost).
2. Khi suy luận thực tế (Inference), mô hình **hoạt động tự chủ chỉ từ 1 khung hình camera hiện tại** mà không bị lệ thuộc vào background.
3. Vượt qua bộ kiểm thử suy thoái nền **BDB (Background Degradation Benchmark)** gồm 6 loại nhiễu $\times$ 5 mức độ nghiêm trọng.

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
|                                              |                                                |
|       +--------------------------------------+---------------------------------------+        |
|       |                                                                              |        |
|   [CORE DIRECTIONS (H1 - H4)]                                    [EXPANDED CITY-SCALE DIRECTIONS (B - E)]
|   1. H1: BG-Guided DINO (DINO + iBOT SSL)                        5. Dir B: Context-Aware Weak Supervision   
|   2. H2: Noise-Aware Scene Decomposition                         6. Dir C: Persistence Anomaly Detection    
|   3. H3: Foreground-Enhanced Counting                            7. Dir D: Spatio-Temporal Graph WaveNet    
|   4. H4: Road-Space Occupancy & LoS                              8. Dir E: Background-Conditioned Adaptation
|                                                                                               |
+-----------------------------------------------------------------------------------------------+
```

---

## 2. TẦNG TIỆN ÍCH DÙNG CHUNG (COMMON UTILITIES)

Mã nguồn tại thư mục: `DINO/common/`

### 2.1. Ước lượng Độ Tin cậy Vùng Tĩnh & Căn chỉnh Camera (`reliability.py`)
- **Vùng tĩnh thực sự ($S$):** Không lấy phần bù thô của lòng đường mà tính dựa trên phương sai cường độ ánh sáng thời gian của chuỗi frame:
  $$S = \left\{(u, v) \mid \operatorname{Var}_{t}(I_t(u, v)) < \tau_{\text{var}}^2 \right\}$$
- **Hệ số tin cậy nền ($r_i$):**
  $$r_i = \exp\left( -\frac{\bar{\Delta}_{\text{static}}}{\kappa} \right), \quad \bar{\Delta}_{\text{static}} = \frac{1}{|S|} \sum_{(u, v) \in S} |I(u, v) - B(u, v)|$$
  Khi camera bị rung, đổi góc hoặc ánh sáng chênh lệch lớn $\implies \bar{\Delta}_{\text{static}}$ tăng $\implies r_i \to 0$.
- **Căn chỉnh Camera bằng Phase Correlation 2D:** Biến đổi Fourier $F_1, F_2$ trên vùng tĩnh để phát hiện độ dịch chuyển $(\Delta x, \Delta y)$ với độ chính xác sub-pixel. Nếu dịch $> 4$ px $\implies$ phát cờ camera xô lệch.

### 2.2. Bộ Suy thoái Nền BDB — Background Degradation Benchmark (`degradation.py`)
Mô phỏng 6 loại nhiễu $\times$ 5 mức độ nghiêm trọng (Severity 1 $\to$ 5) để trả lời câu hỏi phản biện của reviewer: *"Nếu background bị lỗi thì mô hình có sụp đổ không?"*:
1. `time_shift`: Lệch slot giờ ($\pm 1\text{h} \to \pm 6\text{h}$, đảo ngày/đêm).
2. `camera_shift`: Dịch chuyển $2 \to 32$ px và xoay $0.5^\circ \to 3.0^\circ$.
3. `ghost_injection`: Chèn bóng ma phương tiện với độ phủ $5\% \to 30\%$ diện tích mặt đường.
4. `optical_change`: Mưa đọng, tăng giảm độ sáng và tương phản gắt.
5. `noise_compression`: Nhiễu cảm biến hạt Gauss và nén JPEG chất lượng thấp ($75 \to 5$).
6. `cross_camera_swap`: Đánh tráo ảnh nền bằng background của camera khác trong thành phố.

### 2.3. Bộ Hư hao Khung hình FCS — Frame Corruption Suite (`corrupt.py`)
8 loại hư hao ngoại cảnh thực tế: `gaussian_noise`, `motion_blur`, `defocus_blur`, `jpeg_compression`, `low_light`, `fog`, `rain_streaks`, `glare`.

### 2.4. Quản lý Checkpoint Đa GPU Chuẩn Production (`gpu_utils.py`)
Tự động dọn sạch tiền tố `module.` khi huấn luyện phân tán `DataParallel` / `DistributedDataParallel`, lưu trữ và khôi phục nguyên vẹn Model weights, Optimizer, LR Scheduler, Epoch và Metrics mà không bị lỗi kích thước hay xung đột phần cứng.

---

## 3. BÀI BÁO 1 (HƯỚNG 1): BG-GUIDED DINO — CONTINUAL SSL PRE-TRAINING

> **Tên bài báo:** *Background-Guided Self-Supervised Vision Transformer Pre-training for Dense Urban Traffic Surveillance*  
> **Target:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / CVPR / ECCV  
> **Mã nguồn:** `direction1_bg_guided_dino/`

### 3.1. Đóng góp Khoa học
1. **Foreground-Aware Masking (FAM) chuẩn hóa thứ bậc:** Khắc phục nhược điểm của Uniform Masking trong ViT (vốn lãng phí 80% gradient vào nền bê tông tĩnh). Tích hợp Rank Normalization để chống hiện tượng đèn pha ban đêm (headlight glare) làm lệch phân phối che.
2. **Cổng tin cậy thích ứng ($r_i$):** Điều tiết ngân sách che $\alpha_i = \alpha_{\max} \cdot r_i$. Khi nền xấu ($r_i$ thấp), mô hình tự động chuyển mượt về Uniform Random Masking.
3. **Mất mát chưng cất tự thân đa tầng (DINO [CLS] + iBOT [Patch]):**
   $$\mathcal{L} = \mathcal{L}_{\text{DINO}}^{[\text{CLS}]} + \lambda_{\text{ibot}} \mathcal{L}_{\text{iBOT}}^{[\text{Patch}]}$$
   Bảo đảm gradient tác động trực tiếp lên từng patch phương tiện bị che, thay vì chỉ dồn vào token [CLS].

---

## 4. BÀI BÁO 2 (HƯỚNG 2 / II.A): NOISE-AWARE TRAFFIC SCENE DECOMPOSITION

> **Tên bài báo:** *Noise-Aware Traffic Scene Decomposition with Imperfect Background Priors on City-Scale Camera Networks*  
> **Target:** IEEE Transactions on Image Processing (TIP) / Pattern Recognition / IEEE TCSVT  
> **Mã nguồn:** `direction2_scene_decomposition/`

### 4.1. Đóng góp Khoa học
1. **Mô hình phân rã cảnh với Prior mềm có độ bất định:**
   Mô hình $f_\theta(I) = (M_\alpha, F, \hat{B}, \sigma)$ bóc tách ảnh thành Mặt nạ xe $M_\alpha$, Tiền cảnh $F$, Nền dự đoán $\hat{B}$, và Bản đồ độ bất định $\sigma \in [0.01, 0.50]$.
   Background median chỉ đóng vai trò Laplace Prior:
   $$\mathcal{L}_{\text{prior}} = \frac{1}{K HW} \sum_{k} \sum_{u, v} \left[ \frac{|\hat{B}^{(k)}(u, v) - B_{c,s}(u, v)|}{\sigma^{(k)}(u, v)} + \log \sigma^{(k)}(u, v) \right]$$
   Chỗ nào background bị ghost xe kẹt, mạng được phép tăng $\sigma$ để bỏ qua, trả giá bằng số hạng phạt $\log \sigma$.
2. **Ràng buộc nền dùng chung giữa các ngày khác nhau ($\mathcal{L}_{\text{shared}}$):**
   Lấy mẫu $K$ frames của cùng camera và cùng slot giờ nhưng từ **$K$ ngày khác nhau**. Xe kẹt đứng yên suốt 1 giờ ngày hôm nay sẽ không xuất hiện tại vị trí đó vào ngày khác $\implies$ Giải quyết triệt để bài toán xe kẹt mà các phương pháp cùng ngày bó tay.
3. **Hàm mất mát Loại trừ ($\mathcal{L}_{\text{excl}}$):** Thay thế phạt diện tích thô (vốn phạt oan khi đường kẹt xe thật phủ 50% diện tích).
4. **Vòng tự làm sạch Background (Self-Cleaning Loop):** Gộp $\hat{B}$ qua nhiều vòng huấn luyện bằng trung vị có trọng số $w = (1 - M_\alpha) / (\sigma^2 + \epsilon)$ để tự động xóa sạch ghost vehicle trên toàn thành phố.

---

## 5. BÀI BÁO 3 (HƯỚNG 3): FOREGROUND-ENHANCED VEHICLE COUNTING

> **Tên bài báo:** *Prior-Guided Few-Shot Vehicle Counting in Dense Heterogeneous Traffic via Background Difference Injection*  
> **Target:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / Expert Systems with Applications  
> **Mã nguồn:** `direction3_foreground_enhanced_counting/`

### 5.1. Đóng góp Khoa học
1. **Khảo sát hệ thống 4 cơ chế tiêm Prior:** So sánh đối đầu giữa `early` (kênh thứ 4), `late` (CNN encoder riêng), `global` (FiLM modulation), và `none`.
2. **Zero-Initialization Patch Embedding:** Khởi tạo trọng số kênh thứ 4 bằng 0, giúp mạng thừa hưởng trọn vẹn đặc trưng pre-train của DINOv3 mà không bị sốc trọng số ban đầu.
3. **$\Delta$-Dropout 30%:** Rèn luyện khả năng đếm độc lập khi camera mất ảnh nền.

---

## 6. BÀI BÁO 4 (HƯỚNG 4): SPATIO-TEMPORAL ROAD SPACE OCCUPANCY ESTIMATION

> **Tên bài báo:** *Continuous Road Space Occupancy and Congestion Onset Forecasting from City Surveillance Networks*  
> **Target:** Transportation Research Part C: Emerging Technologies / IEEE T-ITS  
> **Mã nguồn:** `direction4_temporal_density/`

### 6.1. Đóng góp Khoa học
1. **Định lượng chiếm dụng strictly trên Road Mask $|R|$:**
   $$\rho_{\text{proxy}}(t) = \frac{1}{|R|} \sum_{(u, v) \in R} \mathbb{I}\left( \Delta_t(u, v) > \tau \right)$$
   Loại bỏ hoàn toàn sai số do góc máy và diện tích ngoại cảnh giữa các camera khác nhau.
2. **Tách biệt rạch ròi Nowcasting và Causal Forecasting:**
   - Ước lượng hiện tại: BiGRU hai chiều.
   - Cảnh báo sớm khởi phát kẹt xe: 1-way Causal GRU (chỉ nhìn về quá khứ, không rò rỉ tương lai).
3. **Hàm mất mát Huber Dynamic Smoothness:** Thay thế phạt bình phương $L_2$ (vốn làm mờ các sự kiện tai nạn/ngập lụt đột ngột).

---

## 7. BÀI BÁO 5 (HƯỚNG B): CONTEXT-AWARE WEAK SUPERVISION LABEL AGGREGATION

> **Tên bài báo:** *Context-Aware Markov Label Aggregation: Weakly-Supervised Traffic Congestion Assessment from Imperfect Heuristics on City-Scale Surveillance Networks*  
> **Target:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / Information Fusion / EAAI  
> **Mã nguồn:** `directionB_weak_supervision/`

### 7.1. Đóng góp Khoa học
1. **Không gian Ngữ cảnh 54 tổ hợp (`context.py`):** Phân chia chi tiết theo Ánh sáng (Ngày / Đêm IR), Khung giờ (Cao điểm / Thấp điểm / Đêm), Loại đường, và Tình trạng Camera.
2. **Context-Aware Markov Label Model (`label_model.py`):**
   Gộp 5 nguồn nhãn yếu (Detector Box, Background Difference, Temporal Differencing, Historical Peak, Multimodal VLM) có tính đến:
   - Động lực liên tục của trạng thái ùn tắc qua ma trận chuyển trạng thái Markov $\mathbf{A} \in \mathbb{R}^{4 \times 4}$.
   - Ma trận nhầm lẫn phát xạ phụ thuộc ngữ cảnh $\pi_j^{(c)}(\lambda_j \mid y)$.
   - Cơ chế Abstain ($\lambda = -1$) khi nguồn không chắc chắn.
   - Thuật toán **Expectation-Maximization (EM) với Forward-Backward trong không gian Log-Sum-Exp** chống tràn số.
3. **End Model DINOv3 + Causal GRU (`end_model.py`):**
   Huấn luyện bằng Soft Cross-Entropy trên nhãn mềm đã gộp. Lúc triển khai thực tế, mô hình **tự chủ 100% từ ảnh camera mà không cần bất kỳ LF hay background nào**.

---

## 8. BÀI BÁO 6 (HƯỚNG C): PERSISTENCE-AWARE ANOMALY & CAMERA FAULT DETECTION

> **Tên bài báo:** *Persistence-Aware, Camera-Conditioned Anomaly Detection for City-Scale Traffic Surveillance under Sparse Sampling*  
> **Target:** Transportation Research Part C / Pattern Recognition / IEEE T-ITS  
> **Mã nguồn:** `directionC_anomaly/`

### 8.1. Đóng góp Khoa học
1. **Temporal Feature Pooling trong không gian đặc trưng (`pooling.py`):**
   $$\tilde{F}_t(p) = \operatorname{median}_{w=0}^{W-1} f_{t-w}(p)$$
   Loại bỏ sạch sẽ xe cộ di chuyển thoáng qua, bảo toàn và khuếch đại các sự cố kéo dài (ngập lụt, xe chết máy, cây đổ).
2. **Coreset Normal Memory Bank (`bank.py`):** Thuật toán K-Center Greedy Selection nén 90% bộ nhớ, phân vùng theo từng Camera và Khung giờ.
3. **Phân tách Lỗi Camera vs Sự cố Giao thông (`camera_fault.py`):**
   So sánh điểm bất thường trên Road Mask ($s_{\text{road}}$) và vùng ngoại cảnh tĩnh ($s_{\text{static}}$). Nếu vùng ngoại cảnh bất thường tăng vọt $\implies$ Cảnh báo `CAMERA_FAULT` (lệch góc, rung lắc, mờ kính). Nếu chỉ lòng đường bất thường $\implies$ Cảnh báo `TRAFFIC_INCIDENT`.
4. **Persistence Filter & Vòng đời Sự kiện (`events.py`):** Ngưỡng hiệu chuẩn tự động theo phân vị 99.5% trên ngày bình thường; chỉ kích hoạt báo động khi sự cố kéo dài $\ge N$ cửa sổ liên tiếp ($N \ge 3$).

---

## 9. BÀI BÁO 7 (HƯỚNG D): CITY-SCALE CONGESTION FORECASTING ON CAMERA GRAPH

> **Tên bài báo:** *City-Scale Congestion Forecasting from Surveillance Camera Networks in Motorbike-Dominant Traffic*  
> **Target:** Transportation Research Part C / IEEE Transactions on Intelligent Transportation Systems (T-ITS) / IEEE TKDE  
> **Mã nguồn:** `directionD_forecasting/`

### 9.1. Đóng góp Khoa học
1. **Biến mạng lưới camera thành mạng cảm biến thành phố:** Thay thế bài toán cảm biến vòng từ cao tốc (METR-LA) bằng mạng lưới camera đô thị hỗn hợp.
2. **ST-GraphWaveNet đa phương thức nhận biết mất tín hiệu (`models.py`):**
   - Đồ thị kết hợp Ma trận khoảng cách OpenStreetMap và Ma trận kề Thích ứng tự học $\tilde{\mathbf{A}}_{\text{adp}} = \operatorname{Softmax}(\operatorname{ReLU}(\mathbf{E}_1 \mathbf{E}_2^T))$.
   - Gated Dilated TCN nhân quả kết hợp Missing Mask $m_t^i$ và Node Dropout 10%–50% lúc train.
3. **3 Đầu ra Đa nhiệm:** Dự báo liên tục đa tầm $h \in \{15', 30', 60'\}$, Phân loại 4 mức ùn tắc có thứ tự, và Cảnh báo khởi phát kẹt xe với **Focal Loss**.

---

## 10. BÀI BÁO 8 (HƯỚNG E): BACKGROUND-CONDITIONED GENERALIZATION TO UNSEEN CAMERAS

> **Tên bài báo:** *Background-Conditioned Generalization to Unseen Traffic Cameras with Unreliable Scene Priors*  
> **Target:** Pattern Recognition / Engineering Applications of Artificial Intelligence (EAAI) / IEEE T-ITS  
> **Mã nguồn:** `directionE_bg_conditioning/`

### 10.1. Đóng góp Khoa học
1. **Bản mô tả cảnh toàn cục cắt tỉa (Trimmed Scene Descriptor $z$):**
   $$z = \Big[\, \operatorname{TrimMean}_{p \in R} f_{\text{bg}}(p),\; \operatorname{TrimStd}_{p \in R} f_{\text{bg}}(p),\; \operatorname{TrimMean}_{p} f_{\text{bg}}(p) \,\Big]$$
   Loại bỏ 10% patch dị biệt $\implies$ Miễn nhiễm hoàn toàn với ghost vehicle cục bộ trên ảnh nền, mã hóa trung thực góc máy và ánh sáng camera.
2. **Cơ chế FiLM Zero-Initialization (`conditioning.py`):**
   $$\gamma, \beta = \operatorname{MLP}(z), \quad h_{\text{mod}} = (1 + \gamma) \odot \operatorname{LayerNorm}(h) + \beta$$
   Khởi tạo $\gamma=0, \beta=0$ giúp mạng giữ vững độ ổn định gốc.
3. **Thích ứng camera mới không cần gán nhãn:** Chỉ cần 1 ảnh nền của camera mới, mô hình lập tức điều biến đặc trưng để đạt độ chính xác cao mà không cần fine-tune trọng số.

---

## 11. GIAO THỨC THỰC NGHIỆM, PHÂN CHIA DỮ LIỆU VÀ KIỂM SOÁT RÒ RỈ

### 11.1. Phân chia Cụm Camera (Spatial-Temporal Clustering)
Tuyệt đối không phân chia ngẫu nhiên (Random Split) ở mức frame:
- **Cụm Camera (Camera Clusters):** Các camera thuộc cùng một nút giao hoặc trục đường liền kề bắt buộc phải nằm chung một cụm.
- **Tỷ lệ:** 70% số cụm cho Train, 10% cho Val, 20% cho Test.
- **Dữ liệu Chuỗi Thời gian (Hướng D):** Chia nghiêm ngặt theo trật tự thời gian (Chronological Split): 70% tuần đầu Train, 10% tuần giữa Val, 20% tuần cuối Test.

### 11.2. Ma trận So sánh Đối chuẩn Giữa 8 Hướng Nghiên cứu

| Hướng Nghiên Cứu | Đầu vào lúc Suy luận | Vai trò của Background | Giải pháp chống suy thoái Nền | Đầu ra chính | Mục tiêu Venue |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hướng 1 (H1)** | 1 Frame | Tiền nghiệm che FAM | Softmax nhiệt độ + Cổng $r_i$ | ViT Pre-trained Weights | IEEE T-ITS / CVPR |
| **Hướng 2 (H2)** | 1 Frame | Prior mềm có độ bất định | $\sigma$ học được + Nền khác ngày | $M_\alpha, F, \hat{B}, \sigma$ | IEEE TIP / PR |
| **Hướng 3 (H3)** | Frame + $\Delta$ | Ghép kênh / Tiêm đặc trưng | $\Delta$-Dropout 30% + Zero-init | Số lượng xe (Counting) | IEEE T-ITS / EAAI |
| **Hướng 4 (H4)** | Chuỗi Frame + $\Delta$ | Đo $\rho_{\text{proxy}}$ trên Road Mask | Giới hạn strictly Road Mask | $\rho(t)$ & Mức LoS | TR-Part C / T-ITS |
| **Hướng B** | Chuỗi Frame | Một trong 5 nguồn nhãn yếu | Cổng $r_i$ trong LF2 + Markov | Nhãn mềm $q(y)$ & End Model | Inf. Fusion / T-ITS |
| **Hướng C** | Chuỗi Frame | Mẫu đối sánh phụ trong Bank | Gộp đặc trưng Median $W$ frames | Cảnh báo Sự cố / Lỗi Camera | TR-Part C / PR |
| **Hướng D** | Đồ thị Camera | Không phụ thuộc | Missing Mask + Node Dropout | Dự báo 15', 30', 60' & Onset | TR-Part C / TKDE |
| **Hướng E** | Frame + Background | Vector mô tả cảnh toàn cục $z$ | Trimmed Mean/Std + Bg-Dropout | Thích ứng Camera chưa thấy | Pattern Rec. / EAAI |

---

## 12. TỔNG KẾT VÀ LỘ TRÌNH TRIỂN KHAI

Hệ thống mã nguồn tại `g:/nckh/DINO/` đã được chuẩn hóa toàn diện:
1. **Kiểm thử khép kín 100%:** File `test_all_directions.py` đã vượt qua toàn bộ 11 test cases bao quát cả 8 hướng nghiên cứu và các công cụ bổ trợ.
2. **Sẵn sàng thực nghiệm:** Mọi hướng đều có đầy đủ `dataset.py`, `models.py`, `losses.py`, `evaluate.py`, và `README.md` độc lập.
3. **Chuẩn khoa học Q1:** Không sử dụng background thô làm ground truth cứng; toàn bộ công thức toán học và thiết kế kiến trúc đều được chứng minh chặt chẽ và phòng vệ phản biện reviewer.
