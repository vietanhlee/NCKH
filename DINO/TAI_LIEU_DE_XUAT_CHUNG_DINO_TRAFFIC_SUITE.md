# TÀI LIỆU ĐỀ XUẤT CHUNG: HỆ SINH THÁI NGHIÊN CỨU DINO TRAFFIC SUITE CHO CAMERA GIAO THÔNG ĐÔ THỊ TP.HCM

> **Dự án:** Nghiên cứu và Triển khai Nền tảng Thị giác Máy tính Tự giám sát Cho Giám sát Giao thông Đô thị Thông minh  
> **Địa bàn & Dữ liệu:** Mạng lưới camera giao thông TP.HCM (>600 camera CCTV, mật độ xe máy chiếm 70–80%, chu kỳ chụp thưa 3–5 phút/frame)  
> **Triết lý khoa học cốt lõi:** **Tách bạch bản chất biến thiên của phương tiện khỏi đa tạp nền tĩnh; coi Background Median là tín hiệu yếu có nhiễu (Noisy Weak Prior) hoặc tự học hệ cơ sở nền không cần Prior.**  
> **Kho mã nguồn:** `g:/nckh/DINO/`  

---

## 1. TỔNG QUAN HỆ SINH THÁI VÀ BẢN ĐỒ NGHIÊN CỨU

Hệ sinh thái DINO Traffic Suite được thiết kế theo kiến trúc module hóa phân tầng, bao gồm 2 hướng mũi nhọn đột phá tự học không cần nền, 6 hướng ứng dụng thực tiễn downstream và 1 công trình dữ liệu quy mô thành phố:

```
+----------------------------------------------------------------------------------------------------+
|                                    DINO TRAFFIC SUITE ECOSYSTEM                                    |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   +--------------------------------------------------------------------------------------------+   |
|   |                       TẦNG NỀN TẢNG & KIỂM CHỨNG CHUNG (common/)                           |   |
|   |  - Static Reliability Estimator (r_i)            - Phase Correlation Camera Alignment      |   |
|   |  - Background Degradation Benchmark (BDB)        - Frame Corruption Suite (FCS)            |   |
|   |  - Multi-GPU Smart Checkpointing                 - PyTorch DDP / DataParallel Pipeline     |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                 │                                                  |
|         ┌───────────────────────────────────────┴───────────────────────────────────────┐          |
|         ▼                                                                               ▼          |
|   [NHÓM SSL MŨI NHỌN & PHÂN RÃ CẢNH]                           [NHÓM DOWNSTREAM & DỮ LIỆU ĐÔ THỊ]  |
|   1. Hướng 1 Mới (direction1_new):                             5. Hướng 3 (direction3_counting):   |
|      Vehicle-Centric SSL (TAM + AGM + SRS)                        Foreground-Enhanced Counting 4ch |
|   2. Hướng 2 Mới (direction2_new):                             6. Hướng 4 (direction4_density):    |
|      Prior-Free Scene Decomposition (SceneBasis + σ)              Spatio-Temporal Density & LoS    |
|   3. Hướng 1 Cũ (direction1_bg_guided_dino):                   7. Hướng 6 (direction6_anomaly):    |
|      Continual SSL với FAM-Δ (Baseline Prior)                     Persistence Anomaly & Cam Fault  |
|   4. Hướng 2 Cũ (direction2_scene_decomposition):              8. Hướng 7 (direction7_forecasting):|
|      Scene Decomposition với Median Prior                         ST-GraphWaveNet City Forecasting |
|                                                                9. Hướng 8 (direction8_conditioning)|
|                                                                   FiLM Generalization to New Cams  |
|                                                               10. Data Article (data_article/):    |
|                                                                   IC4SD-TrafficSnap (Elsevier DiB) |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. DANH MỤC CHI TIẾT CÁC HƯỚNG NGHIÊN CỨU VÀ MỤC TIÊU CÔNG BỐ

### 2.1 Hướng 1 Mới: Vehicle-Centric SSL Pre-training Không Cần Nền (`direction1_new/`)
- **Tên khoa học:** *Vehicle-Centric Representation Learning from Sparse Static-Camera Feeds via Long-Term Temporal Atypicality and Counterfactual Background Swapping*
- **Độ mới:** ⭐⭐⭐⭐⭐ (5/5)
- **Tạp chí mục tiêu:** IEEE TPAMI / CVPR / ECCV / IEEE T-ITS
- **Đột phá:** 
  1. **TAM (Temporal Atypicality Map):** Ước lượng xác suất tiền cảnh $\pi(p)$ ở cấp độ patch qua thống kê trực tuyến $K=4$ cụm trạng thái và GMM mà không cần ảnh nền pixel.
  2. **AGM (Atypicality-Guided Masking):** Che phân tầng có kiểm soát ngân sách xe $\phi = 0.5$ và khống chế trần $q_{\max} = 0.6$.
  3. **SRS (Static-Region Swap):** Hoán đổi patch nền tĩnh giữa các ngày khác nhau cùng camera, triệt tiêu Background Shortcut trap.
  4. Hàm mất mát chưng cất tự thân: DINO [CLS] + iBOT [Patch] có trọng số $\pi$ + KoLeo Regularizer.

### 2.2 Hướng 2 Mới: Phân Rã Cảnh Giao Thông Không Cần Nền (`direction2_new/`)
- **Tên khoa học:** *Prior-Free Traffic Scene Decomposition via Multi-Illumination Manifold Learning and Uncertainty-Aware Composite Reconstruction*
- **Độ mới:** ⭐⭐⭐⭐⭐ (5/5)
- **Tạp chí mục tiêu:** IEEE TIP / Pattern Recognition / IEEE TCSVT
- **Đột phá:**
  1. Tự học hệ cơ sở quang học `SceneBasis` $B(t) = E_0 + \sum_{j=1}^J \ell_j(t) E_j$ từ chuỗi ảnh nhiều ngày không nhãn, không phụ thuộc ảnh nền median.
  2. Bộ giải trực tuyến $\boldsymbol{\ell}(t)$ qua Least Squares / Robust Huber-IRLS trên vùng tĩnh $(1-\alpha)^2$.
  3. Mạng phân rã bóc tách đồng thời: $(\hat{I}_{\text{recon}}, \hat{B}, \hat{F}, \alpha, \sigma, \boldsymbol{\ell})$.
  4. Hàm mất mát Laplace Negative Log-Likelihood với bản đồ độ bất định $\sigma(u, v)$ tự học, chịu đựng hoàn hảo lóa đèn ban đêm và vùng giao cắt phức tạp.

### 2.3 Hướng 1 Cũ: Continual SSL với Foreground-Aware Masking (`direction1_bg_guided_dino/`)
- **Tên khoa học:** *Background-Guided Self-Supervised Vision Transformer Pre-training for Dense Urban Traffic Surveillance*
- **Độ mới:** ⭐⭐⭐⭐ (4/5)
- **Tạp chí mục tiêu:** IEEE T-ITS / EAAI
- **Đóng góp:** FAM-$\Delta$ chuẩn hóa thứ bậc kết hợp cổng tin cậy thích ứng $r_i$ làm mỏ neo chuyển mượt về Uniform Masking khi nền xấu; làm baseline đối sánh trực tiếp với Hướng 1 Mới.

### 2.4 Hướng 2 Cũ: Phân Rã Cảnh Dùng Median Prior (`direction2_scene_decomposition/`)
- **Tên khoa học:** *Noise-Aware Traffic Scene Decomposition with Imperfect Background Priors on City-Scale Camera Networks*
- **Độ mới:** ⭐⭐⭐⭐ (4/5)
- **Tạp chí mục tiêu:** Pattern Recognition / IEEE TCSVT
- **Đóng góp:** Mô hình hóa Alpha Compositing với Laplace Prior trên ảnh nền median và cơ chế phạt $\log \sigma$, kết hợp ràng buộc nền dùng chung giữa các ngày khác nhau $\mathcal{L}_{\text{shared}}$.

### 2.5 Hướng 3: Ước Lượng Lưu Lượng Phương Tiện Ít Mẫu 4 Kênh (`direction3_foreground_enhanced_counting/`)
- **Tên khoa học:** *Prior-Guided Few-Shot Vehicle Counting in Dense Heterogeneous Traffic via Background Difference Injection*
- **Độ mới:** ⭐⭐⭐½ (3.5/5)
- **Tạp chí mục tiêu:** EAAI / ITSC
- **Đóng góp:** Khảo sát 4 cơ chế tiêm prior (early 4th channel, late CNN, global FiLM, none); khởi tạo trọng số Zero-Init Patch Embedding; áp dụng $\Delta$-Dropout 30% để mô hình tự chủ hoàn toàn khi mất tín hiệu nền.

### 2.6 Hướng 4: Dự Báo Mật Độ Không-Thời Gian và Phân Loại LoS (`direction4_temporal_density/`)
- **Tên khoa học:** *Continuous Road Space Occupancy and Congestion Onset Forecasting from City Surveillance Networks*
- **Độ mới:** ⭐⭐⭐⭐½ (4.5/5)
- **Tạp chí mục tiêu:** Transportation Research Part C / IEEE T-ITS
- **Đóng góp:** Định lượng tỷ lệ chiếm dụng $\rho(t)$ giới hạn nghiêm ngặt trên Road Mask $|R|$; tách bạch mạng hai chiều Bi-GRU (nowcasting) và mạng nhân quả Causal GRU (forecasting); hàm mất mát động Huber Smoothness.

### 2.7 Hướng 6: Phát Hiện Sự Cố và Lỗi Camera Theo Tính Kiên Định (`direction6_anomaly_detection/`)
- **Tên khoa học:** *Persistence-Aware, Camera-Conditioned Anomaly Detection for City-Scale Traffic Surveillance under Sparse Sampling*
- **Độ mới:** ⭐⭐⭐⭐ (4/5)
- **Tạp chí mục tiêu:** Transportation Research Part C / Pattern Recognition
- **Đóng góp:** Gộp đặc trưng thời gian Temporal Median Pooling trên cửa sổ $W$; Coreset Normal Memory Bank phân vùng camera và khung giờ; phân tách lỗi camera (ngoại cảnh tĩnh) vs sự cố giao thông (lòng đường); bộ lọc kiên định $\ge N$ cửa sổ liên tiếp.

### 2.8 Hướng 7: Dự Báo Ùn Tắc Quy Mô Thành Phố Trên Đồ Thị Camera (`direction7_traffic_forecasting/`)
- **Tên khoa học:** *City-Scale Congestion Forecasting from Surveillance Camera Networks in Motorbike-Dominant Traffic*
- **Độ mới:** ⭐⭐⭐⭐½ (4.5/5)
- **Tạp chí mục tiêu:** Transportation Research Part C / IEEE TKDE
- **Đóng góp:** Mạng ST-GraphWaveNet đa phương thức tích hợp ma trận khoảng cách vật lý OpenStreetMap và ma trận kề thích ứng tự học $\tilde{\mathbf{A}}_{\text{adp}}$; Gated Dilated TCN nhân quả kết hợp Missing Mask và Node Dropout; 3 đầu ra đa nhiệm (hồi quy đa tầm, phân loại cấp độ LoS và cảnh báo khởi phát kẹt xe bằng Focal Loss).

### 2.9 Hướng 8: Thích Ứng Camera Mới Qua Background Conditioning (`direction8_bg_conditioning/`)
- **Tên khoa học:** *Background-Conditioned Generalization to Unseen Traffic Cameras with Unreliable Scene Priors*
- **Độ mới:** ⭐⭐⭐⭐ (4/5)
- **Tạp chí mục tiêu:** Pattern Recognition / EAAI
- **Đóng góp:** Vector mô tả cảnh toàn cục cắt tỉa (Trimmed Scene Descriptor $z$) kháng 10% outlier; cơ chế FiLM Zero-Initialization điều biến đặc trưng; zero-shot transfer sang camera mới chỉ với 1 ảnh nền không cần fine-tune.

### 2.10 Bài Báo Dữ Liệu: IC4SD-TrafficSnap (`direction_data_article/`)
- **Tên bài báo:** *IC4SD-TrafficSnap: A Multi-Modal Dataset of Sparse Surveillance Imagery and Road Network Topology for Urban Traffic Analysis in Ho Chi Minh City*
- **Mục tiêu tạp chí:** Elsevier Data in Brief (Đã hoàn thiện bản thảo 26 trang `paper/main.tex`, 5 bảng LaTeX và 5 hình trích xuất từ dữ liệu thực tế).
- **Quy mô:** 608 trạm camera, 714,123 ảnh JPEG, 44.38 GiB, đồ thị OSRM 2,450 cạnh có hướng, kiểm toán bảo mật PII chuẩn mực Rule of Three ($p \le 0.30\%$).

---

## 3. TỔNG HỢP MA TRẬN PHÂN LOẠI VÀ ĐỐI CHUẨN KỸ THUẬT

| STT | Hướng Nghiên Cứu | Thư Mục Mã Nguồn | Đầu Vào Lúc Suy Luận | Vai Trò Background | Cơ Chế Chống Suy Thoái Nền |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Hướng 1 Mới** | `direction1_new/` | 1 Frame | **Không cần nền** | TAM + SRS hoán đổi đa ngày |
| **2** | **Hướng 2 Mới** | `direction2_new/` | 1 Frame | **Không cần nền** | SceneBasis đa chiếu sáng + $\sigma$ |
| **3** | Hướng 1 Cũ | `direction1_bg_guided_dino/` | 1 Frame | Tiền nghiệm che FAM | Cổng tin cậy thích ứng $r_i$ |
| **4** | Hướng 2 Cũ | `direction2_scene_decomposition/`| 1 Frame | Laplace Prior mềm | Bản đồ bất định $\sigma$ + Nền đa ngày |
| **5** | Hướng 3 | `direction3_foreground_.../` | Frame + $\Delta$ | Ghép kênh thứ 4 / FiLM | $\Delta$-Dropout 30% + Zero-Init |
| **6** | Hướng 4 | `direction4_temporal_density/` | Chuỗi Frame + $\Delta$ | Đo $\rho_{\text{proxy}}$ trên Road Mask| Giới hạn strictly Road Mask $|R|$ |
| **7** | Hướng 6 | `direction6_anomaly_.../` | Chuỗi Frame | Mẫu phụ trong Memory Bank | Gộp đặc trưng Median $W$ frames |
| **8** | Hướng 7 | `direction7_traffic_.../` | Đồ thị Camera | Không phụ thuộc | Missing Mask + Node Dropout |
| **9** | Hướng 8 | `direction8_bg_conditioning/` | Frame + Background | Vector mô tả cảnh $z$ | Trimmed Mean/Std + Bg-Dropout |

---

## 4. TÌNH TRẠNG MÃ NGUỒN VÀ BẢO ĐẢM KHOA HỌC

1. **Kiểm thử khép kín 100%:** Test suite tại `tests/test_all_directions.py` đã xác nhận 13/13 test cases vượt qua thành công, bao gồm cả các module mới `direction1_new` và `direction2_new`.
2. **Tuân thủ quy tắc người dùng:** 
   - Mã nguồn chuẩn production, chú thích tiếng Việt chuyên sâu, kiểm tra chặt chẽ khai báo biến và thư viện.
   - Các file test được tổ chức gọn gàng trong thư mục `tests/` và được đưa vào `.gitignore`.
   - Giữ nguyên vẹn tính độc lập và số liệu của Bài báo Dữ liệu (`direction_data_article`).
