# TÀI LIỆU ĐỀ XUẤT CHUNG: HỆ SINH THÁI NGHIÊN CỨU DINO TRAFFIC SUITE CHO CAMERA GIAO THÔNG ĐÔ THỊ TP.HCM

> **Dự án:** Nghiên cứu và Triển khai Nền tảng Thị giác Máy tính Tự giám sát Cho Giám sát Giao thông Đô thị Thông minh  
> **Địa bàn & Dữ liệu:** Mạng lưới camera giao thông TP.HCM (>600 camera CCTV, mật độ xe máy chiếm 70–80%, chu kỳ chụp thưa 3–5 phút/frame)  
> **Triết lý khoa học cốt lõi:** **Tách bạch bản chất biến thiên của phương tiện khỏi đa tạp nền tĩnh; Tự học hệ cơ sở nền không cần Prior (Prior-Free) hoặc coi Background Median là tín hiệu yếu có nhiễu (Noisy Weak Prior).**  
> **Kho mã nguồn:** `g:/nckh/DINO/`  

---

## 1. TỔNG QUAN HỆ SINH THÁI TINH GỌN VÀ BẢN ĐỒ NGHIÊN CỨU

Hệ sinh thái DINO Traffic Suite được tái cấu trúc tinh gọn, tập trung nguồn lực vào 2 hướng nghiên cứu mũi nhọn đột phá tự học không cần nền, 2 hệ thống baseline đối chuẩn có nền và 1 công trình dữ liệu quy mô thành phố:

```
+----------------------------------------------------------------------------------------------------+
|                                 DINO TRAFFIC SUITE CORE ECOSYSTEM                                  |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   +--------------------------------------------------------------------------------------------+   |
|   |                       TẦNG NỀN TẢNG & KIỂM CHỨNG CHUNG (common/)                           |   |
|   |  - Static Reliability Estimator (r_i)            - Phase Correlation Camera Alignment      |   |
|   |  - Background Degradation Benchmark (BDB)        - Frame Corruption Suite (FCS)            |   |
|   |  - Multi-GPU Smart Checkpointing                 - PyTorch DDP / DataParallel Pipeline     |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                 │                                                  |
|         ┌───────────────────────────────────────┼───────────────────────────────────────┐          |
|         ▼                                       ▼                                       ▼          |
|   [SSL MŨI NHỌN (PRIOR-FREE)]         [WEAK SUPERVISION & ANOMALY]          [BASELINES & DỮ LIỆU]  |
|   1. Hướng 1 Mới (direction1_new):    3. Hướng 5 (direction5_weak):         5. Hướng 1 Cũ (H1):    |
|      Vehicle-Centric SSL                 Context-Aware Markov Label Agg        Continual FAM-Δ     |
|   2. Hướng 2 Mới (direction2_new):    4. Hướng 6 (direction6_anomaly):      6. Hướng 2 Cũ (H2):    |
|      Prior-Free Scene Decomposition      Persistence Anomaly & Camera Fault    Noise-Aware Decomp  |
|                                                                             7. Data Article Q1     |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. DANH MỤC CHI TIẾT CÁC HƯỚNG NGHIÊN CỨU TRỌNG TÂM

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

### 2.3 Hướng 5: Context-Aware Weak Supervision Cho Giám Sát Giao Thông (`direction5_weak_supervision/`)
- **Tên khoa học:** *Context-Aware Markov Label Aggregation: Weakly-Supervised Traffic Congestion Assessment from Imperfect Heuristics on City-Scale Surveillance Networks*
- **Độ mới:** ⭐⭐⭐⭐⭐ (5/5)
- **Tạp chí mục tiêu:** IEEE T-ITS / NeurIPS / CVPR
- **Đóng góp:**
  1. Khai thác 5 hàm sinh nhãn yếu (Detector Bounding Box, Background Difference, Temporal Differencing, Historical Peak Profile, Multimodal VLM).
  2. Phân loại 54 ngữ cảnh đô thị (Ánh sáng x Khung giờ x Cấp đường x Độ tin cậy camera $r_i$).
  3. Mô hình đồ thị xác suất Markov ẩn với thuật toán EM trong không gian log-sum-exp, triệt tiêu hoàn toàn tràn số underflow.
  4. Huấn luyện mô hình đích (DINOv3 ViT + Causal GRU) qua Soft Cross-Entropy; khi suy luận hoạt động độc lập 100%, không cần LF hay nền.

### 2.4 Hướng 6: Phát Hiện Sự Cố Bất Thường Kéo Dài & Bóc Tách Lỗi Camera (`direction6_anomaly_detection/`)
- **Tên khoa học:** *Persistence-Aware, Camera-Conditioned Anomaly Detection for City-Scale Traffic Surveillance under Sparse Sampling*
- **Độ mới:** ⭐⭐⭐⭐⭐ (5/5)
- **Tạp chí mục tiêu:** IEEE T-ITS / Transportation Research Part C / Pattern Recognition
- **Đóng góp:**
  1. Temporal Median Feature Pooling qua cửa sổ trượt $W$ khung hình trên patch tokens DINOv3 ($d=128$), triệt tiêu hoàn toàn xe di chuyển thoáng qua, bảo toàn sự cố kéo dài (ngập lụt, tai nạn, rào chắn).
  2. Coreset Normal Memory Bank nén 90% bằng K-Center Greedy (PatchCore-style), phân vùng theo camera và khung giờ.
  3. Bóc tách lỗi kỹ thuật camera ($s_{\text{static}}$ ngoài đường) và sự cố giao thông ($s_{\text{road}}$ trong lòng đường).
  4. Persistence Filtering với ngưỡng phân vị 99.5% khống chế báo động sai $< 1$ lần/camera/ngày.

### 2.5 Hướng 1 Cũ (Baseline): Continual SSL với Foreground-Aware Masking (`direction1_bg_guided_dino/`)
- **Tên khoa học:** *Background-Guided Self-Supervised Vision Transformer Pre-training for Dense Urban Traffic Surveillance*
- **Độ mới:** ⭐⭐⭐⭐ (4/5)
- **Tạp chí mục tiêu:** IEEE T-ITS
- **Đóng góp:** FAM-$\Delta$ chuẩn hóa thứ bậc kết hợp cổng tin cậy thích ứng $r_i$ làm mỏ neo chuyển mượt về Uniform Masking khi nền xấu; làm baseline đối sánh trực tiếp với Hướng 1 Mới.

### 2.6 Hướng 2 Cũ (Baseline): Phân Rã Cảnh Dùng Median Prior (`direction2_scene_decomposition/`)
- **Tên khoa học:** *Noise-Aware Traffic Scene Decomposition with Imperfect Background Priors on City-Scale Camera Networks*
- **Độ mới:** ⭐⭐⭐⭐ (4/5)
- **Tạp chí mục tiêu:** Pattern Recognition / IEEE TCSVT
- **Đóng góp:** Mô hình hóa Alpha Compositing với Laplace Prior trên ảnh nền median và cơ chế phạt $\log \sigma$, kết hợp ràng buộc nền dùng chung giữa các ngày khác nhau $\mathcal{L}_{\text{shared}}$.

### 2.7 Bài Báo Dữ Liệu: IC4SD-TrafficSnap (`direction_data_article/`)
- **Tên bài báo:** *IC4SD-TrafficSnap: A Multi-Modal Dataset of Sparse Surveillance Imagery and Road Network Topology for Urban Traffic Analysis in Ho Chi Minh City*
- **Mục tiêu tạp chí:** Elsevier Data in Brief
- **Quy mô:** 608 trạm camera, 714,123 ảnh JPEG, 44.38 GiB, đồ thị OSRM 2,450 cạnh có hướng, kiểm toán bảo mật PII chuẩn mực Rule of Three ($p \le 0.30\%$).

---

## 3. MA TRẬN ĐỐI CHUẨN KỸ THUẬT

| Hướng Nghiên Cứu | Thư Mục Mã Nguồn | Đầu Vào Lúc Suy Luận | Vai Trò Background | Cơ Chế Chống Suy Thoái Nền | Tạp Chí Mục Tiêu |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H1 Mới** | `direction1_new/` | 1 Frame | **Không cần nền** | TAM + SRS hoán đổi đa ngày | IEEE TPAMI / CVPR |
| **H2 Mới** | `direction2_new/` | 1 Frame | **Không cần nền** | SceneBasis đa chiếu sáng + $\sigma$ | IEEE TIP / PR |
| **H1 Cũ** | `direction1_bg_guided_dino/` | 1 Frame | Tiền nghiệm che FAM | Cổng tin cậy thích ứng $r_i$ | IEEE T-ITS |
| **H2 Cũ** | `direction2_scene_decomposition/`| 1 Frame | Laplace Prior mềm | Bản đồ bất định $\sigma$ + Nền đa ngày | PR / TCSVT |
| **H5** | `direction5_weak_supervision/` | Chuỗi Frame | LF2 chênh lệch nền | Markov Label Model 54 ngữ cảnh + EM | IEEE T-ITS / NeurIPS |
| **H6** | `direction6_anomaly_detection/` | Cửa sổ $W$ | Không cần nền mốc | Median Feature Pooling + Coreset Bank | IEEE T-ITS / TR-C |
| **Data Article** | `direction_data_article/` | Census Dữ liệu | Đồ thị + Ảnh thưa | Multi-tier Audit + Rule of Three | Elsevier DiB |

---

## 4. TÌNH TRẠNG MÃ NGUỒN VÀ BẢO ĐẢM KHOA HỌC

1. **Kiểm thử khép kín 100%:** Test suite tại `tests/test_all_directions.py` đã xác nhận toàn bộ các test cases cho các hướng nghiên cứu trọng tâm đều vượt qua thành công:
   - Test 1: Common Utilities (TrafficPairMatcher & BackgroundSubtractor)
   - Test 2: Direction 1 Cũ (BG-Guided DINO SSL Baseline)
   - Test 3: Direction 2 Cũ (Scene Decomposition Baseline)
   - Test 6: Multi-GPU Smart Checkpointing Interoperability
   - Test 7: Common Advanced (Reliability $r_i$, BDB Degradation & FCS Frame Corruptions)
   - Test 8: Direction 5 (Context-Aware Weak Supervision)
   - Test 9: Direction 6 (Persistence Anomaly Detection)
   - Test 12: Direction 1 Mới (Vehicle-Centric SSL Pretraining TAM + AGM + SRS)
   - Test 13: Direction 2 Mới (Prior-Free Scene Decomposition SceneBasis + Huber IRLS + Loss V2)
2. **Tuân thủ tuyệt đối quy chuẩn:** 
   - Mã nguồn chuẩn production, chú thích tiếng Việt chuyên sâu, kiểm tra chặt chẽ biến và thư viện.
   - Thư mục `tests/` nằm trong `.gitignore` không đẩy lên repo.
   - Giữ nguyên vẹn tính độc lập và số liệu của Bài báo Dữ liệu (`direction_data_article`).
