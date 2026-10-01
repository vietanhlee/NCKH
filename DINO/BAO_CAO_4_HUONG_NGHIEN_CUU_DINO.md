# BÁO CÁO KHOA HỌC: PHƯƠNG PHÁP LUẬN VÀ THIẾT KẾ KIẾN TRÚC 8 HƯỚNG NGHIÊN CỨU KHAI THÁC CẶP ẢNH NỀN TĨNH VÀ PHƯƠNG TIỆN TRONG THỊ GIÁC GIAO THÔNG

**Dự án:** Khai thác tự giám sát cặp ảnh Background tĩnh và Origin phương tiện phục vụ bài toán thị giác máy tính giám sát giao thông đô thị  
**Dữ liệu thực nghiệm:** Hệ thống camera giao thông đô thị TP.HCM (IC4SD-Traffic-HCM)  
**Địa chỉ mã nguồn:** Thư mục `g:/nckh/DINO/`

---

## 1. TỔNG QUAN HỆ THỐNG VÀ BẢN CHẤT DỮ LIỆU ĐẦU VÀO

Trong giám sát giao thông qua camera cố định (CCTV), hệ thống có hai nguồn tín hiệu hình ảnh tự nhiên cùng góc quan sát:
1. **Ảnh nguồn có phương tiện ($I_{\text{origin}} \in \mathbb{R}^{H \times W \times 3}$):** Chứa các đối tượng động (xe máy, ô tô, xe tải, xe buýt, người đi bộ) di chuyển trên nền đường đô thị phức tạp. Được thu thập theo định dạng `output/{stt}_{timestamp}.jpg`.
2. **Ảnh nền tĩnh sạch bóng xe ($I_{\text{bg}} \in \mathbb{R}^{H \times W \times 3}$):** Được tổng hợp thông qua thuật toán lọc trung vị theo thời gian (Temporal Median Filtering) kết hợp chuẩn hóa độ sáng không gian màu HSV trên từng camera và từng slot giờ ($00\text{h} - 23\text{h}$ theo giờ Việt Nam UTC+7). Được lưu trữ tại `traffic_backgrounds/route_{stt}/background_slot_{slot}h.jpg`.

Sự kết hợp giữa $I_{\text{origin}}$ và $I_{\text{bg}}$ cung cấp một tín hiệu vật lý quang học tiên nghiệm (Physical Prior) vô cùng mạnh mẽ:
$$\Delta(u, v) = \|I_{\text{origin}}(u, v) - I_{\text{bg}}(u, v)\|$$

Thay vì bỏ phí ảnh nền tĩnh chỉ để xem trực quan, hệ thống DINO Suite được xây dựng gồm **8 hướng nghiên cứu** độc lập nhưng bổ trợ lẫn nhau, tận dụng $\Delta$ và $I_{\text{bg}}$ để giải quyết triệt để các bài toán cốt lõi của thị giác máy tính trong giao thông thông minh (Intelligent Transportation Systems - ITS).

---

## 2. HƯỚNG 1: BG-GUIDED DINO (CONTINUAL SELF-SUPERVISED PRE-TRAINING)

### 2.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction1_bg_guided_dino/`
* Bộ nạp dữ liệu và biến đổi Multi-Crop: [`dataset.py`](file:///g:/nckh/DINO/direction1_bg_guided_dino/dataset.py) (`BGGuidedDINODataset`, `MultiCropBGGuidedAugmentation`)
* Kiến trúc mạng Student - Teacher: [`models.py`](file:///g:/nckh/DINO/direction1_bg_guided_dino/models.py) (`BGGuidedDINOModel`, `DINOHead`)
* Hàm mất mát tự chưng cất: [`losses.py`](file:///g:/nckh/DINO/direction1_bg_guided_dino/losses.py) (`BGGuidedDINOLoss`)
* Quy trình huấn luyện: [`train.py`](file:///g:/nckh/DINO/direction1_bg_guided_dino/train.py)
* Tiện ích trích xuất bản đồ sai khác: [`common/subtraction.py`](file:///g:/nckh/DINO/common/subtraction.py) (`BackgroundSubtractor.compute_patch_mask_weights`)

### 2.2. Đặt vấn đề và Mục tiêu
Trong các phương pháp học tự giám sát (Self-Supervised Learning - SSL) hiện đại dựa trên kiến trúc Vision Transformer (ViT) như DINOv2 (Oquab et al., TMLR 2024), DINOv3, hoặc Masked Autoencoders (He et al., CVPR 2022), cơ chế che ảnh (Masking) được thực hiện hoàn toàn ngẫu nhiên đồng đều (Uniform Random Masking).

Tuy nhiên, trong miền ảnh camera giao thông đô thị:
* Hơn 70% đến 80% diện tích bề mặt khung hình là nền đường tĩnh, vạch sơn, dải phân cách bê tông, vỉa hè hoặc bầu trời.
* Các đối tượng phương tiện (mang giá trị thông tin ngữ nghĩa cao nhất) chỉ chiếm từ 20% đến 30% diện tích và thường phân bố cục bộ.
* Nếu áp dụng Masking ngẫu nhiên đồng đều, đa số các patch bị che sẽ rơi vào nền đường nhựa. Mô hình ViT bị lãng phí phần lớn dung lượng biểu diễn (Representational Capacity) và gradient cập nhật chỉ để học tái tạo những mảng bê tông vô nghĩa, làm chậm quá trình hội tụ và giảm độ nhạy với đặc trưng phân biệt phương tiện.

**Mục tiêu:** Xây dựng cơ chế che có định hướng vùng tiền cảnh (Foreground-Aware Masking - FAM) kết hợp kiến trúc tự chưng cất tri thức Multi-Crop của DINOv3, ép Vision Transformer phải tập trung học biểu diễn hình thái, đường biên và ngữ nghĩa của xe cộ mà hoàn toàn không cần con người gán nhãn thủ công (Zero human annotation).

### 2.3. Cơ sở lý thuyết và Phân tích công thức xác suất che Foreground-Aware Masking (FAM)

#### 2.3.1. Tính toán trọng số mức độ sai khác theo Patch
Ảnh gốc $I_{\text{origin}}$ và ảnh nền $I_{\text{bg}}$ được đưa về cùng kích thước và tính toán bản đồ sai khác trong không gian màu CIE-LAB để giảm thiểu ảnh hưởng của sự thay đổi cường độ ánh sáng:
$$\Delta(u, v) = 0.5 \cdot \frac{|L_{\text{origin}} - L_{\text{bg}}|}{255} + 0.25 \cdot \frac{|a_{\text{origin}} - a_{\text{bg}}|}{255} + 0.25 \cdot \frac{|b_{\text{origin}} - b_{\text{bg}}|}{255}$$

Ảnh được chia thành lưới gồm $N_{\text{patches}} = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right)$ patches rời rạc, với $P = 16$ là kích thước cạnh của một patch ViT. Đối với mỗi patch $p$, trọng số chuyển động trung bình $w_p$ được định nghĩa:
$$w_p = \frac{1}{P^2} \sum_{(u, v) \in \text{Patch}_p} \Delta(u, v)$$

#### 2.3.2. Phân tích nguồn gốc và ý nghĩa công thức xác suất che
Xác suất để một patch $p$ được lựa chọn đưa vào danh sách bị che (Masked) được mô hình hóa theo công thức nội suy tuyến tính:
$$P_{\text{mask}}(p) = \alpha \cdot \frac{w_p}{\max_{q} w_q + \epsilon} + (1 - \alpha) \cdot \frac{1}{N_{\text{patches}}}$$
Trong đó:
* $w_p$: Trọng số sai khác đo được tại patch $p$.
* $\max_q w_q$: Giá trị sai khác lớn nhất trên toàn bộ các patch của ảnh hiện tại.
* $\epsilon = 10^{-8}$: Hệ số chống chia cho 0 nhằm đảm bảo ổn định số học.
* $N_{\text{patches}}$: Tổng số lượng patch của ảnh (với kích thước $224 \times 224$ và $P=16$, $N_{\text{patches}} = 14 \times 14 = 196$).
* $\alpha \in [0, 1]$: Tham số điều tiết cân bằng (Trade-off Hyperparameter), được chọn mặc định là $\alpha = 0.75$.

**Tại sao lại xây dựng công thức này thay vì dùng trực tiếp $w_p$?**
1. **Thành phần thứ nhất $\frac{w_p}{\max_q w_q}$ (Foreground Exploitation Term):** 
   Đóng vai trò là hàm mật độ nổi bật (Saliency Density). Việc chia cho $\max_q w_q$ thực hiện chuẩn hóa Min-Max cục bộ, đưa mọi giá trị $w_p$ về đoạn $[0, 1]$. Những patch chứa thân xe, đầu xe có độ sai khác màu sắc lớn so với mặt đường sẽ có $w_p \approx \max_q w_q$, khiến xác suất được chọn để che đạt cực đại. Khi vùng xe bị che, Student Network bị tước đi thông tin thị giác trực tiếp và bắt buộc phải học mối tương quan giữa ngữ cảnh xung quanh để suy diễn ra phương tiện.
2. **Thành phần thứ hai $\frac{1}{N_{\text{patches}}}$ (Background Exploration / Regularization Term):**
   Đây là phân phối đều tiên nghiệm (Uniform Prior). Nếu ta chọn $\alpha = 1.0$ (chỉ che theo $\Delta$), hệ thống sẽ gặp hai lỗi nghiêm trọng:
   * *Hiện tượng sụp đổ ngữ cảnh nền (Background Context Collapse):* Các patch mặt đường, làn xe không bao giờ bị che, khiến mô hình bỏ qua việc học mối quan hệ không gian giữa làn đường và xe cộ.
   * *Hiện tượng camera vắng xe (Degenerate Case):* Vào các khung giờ đêm vắng vẻ hoặc camera không có xe, $w_p \approx 0$ ở mọi patch. Phép chia sẽ mất ổn định hoặc không chọn đủ số lượng patch cần che theo tỷ lệ `mask_ratio`. Thành phần đều $\frac{1}{N_{\text{patches}}}$ đảm bảo trong mọi tình huống, mô hình luôn có một baseline ngẫu nhiên tối thiểu.
3. **Ý nghĩa của việc chọn $\alpha = 0.75$:**
   Đây là tỷ lệ vàng được kế thừa từ nguyên lý cân bằng Thăm dò - Khai thác (Exploration vs Exploitation) trong lý thuyết quyết định và Importance Sampling: 75% ngân sách che tập trung khai thác các vùng phương tiện (nơi chứa thông tin đặc trưng giao thông), và 25% ngân sách che được phân bổ ngẫu nhiên để duy trì khả năng biểu diễn tổng thể toàn khung hình.

### 2.4. Bản chất của Khái niệm "Global Views" và "Local Views" trong DINO

Khái niệm Multi-Crop là một trong những đóng góp cốt lõi của kiến trúc DINO (Caron et al., ICCV 2021). Trong hệ thống này:

#### 2.4.1. Global Views (Góc nhìn toàn cảnh)
* **Kích thước không gian:** $224 \times 224$ pixel.
* **Tỷ lệ diện tích cắt (Area Scale):** Cắt ngẫu nhiên từ $40\%$ đến $100\%$ diện tích khung hình gốc.
* **Số lượng:** Hệ thống sinh ra 2 Global Views ($V_1^g, V_2^g$).
* **Vai trò nghiệp vụ:** Chứa bố cục vĩ mô của khung cảnh giao thông (bao gồm toàn bộ mặt cắt tuyến đường, nhiều phương tiện cùng lúc, vỉa hè và góc phối cảnh camera). 
* **Phân phối mạng:**
  * **Teacher Network** nhận $V_1^g$ và $V_2^g$ ở trạng thái **nguyên bản không bị che (Unmasked)**. Do nhìn thấy toàn cảnh đầy đủ, Teacher đóng vai trò thiết lập "Mỏ neo ngữ nghĩa chuẩn" (Semantic Ground-Truth Representation).
  * **Student Network** nhận Global View bị áp dụng mặt nạ Foreground-Aware Masking (FAM). Student bị che mất các phần thân xe quan trọng và phải cố gắng tạo ra vector đặc trưng khớp với Teacher.

#### 2.4.2. Local Views (Góc nhìn chi tiết cục bộ)
* **Kích thước không gian:** Nhỏ hơn, cố định ở $96 \times 96$ pixel.
* **Tỷ lệ diện tích cắt (Area Scale):** Cắt trong phạm vi hẹp từ $5\%$ đến $40\%$ diện tích khung hình gốc.
* **Số lượng:** Mặc định sinh ra 4 Local Views ($V_1^l, V_2^l, V_3^l, V_4^l$).
* **Vai trò nghiệp vụ:** Phóng to vào các chi tiết cấu trúc vi mô của phương tiện (bánh xe, biển số, đèn pha, kính chắn gió).
* **Phân phối mạng:** **Chỉ duy nhất Student Network** được nhìn thấy các Local Views. Teacher hoàn toàn không nhận Local Views nhằm tiết kiệm bộ nhớ GPU và ngăn chặn hiện tượng trôi dạt ngữ nghĩa vĩ mô.

#### 2.4.3. Nguyên lý Chưng cất Cục bộ - Toàn thể (Local-to-Global Self-Distillation)
Hàm mất mát ép Student dù chỉ nhìn một mảnh cắt nhỏ $96 \times 96$ (ví dụ: chỉ nhìn thấy cụm đèn đuôi xe máy) vẫn phải dự đoán ra phân bố xác suất ngữ nghĩa của toàn bộ khung cảnh lớn $224 \times 224$ mà Teacher nhìn thấy:
$$\mathcal{L}_{\text{DINO}} = -\sum_{x \in \{V^g, V^l\}} \sum_{x' \in \{V_1^g, V_2^g\}, x' \neq x} P_t(x') \log P_s(x)$$
Cơ chế này ép Vision Transformer tự động trừu tượng hóa mối quan hệ giữa "bộ phận" và "tổng thể", tạo ra một bộ trích xuất đặc trưng có khả năng bất biến với góc nhìn và độ che khuất cực kỳ cao.

---

## 3. HƯỚNG 2: SELF-SUPERVISED SCENE DECOMPOSITION NETWORK (TRAFFIC-DECOMPOSE)

### 3.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction2_scene_decomposition/`
* Bộ nạp dữ liệu cặp ảnh đồng bộ: [`dataset.py`](file:///g:/nckh/DINO/direction2_scene_decomposition/dataset.py) (`DecompositionDataset`)
* Kiến trúc phân rã cảnh 3 nhánh: [`models.py`](file:///g:/nckh/DINO/direction2_scene_decomposition/models.py) (`TrafficDecompositionNet`)
* Hàm mất mát giám sát vật lý đa mục tiêu: [`losses.py`](file:///g:/nckh/DINO/direction2_scene_decomposition/losses.py) (`DecompositionLoss`)
* Huấn luyện mạng: [`train.py`](file:///g:/nckh/DINO/direction2_scene_decomposition/train.py)
* Suy luận Inpainting xóa xe tự động: [`infer.py`](file:///g:/nckh/DINO/direction2_scene_decomposition/infer.py)

### 3.2. Đặt vấn đề và Mục tiêu
Trong đồ họa máy tính và thị giác vật lý, một khung cảnh quan sát được mô hình hóa theo công thức hòa trộn Alpha (Alpha Compositing Formulation):
$$I_{\text{origin}} = M_{\alpha} \odot I_{\text{fg}} + (1 - M_{\alpha}) \odot I_{\text{bg}}$$
Trong đó:
* $I_{\text{bg}} \in \mathbb{R}^{H \times W \times 3}$: Lớp nền đường sạch bóng xe.
* $I_{\text{fg}} \in \mathbb{R}^{H \times W \times 3}$: Lớp chứa các thực thể phương tiện cô lập.
* $M_{\alpha} \in [0, 1]^{H \times W \times 1}$: Mặt nạ mờ trong suốt (Alpha Matte) biểu diễn mức độ hiện diện của phương tiện tại từng tọa độ không gian.
* $\odot$: Phép nhân Hadamard (nhân từng phần tử).

Thông thường, việc phân rã một bức ảnh duy nhất $I_{\text{origin}}$ thành cả 3 thành phần $\{I_{\text{bg}}, I_{\text{fg}}, M_{\alpha}\}$ là một bài toán nghịch đảo vô nghiệm xác định (Ill-posed Inverse Problem) vì số lượng ẩn số gấp 3 lần số lượng phương trình quan sát.

**Mục tiêu:** Xây dựng một mạng nơ-ron sâu tự giám sát hoàn toàn mang tên **TrafficDecompositionNet**. Bằng cách sử dụng ảnh nền thực tế $I_{\text{bg\_real}}$ làm mỏ neo giám sát vật lý, mạng học cách tự động bóc tách bất kỳ bức ảnh giao thông nào thành 3 lớp vật lý độc lập. Ứng dụng trực tiếp cho bài toán: **Tự động xóa sạch xe cộ trên đường (Inpainting) từ một frame duy nhất** mà không cần thuật toán vá ảnh truyền thống.

### 3.3. Thiết kế Kiến trúc và Hàm Mất Mát Đa Mục Tiêu

```text
                              ┌────────────────────┐
                              │  Ảnh Origin (x)    │
                              └─────────┬──────────┘
                                        │
                                        ▼
                         [Shared ViT Feature Backbone]
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
    [Background Head]           [Foreground Head]            [Alpha Head]
             │                          │                          │
             ▼                          ▼                          ▼
      I_bg (Nền sạch)            I_fg (Lớp xe)             M_α (Mặt nạ trong suốt)
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        │
                                        ▼
                   [Alpha Compositing: I_recon = α*fg + (1-α)*bg]
                                        │
               ┌────────────────────────┴────────────────────────┐
               ▼                                                 ▼
      L_recon(I_recon, x)                               L_bg(I_bg, I_bg_real)
      (Ép khớp ảnh gốc)                                 (Mỏ neo nền thật)
```

Hàm mục tiêu tối ưu hóa của toàn bộ mạng được thiết kế gồm 4 thành phần ràng buộc vật lý chặt chẽ:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{rec}} \mathcal{L}_{\text{recon}} + \lambda_{\text{bg}} \mathcal{L}_{\text{bg}} + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparsity}} + \lambda_{\text{tv}} \mathcal{L}_{\text{tv}}$$

* $\mathcal{L}_{\text{recon}} = \|I_{\text{origin}} - \hat{I}_{\text{origin}}\|_{1} + \big(1 - \text{SSIM}(I_{\text{origin}}, \hat{I}_{\text{origin}})\big)$
* $\mathcal{L}_{\text{bg}} = \|\hat{I}_{\text{bg}} - I_{\text{bg\_real}}\|_{1}$ (Mỏ neo nền thật)
* $\mathcal{L}_{\text{sparsity}} = \frac{1}{HW} \sum_{u, v} M_\alpha(u, v)$ (Ràng buộc thưa diện tích phương tiện)
* $\mathcal{L}_{\text{tv}} = \text{TotalVariation}(M_\alpha)$ (Làm mịn đường biên thân xe)

---

## 4. HƯỚNG 3: FOREGROUND-ENHANCED TRAFFIC COUNTING (STAGE 1 UPGRADE)

### 4.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction3_foreground_enhanced_counting/`
* Nạp nhãn đếm và tạo tensor 4 kênh: [`dataset.py`](file:///g:/nckh/DINO/direction3_foreground_enhanced_counting/dataset.py) (`FGCountingDataset`)
* Kiến trúc ViT mở rộng 4 kênh và Warm-start: [`models.py`](file:///g:/nckh/DINO/direction3_foreground_enhanced_counting/models.py) (`DINOv3FGCountingModel`, `adapt_patch_embed_to_4ch`, `RegressionHead`)
* Huấn luyện mô hình đếm xe: [`train.py`](file:///g:/nckh/DINO/direction3_foreground_enhanced_counting/train.py)
* Đánh giá hiệu năng và tự động xuất bảng $\text{\LaTeX}$: [`evaluate.py`](file:///g:/nckh/DINO/direction3_foreground_enhanced_counting/evaluate.py)

### 4.2. Đặt vấn đề và Phương pháp kỹ thuật
Bài toán ước lượng lưu lượng phương tiện (xe máy, ô tô và tổng lưu lượng) gặp thách thức lớn do hiện tượng che khuất nghiêm trọng (Heavy Occlusion) trong giờ cao điểm và điều kiện học ít mẫu (Few-shot learning: 5%, 10%, 20% nhãn).

**Giải pháp:** Mở rộng tầng Patch Embedding của ViT từ 3 kênh chuẩn (RGB) lên 4 kênh (RGB + $\Delta$), tiêm trực tiếp trường sai khác chuyển động vào tầng sâu của mạng:
$$X_{\text{4ch}} = [\text{R}, \text{G}, \text{B}, \Delta_{\text{norm}}] \in \mathbb{R}^{4 \times H \times W}$$

**Chiến lược khởi tạo thích ứng ấm (Warm-Start Weight Adaptation):**
$$W_{\text{4ch}}[:, 0:3, :, :] = W_{\text{pretrained}}, \quad W_{\text{4ch}}[:, 3, :, :] = \frac{1}{3} \sum_{c=0}^{2} W_{\text{pretrained}}[:, c, :, :]$$
Ba kênh đầu giữ nguyên 100% tri thức trích xuất biên cạnh đã tiền huấn luyện, kênh thứ 4 nhận trung bình cộng cường độ 3 kênh RGB, đảm bảo tại epoch 0 kênh $\Delta$ đóng góp năng lượng đồng mức mà không gây sốc gradient.

Đánh giá chuẩn qua phân chia phân tách không gian (Spatial Disjoint Splitting theo Camera ID) để triệt tiêu hoàn toàn hiện tượng rò rỉ dữ liệu (Data Leakage).

---

## 5. HƯỚNG 4: UNSUPERVISED TRAFFIC ANOMALY DETECTION VIA $\Delta$-CONDITIONED DINO EMBEDDINGS

### 5.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction4_anomaly_detection/`
* Bộ nhớ đặc trưng chuẩn: [`memory_bank.py`](file:///g:/nckh/DINO/direction4_anomaly_detection/memory_bank.py) (`AnomalyMemoryBank`)
* Trích xuất đặc trưng kết hợp: [`feature_extractor.py`](file:///g:/nckh/DINO/direction4_anomaly_detection/feature_extractor.py) (`DeltaConditionedExtractor`)
* Thuật toán phát hiện bất thường: [`detector.py`](file:///g:/nckh/DINO/direction4_anomaly_detection/detector.py) (`TrafficAnomalyDetector`)
* Pipeline suy luận và trực quan hóa: [`run_detection.py`](file:///g:/nckh/DINO/direction4_anomaly_detection/run_detection.py)

### 5.2. Đặt vấn đề và Mục tiêu
Các sự kiện bất thường trong giao thông đô thị (tai nạn, xe chết máy dừng giữa đường, phương tiện đi ngược chiều, vật cản rơi vãi) mang bản chất cực kỳ hiếm gặp và phân phối lệch (long-tail), không thể thu thập đủ dữ liệu gán nhãn để huấn luyện các bộ phân loại có giám sát.

### 5.3. Giả thuyết Khoa học và Mô hình Hóa
* **Trạng thái bình thường:** Trường sai khác $\Delta$ tuân theo một phân phối biến thiên ổn định đặc trưng cho luồng di chuyển đều đặn của dòng xe.
* **Trạng thái bất thường:** Khi xảy ra sự cố, $\Delta$ lệch hẳn khỏi phân phối bình thường cả về cường độ (vùng sai khác lớn tồn tại tĩnh qua nhiều khung hình) lẫn cấu trúc không gian (nằm ở vị trí bất thường).

Xây dựng bộ mô tả kép kết hợp tín hiệu vật lý quang học và ngữ nghĩa sâu:
$$\mathbf{f}_{\text{anomaly}} = \text{Concat}\Big(\text{Stats}(\Delta), \ \text{DINO}_{[CLS]}\Big) \in \mathbb{R}^{D + 3}$$
trong đó $\text{Stats}(\Delta) = [\text{mean}(\Delta), \text{std}(\Delta), \text{ratio}(\Delta > \tau)]$.

Điểm bất thường (Anomaly Score) được tính dựa trên khoảng cách $k$-NN đối với ngân hàng nhớ (Memory Bank) lưu trữ các vector chuẩn trạng thái bình thường:
$$\text{Score}(x) = \frac{1}{k} \sum_{j=1}^k d_{\text{Euclidean}}(\mathbf{f}_x, \text{NN}_j(\text{MemoryBank}))$$
Memory Bank được cập nhật liên tục qua trung bình trượt hàm mũ (EMA) để thích ứng với biến thiên quang học theo thời gian.

---

## 6. HƯỚNG 5: TEMPORAL CONTRASTIVE LEARNING FOR TRAFFIC DENSITY ESTIMATION

### 6.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction5_temporal_density/`
* Nạp chuỗi thời gian: [`dataset.py`](file:///g:/nckh/DINO/direction5_temporal_density/dataset.py) (`TemporalTrafficDataset`)
* Kiến trúc mã hóa thời gian: [`models.py`](file:///g:/nckh/DINO/direction5_temporal_density/models.py) (`TemporalTrafficEncoder`, `DeltaSpatialEncoder`)
* Hàm mất mát tương phản thời gian: [`losses.py`](file:///g:/nckh/DINO/direction5_temporal_density/losses.py) (`TemporalContrastiveLoss`)
* Quy trình huấn luyện: [`train.py`](file:///g:/nckh/DINO/direction5_temporal_density/train.py)

### 6.2. Đặt vấn đề và Động lực Khoa học
Các phương pháp thị giác hiện tại chủ yếu xử lý từng khung hình độc lập (Static Frame-by-Frame). Tuy nhiên, camera CCTV cung cấp luồng quan sát chuỗi thời gian liên tục $\{I^{t_1}, I^{t_2}, \dots, I^{t_T}\}$. Vận tốc trung bình và xu hướng ùn tắc của dòng xe được phản ánh trực tiếp qua đạo hàm thời gian của trường sai khác $\frac{\partial \Delta}{\partial t}$.

### 6.3. Kiến trúc Mạng và Hàm Mất Mát InfoNCE Thời Gian
Mô hình tổ chức khung hình theo các cửa sổ thời gian kích thước $K$ (mặc định $K=4$). Hai cửa sổ liền kề nhau $W_a = [I^{t_1}, \dots, I^{t_K}]$ và $W_b^+ = [I^{t_{K+1}}, \dots, I^{t_{2K}}]$ từ cùng một camera tạo thành cặp dương tự nhiên (Positive Pair).

Vector đặc trưng tổng hợp thời gian $\mathbf{h} \in \mathbb{R}^{256}$ được hình thành qua:
$$\mathbf{h} = \text{TemporalTransformer}\Big(\big[\text{DINO}(I^t) \oplus \text{CNN}(\Delta^t) + \text{PE}(t)\big]_{t=1}^K\Big)$$

Mô hình tối ưu hóa hàm mất mát tương phản thời gian đối xứng (Symmetric Temporal InfoNCE):
$$\mathcal{L}_{\text{temporal}} = -\frac{1}{2} \left[ \log \frac{\exp(\mathbf{z}_a \cdot \mathbf{z}_b^+ / \tau)}{\sum_j \exp(\mathbf{z}_a \cdot \mathbf{z}_j^- / \tau)} + \log \frac{\exp(\mathbf{z}_b^+ \cdot \mathbf{z}_a / \tau)}{\sum_j \exp(\mathbf{z}_b^+ \cdot \mathbf{z}_j^- / \tau)} \right]$$
Vector $\mathbf{h}$ được kết nối trực tiếp với Regression Head để ước lượng mật độ phương tiện và chỉ số mức độ phục vụ (Level of Service - LoS).

---

## 7. HƯỚNG 6: $\Delta$-GUIDED UNSUPERVISED VEHICLE RE-IDENTIFICATION ACROSS CAMERAS

### 7.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction6_vehicle_reid/`
* Trích xuất vùng xe không cần detector: [`roi_extractor.py`](file:///g:/nckh/DINO/direction6_vehicle_reid/roi_extractor.py) (`DeltaRoIExtractor`)
* Kiến trúc Re-ID chuẩn BNNeck: [`models.py`](file:///g:/nckh/DINO/direction6_vehicle_reid/models.py) (`VehicleReIDModel`)
* Hàm mất mát tương phản tracklet: [`losses.py`](file:///g:/nckh/DINO/direction6_vehicle_reid/losses.py) (`TrackletContrastiveLoss`)
* Công cụ so khớp và đánh giá CMC/mAP: [`matcher.py`](file:///g:/nckh/DINO/direction6_vehicle_reid/matcher.py) (`VehicleReIDMatcher`)
* Pipeline thực thi: [`run_reid.py`](file:///g:/nckh/DINO/direction6_vehicle_reid/run_reid.py)

### 7.2. Đặt vấn đề và Giải pháp Đột phá
Bài toán nhận dạng lại phương tiện qua mạng lưới camera đô thị (Vehicle Re-ID) thường vấp phải hai rào cản chi phí:
1. Phụ thuộc vào Object Detector nặng nề (YOLO/Faster R-CNN) để cắt hộp bao xe.
2. Chi phí gán nhãn danh tính xe (Identity ID) thủ công qua hàng trăm camera là bất khả thi.

**Giải pháp:**
* **$\Delta$-Guided RoI Extraction:** Tận dụng trực tiếp trường sai khác $\Delta$ qua phân tích thành phần liên thông (Connected Components) kết hợp bộ lọc diện tích và tỷ lệ khung hình để tự động cắt các phương tiện chuyển động mà **hoàn toàn không cần Object Detector**.
* **DINO ViT + BNNeck Projector:** Tạo vector nhúng danh tính 256 chiều chuẩn hóa $L_2$ có khả năng bất biến với góc nghiêng camera và ánh sáng.
* **Unsupervised Tracklet Contrastive:** Tận dụng chuỗi frame liên tiếp của cùng camera làm cặp dương tự thân, huấn luyện mô hình phân biệt danh tính xe không cần giám sát.
* **Đánh giá chuẩn quốc tế:** Đánh giá Cumulative Matching Characteristics (CMC Rank-1, Rank-5) và mean Average Precision (mAP) trên các truy vấn liên camera (Cross-Camera Query-Gallery matching).

---

## 8. HƯỚNG 7: OPEN-VOCABULARY TRAFFIC SCENE UNDERSTANDING VIA $\Delta$-CONDITIONED PROPOSALS

### 8.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction7_open_vocabulary/`
* Động cơ sinh đề xuất vùng vật lý: [`proposal_engine.py`](file:///g:/nckh/DINO/direction7_open_vocabulary/proposal_engine.py) (`DeltaProposalEngine`)
* Quản lý từ vựng văn bản mở: [`text_prompts.py`](file:///g:/nckh/DINO/direction7_open_vocabulary/text_prompts.py) (`TrafficPromptVocabulary`)
* Mô hình căn chỉnh DINO sang CLIP: [`models.py`](file:///g:/nckh/DINO/direction7_open_vocabulary/models.py) (`OpenVocabTrafficDetector`)
* Pipeline nhận diện trực quan: [`pipeline.py`](file:///g:/nckh/DINO/direction7_open_vocabulary/pipeline.py)

### 8.2. Đặt vấn đề và Phương pháp kỹ thuật
Các mô hình nhận diện khép kín (Closed-Set Detectors) chỉ phát hiện được các lớp có sẵn trong tập huấn luyện (ví dụ: xe hơi, xe máy). Trên đường phố thực tế tại Việt Nam, sự xuất hiện của các phương tiện đặc thù (xe cứu thương, xe rác, xe ba gác) hoặc chướng ngại vật bất thường đòi hỏi năng lực hiểu cảnh từ vựng mở (Open-Vocabulary Perception).

**Cơ chế hoạt động:**
1. **Delta Region Proposals:** $\Delta$-Mask được lọc qua phân ngưỡng đa mức và Non-Maximum Suppression (NMS) để sinh ra các hộp đề xuất vùng đối tượng không phụ thuộc lớp (Class-Agnostic Proposals) thay thế RPN.
2. **Vision-Language Alignment:** Bộ chiếu đa tầng ánh xạ vector đặc trưng vùng của DINO ViT sang không gian nhúng ngữ nghĩa của CLIP (512 chiều).
3. **Zero-Shot Classification:** Xác suất phân loại của mỗi vùng $r$ đối với danh mục văn bản $c$ được tính qua độ tương đồng Cosine:
   $$P(c | r) = \frac{\exp\big(\text{sim}(\mathbf{z}_r, \mathbf{w}_c) / \tau\big)}{\sum_{k} \exp\big(\text{sim}(\mathbf{z}_r, \mathbf{w}_k) / \tau\big)}$$
   Hỗ trợ người vận hành truy vấn bất kỳ đối tượng nào bằng mô tả ngôn ngữ tự nhiên tiếng Việt hoặc tiếng Anh.

---

## 9. HƯỚNG 8: SELF-SUPERVISED ROAD SURFACE CONDITION ESTIMATION

### 9.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction8_road_condition/`
* Nạp chuỗi ảnh nền 24h: [`dataset.py`](file:///g:/nckh/DINO/direction8_road_condition/dataset.py) (`RoadSurfaceDataset`)
* Mô hình phân tích đa thuộc tính mặt đường: [`models.py`](file:///g:/nckh/DINO/direction8_road_condition/models.py) (`RoadConditionClassifier`)
* Hàm mất mát mỏ neo vật lý: [`losses.py`](file:///g:/nckh/DINO/direction8_road_condition/losses.py) (`SurfaceConsistencyLoss`)
* Đánh giá và gom cụm PCA toàn đô thị: [`eval.py`](file:///g:/nckh/DINO/direction8_road_condition/eval.py)

### 9.2. Đặt vấn đề và Ứng dụng Quản lý Đô thị
Chuỗi ảnh nền 24 giờ $\{I_{\text{bg}}^{h=0}, \dots, I_{\text{bg}}^{h=23}\}$ trên 608 camera chứa đựng thông tin phong phú về môi trường và hạ tầng mặt đường đô thị:
* Trạng thái thời tiết: Mặt đường khô ráo vs. mặt đường ẩm ướt, đọng nước (thể hiện qua các vùng phản chiếu gương Specular Reflection).
* Chu kỳ chiếu sáng: Ngày, chạng vạng hoàng hôn và hệ thống đèn cao áp ban đêm.
* Chất lượng hạ tầng: Độ suy giảm kết cấu, nứt nẻ, gồ ghề của mặt đường nhựa.

**Phương pháp:**
* DINO ViT $[CLS]$ token trích xuất vector biểu diễn kết cấu mặt đường $\mathbf{z}_{\text{road}} \in \mathbb{R}^D$.
* Ba đầu dự đoán đa nhiệm (Multi-Task Heads):
  1. `pred_wetness`: Chỉ số ẩm ướt / đọng nước $\in [0, 1]$ (được neo vật lý bằng tỷ lệ phản chiếu gương quang học).
  2. `logits_illum`: Phân loại 3 mức chiếu sáng (Đêm, Chạng vạng, Ngày).
  3. `pred_degradation`: Chỉ số hư hại kết cấu mặt đường $\in [0, 1]$ (được neo bằng độ biến thiên gradient Sobel).
* Pipeline tự động gom cụm PCA 2D và xuất báo cáo tình trạng toàn đô thị `road_surface_citywide_report.csv` phục vụ quy hoạch và bảo trì mặt đường thông minh.

---

## 10. TỔNG HỢP VÀ HỆ SINH THÁI 8 HƯỚNG NGHIÊN CỨU

Tám hướng nghiên cứu trên tạo thành một hệ sinh thái khoa học khép kín và có tính kế thừa chặt chẽ, tối ưu hóa triệt để cặp tín hiệu vật lý quang học từ camera giao thông:

```text
                           [Dữ Liệu Thô: 608 Camera TP.HCM]
                                          │
                                          ▼
                      [Cặp Ảnh Vật Lý: Origin + Background]
                                          │
           ┌──────────────────────────────┼──────────────────────────────┐
           ▼                              ▼                              ▼
  [HƯỚNG 1: BG-Guided DINO]      [HƯỚNG 3: FG Counting]       [HƯỚNG 8: Road Surface]
  • FAM: Ép ViT học xe cộ        • Mở rộng 4 kênh (RGB+Δ)     • Khai thác chuỗi I_bg 24h
  • Tạo ViT Backbone chuyên biệt • Warm-Start Initialization  • Đánh giá ngập ướt, hạ tầng
           │                              │                              │
           ├──────────────────────────────┼──────────────────────────────┘
           ▼                              ▼
  [HƯỚNG 2: Scene Decomposition] [HƯỚNG 4: Anomaly Detection]
  • Tách 3 lớp: Nền + Xe + Alpha • UAD với Δ + DINO Memory Bank
  • Tự động xóa xe (Inpainting)  • Phát hiện tai nạn, xe dừng đỗ
           │                              │
           └──────────────────────────────┼──────────────────────────────┐
                                          ▼                              ▼
                             [HƯỚNG 5: Temporal Density]   [HƯỚNG 6 & 7: Re-ID & Open-Vocab]
                             • Contrastive trên chuỗi Δ-Maps• H6: Re-ID liên camera qua Δ-RoI
                             • Đo lưu lượng, tốc độ dòng xe• H7: Hiểu cảnh từ vựng mở (CLIP)
```

### Bảng Tổng Hợp So Sánh 8 Hướng Nghiên Cứu

| Hướng | Tên Nghiên Cứu | Thư Mục Mã Nguồn | Cơ Chế Cốt Lõi | Độ Mới (Novelty) | Venue Đề Xuất |
|:---|:---|:---|:---|:---:|:---|
| **H1** | **BG-Guided DINO Continual SSL** | `direction1_bg_guided_dino/` | Foreground-Aware Masking (FAM) ép ViT tập trung học biểu diễn xe cộ | 4/5 | IEEE T-ITS, EAAI |
| **H2** | **Scene Decomposition Network** | `direction2_scene_decomposition/` | Alpha Compositing tự giám sát với mỏ neo nền thật (Road Inpainting) | 5/5 | CVPR, ECCV, NeurIPS |
| **H3** | **Foreground-Enhanced Counting** | `direction3_foreground_enhanced_counting/` | Mở rộng Patch Embedding 4 kênh (RGB+$\Delta$) kết hợp Warm-Start | 3/5 | EAAI Journal, ITSC |
| **H4** | **Unsupervised Anomaly Detection** | `direction4_anomaly_detection/` | Biểu diễn kép $\Delta$+DINO, Memory Bank EMA, phát hiện sự cố không cần nhãn | 4/5 | IEEE T-ITS, WACV |
| **H5** | **Temporal Contrastive Density** | `direction5_temporal_density/` | Học tương phản thời gian trên chuỗi $\Delta$-Maps để ước lượng mật độ và tốc độ | 4.5/5 | CVPR, IEEE T-ITS |
| **H6** | **Delta-Guided Vehicle Re-ID** | `direction6_vehicle_reid/` | Trích xuất RoI từ $\Delta$, BNNeck DINO, tracklet contrastive, xếp hạng CMC/mAP | 4/5 | IEEE T-ITS, ICPR |
| **H7** | **Open-Vocabulary Scene Understanding** | `direction7_open_vocabulary/` | Delta proposals không phụ thuộc lớp kết hợp căn chỉnh DINO sang CLIP text | 4/5 | ECCV, WACV |
| **H8** | **Road Surface Condition Estimation** | `direction8_road_condition/` | Đánh giá đa thuộc tính (đọng nước, chiếu sáng, hư hại) từ chuỗi ảnh nền 24h | 3.5/5 | IEEE T-ITS, TRB |
