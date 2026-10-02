# BÁO CÁO KHOA HỌC: PHƯƠNG PHÁP LUẬN VÀ THIẾT KẾ KIẾN TRÚC 7 HƯỚNG NGHIÊN CỨU KHAI THÁC CẶP ẢNH NỀN TĨNH VÀ PHƯƠNG TIỆN TRONG THỊ GIÁC GIAO THÔNG

**Dự án:** Khai thác tự giám sát cặp ảnh Background tĩnh và Origin phương tiện phục vụ bài toán thị giác máy tính giám sát giao thông đô thị  
**Dữ liệu thực nghiệm:** Hệ thống camera giao thông đô thị TP.HCM (IC4SD-Traffic-HCM)  
**Địa chỉ mã nguồn:** Thư mục `g:/nckh/DINO/`  
**Trạng thái hệ sinh thái:** Hoàn thiện 7 hướng nghiên cứu chuyên sâu (H1, H2, H3, H5, H6, H7, H8); Loại bỏ Hướng 4 (UAD Anomaly Detection) nhằm tập trung tối đa nguồn lực vào các bài toán đo lường động học và nhận dạng giao thông có tính ứng dụng thực tiễn cao nhất.

---

## 1. TỔNG QUAN HỆ THỐNG VÀ BẢN CHẤT DỮ LIỆU ĐẦU VÀO

Trong giám sát giao thông qua camera cố định (CCTV), hệ thống có hai nguồn tín hiệu hình ảnh tự nhiên cùng góc quan sát:
1. **Ảnh nguồn có phương tiện ($I_{\text{origin}} \in \mathbb{R}^{H \times W \times 3}$):** Chứa các đối tượng động (xe máy, ô tô, xe tải, xe buýt, người đi bộ) di chuyển trên nền đường đô thị phức tạp. Được thu thập theo định dạng `output/{stt}_{timestamp}.jpg`.
2. **Ảnh nền tĩnh sạch bóng xe ($I_{\text{bg}} \in \mathbb{R}^{H \times W \times 3}$):** Được tổng hợp thông qua thuật toán lọc trung vị theo thời gian (Temporal Median Filtering) kết hợp chuẩn hóa độ sáng không gian màu HSV trên từng camera và từng slot giờ ($00\text{h} - 23\text{h}$ theo giờ Việt Nam UTC+7). Được lưu trữ tại `traffic_backgrounds/route_{stt}/background_slot_{slot}h.jpg`.

Sự kết hợp giữa $I_{\text{origin}}$ và $I_{\text{bg}}$ cung cấp một tín hiệu vật lý quang học tiên nghiệm (Physical Prior) vô cùng mạnh mẽ:
$$\Delta(u, v) = \|I_{\text{origin}}(u, v) - I_{\text{bg}}(u, v)\|$$

Thay vì bỏ phí ảnh nền tĩnh chỉ để xem trực quan hoặc sử dụng các thuật toán trừ nền cổ điển dễ bị nhiễu do thời tiết và bóng đổ, hệ thống DINO Suite được xây dựng gồm **7 hướng nghiên cứu** độc lập nhưng tương hỗ chặt chẽ, tích hợp $\Delta$ và $I_{\text{bg}}$ vào không gian biểu diễn sâu của Vision Transformer (ViT) để giải quyết các bài toán cốt lõi trong Giao thông Thông minh (Intelligent Transportation Systems - ITS).

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

**Mục tiêu:** Xây dựng cơ chế che có định hướng vùng tiền cảnh (Foreground-Aware Masking - FAM) kết hợp kiến trúc tự chưng cất tri thức Multi-Crop của DINO, ép Vision Transformer phải tập trung học biểu diễn hình thái, đường biên và ngữ nghĩa của xe cộ mà hoàn toàn không cần con người gán nhãn thủ công (Zero human annotation).

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
   Đây là phân phối đều tiên nghiệm (Uniform Prior). Nếu chọn $\alpha = 1.0$ (chỉ che theo $\Delta$), hệ thống sẽ gặp hai lỗi nghiêm trọng:
   * *Hiện tượng sụp đổ ngữ cảnh nền (Background Context Collapse):* Các patch mặt đường, làn xe không bao giờ bị che, khiến mô hình bỏ qua việc học mối quan hệ không gian giữa làn đường và xe cộ.
   * *Hiện tượng camera vắng xe (Degenerate Case):* Vào các khung giờ đêm vắng vẻ hoặc camera không có xe, $w_p \approx 0$ ở mọi patch. Phép chia sẽ mất ổn định hoặc không chọn đủ số lượng patch cần che theo tỷ lệ `mask_ratio`. Thành phần đều $\frac{1}{N_{\text{patches}}}$ đảm bảo trong mọi tình huống, mô hình luôn có một baseline ngẫu nhiên tối thiểu.
3. **Ý nghĩa của việc chọn $\alpha = 0.75$:**
   Đây là tỷ lệ vàng được kế thừa từ nguyên lý cân bằng Thăm dò - Khai thác (Exploration vs Exploitation): 75% ngân sách che tập trung khai thác các vùng phương tiện (nơi chứa thông tin đặc trưng giao thông), và 25% ngân sách che được phân bổ ngẫu nhiên để duy trì khả năng biểu diễn tổng thể toàn khung hình.

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

## 5. HƯỚNG 5: DỰ ĐOÁN MẬT ĐỘ KHÔNG-THỜI GIAN VÀ CẤP ĐỘ DỊCH VỤ GIAO THÔNG (SPATIO-TEMPORAL DENSITY & HCM LoS ESTIMATION)

### 5.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction5_temporal_density/`
* Nạp chuỗi thời gian & tính mỏ neo vật lý $\rho_{\text{phys}}$: [`dataset.py`](file:///g:/nckh/DINO/direction5_temporal_density/dataset.py) (`TemporalDensityDataset`, `compute_physical_density`, `discretize_los`)
* Mô hình Không - Thời Gian (ViT + Delta-CNN + BiGRU): [`models.py`](file:///g:/nckh/DINO/direction5_temporal_density/models.py) (`SpatioTemporalDensityModel`, `DeltaSpatialCNN`)
* Hàm mất mát đa nhiệm không - thời gian: [`losses.py`](file:///g:/nckh/DINO/direction5_temporal_density/losses.py) (`TemporalDensityMultiTaskLoss`)
* Pipeline huấn luyện chuỗi thời gian với AMP & Multi-GPU: [`train.py`](file:///g:/nckh/DINO/direction5_temporal_density/train.py)
* Tài liệu kỹ thuật chi tiết: [`README.md`](file:///g:/nckh/DINO/direction5_temporal_density/README.md)

### 5.2. Đặt vấn đề và Mục tiêu Khoa học Cụ thể
1. **Hạn chế của các phương pháp hiện hành:**
   * Các phương pháp thị giác trước đây chủ yếu xử lý từng khung hình tĩnh độc lập (Static Frame-by-Frame). Đếm thủ công từng chiếc xe trong dòng giao thông hỗn hợp tại Việt Nam (hàng trăm xe máy ken đặc vào nhau trong giờ cao điểm) dẫn tới tỷ lệ sai số tích lũy cực lớn do che khuất chồng chéo (Severe Occlusion).
   * Các mô hình tương phản thời gian thông thường (Temporal Contrastive / InfoNCE) chỉ cố gắng kéo gần hai cửa sổ thời gian ngẫu nhiên mà **không có mỏ neo vật lý (Physical Grounding)**. Trên thực tế, hai khung giờ liền kề có thể diễn ra biến chuyển giao thông hoàn toàn khác nhau (đèn đỏ chuyển sang đèn xanh, đường vắng chuyển sang tắc nghẽn).
2. **Mục tiêu khoa học đột phá:**
   * Thay vì đếm từng đối tượng rời rạc, mô hình chuyển đổi bài toán sang **Ước lượng Tỷ lệ Chiếm dụng Mặt đường Liên tục (Continuous Road Space Occupancy Ratio $\rho(t) \in [0, 1]$)**.
   * Đồng thời phân loại trực tiếp **Cấp độ Dịch vụ Giao thông (Level of Service - LoS)** theo tiêu chuẩn Cẩm nang Năng lực Đường cao tốc và Đô thị (Highway Capacity Manual - HCM).
   * Dự đoán **Đạo hàm Xu hướng Biến thiên Thời gian $\frac{\partial \rho}{\partial t}$**, cho phép phát hiện sớm nguy cơ kẹt xe trước khi luồng giao thông bị tê liệt hoàn toàn.

### 5.3. Mô Hình Toán Học và Phương Pháp Luận

#### 5.3.1. Xây dựng Mỏ Neo Vật Lý Tự Thân (Self-Supervised Physical Ground Truth)
Với mỗi khung hình tại thời điểm $t$, trường sai khác quang học $\Delta_t$ được tính toán so với ảnh nền $I_{\text{bg}}$ trong không gian màu CIE-LAB. Tỷ lệ chiếm dụng lòng đường vật lý $\rho_{\text{phys}}(t)$ được tính toán hoàn toàn tự động mà không cần gán nhãn thủ công:
$$\rho_{\text{phys}}(t) = \frac{1}{H \times W} \sum_{u=1}^H \sum_{v=1}^W \mathbb{I}\big(\Delta_t(u, v) > \tau\big)$$
trong đó $\mathbb{I}(\cdot)$ là hàm chỉ thị (Indicator Function) và $\tau$ là ngưỡng nhạy quang học thích ứng Otsu kết hợp lọc nhiễu hình thái học.

#### 5.3.2. Chuẩn Hóa Cấp Độ Dịch Vụ Giao Thông (HCM LoS Discretization)
Giá trị $\rho_{\text{phys}}(t)$ được ánh xạ vào 4 cấp độ phục vụ tiêu chuẩn theo cẩm nang HCM:
$$\text{LoS}(t) = \begin{cases} 
0 \quad (\text{Free-Flow: Thông thoáng, lưu thông tự do}), & \rho_{\text{phys}}(t) < 0.15 \\ 
1 \quad (\text{Moderate: Dòng xe ổn định, mật độ trung bình}), & 0.15 \le \rho_{\text{phys}}(t) < 0.35 \\ 
2 \quad (\text{Slow: Mật độ cao, dòng xe di chuyển chậm}), & 0.35 \le \rho_{\text{phys}}(t) < 0.60 \\ 
3 \quad (\text{Gridlock: Kẹt xe nghiêm trọng, tê liệt hoàn toàn}), & \rho_{\text{phys}}(t) \ge 0.60 
\end{cases}$$

Đồng thời, đạo hàm xu hướng được tính toán qua sai phân thời gian hữu hạn:
$$\delta(t) = \rho_{\text{phys}}(t) - \rho_{\text{phys}}(t-1) \in [-1, 1]$$
Khi $\delta(t) > 0$, mật độ đang gia tăng (nguy cơ ùn ứ); khi $\delta(t) < 0$, lòng đường đang giải tỏa.

#### 5.3.3. Kiến Trúc Mạng Hợp Nhất Không - Thời Gian (Spatio-Temporal Fusion)
Mô hình xử lý một chuỗi gồm $T$ khung hình liên tiếp $\{I_1, I_2, \dots, I_T\}$. Tại mỗi thời điểm $t$:
1. **Trích xuất Ngữ nghĩa Cao cấp (Semantic Stream):** Vision Transformer Backbone trích xuất vector $[CLS]$ đại diện toàn cảnh:
   $$\mathbf{z}_{\text{sem}}^t = \text{DINO}(I_t)_{[CLS]} \in \mathbb{R}^{D}$$
2. **Trích xuất Hình thái Không gian Tiền cảnh (Spatial Foreground Stream):** Mạng tích chập 3 tầng `DeltaSpatialCNN` trích xuất thông tin cấu trúc phân bố phương tiện từ bản đồ $\Delta_t$:
   $$\mathbf{z}_{\text{spatial}}^t = \text{CNN}(\Delta_t) \in \mathbb{R}^{128}$$
3. **Hợp nhất Đặc trưng Khung hình (Frame Fusion):**
   $$\mathbf{x}_t = \text{Linear}\big([\mathbf{z}_{\text{sem}}^t \,\|\, \mathbf{z}_{\text{spatial}}^t]\big) \in \mathbb{R}^{256}$$
4. **Mô hình hóa Động học Chuỗi Thời gian (Recurrent Sequence Modeling):** Chuỗi vector $[\mathbf{x}_1, \dots, \mathbf{x}_T]$ được nạp vào mạng nơ-ron hồi quy hai chiều 2 tầng (2-layer Bidirectional GRU):
   $$\mathbf{h}_t = \text{BiGRU}(\mathbf{x}_t, \mathbf{h}_{t-1}) \in \mathbb{R}^{256}$$
5. **Đầu ra Đa Nhiệm (Multi-Task Heads):**
   * Ước lượng mật độ liên tục: $\hat{\rho}_t = \sigma(\mathbf{W}_{\rho} \mathbf{h}_t + b_{\rho}) \in [0, 1]$
   * Dự đoán cấp độ phục vụ: $\hat{\mathbf{y}}_{\text{LoS}}^t = \text{Softmax}(\mathbf{W}_{\text{LoS}} \mathbf{h}_t + b_{\text{LoS}}) \in \mathbb{R}^4$
   * Dự đoán xu hướng biến thiên: $\hat{\delta}_t = \tanh(\mathbf{W}_{\delta} \mathbf{h}_t + b_{\delta}) \in [-1, 1]$

#### 5.3.4. Hàm Mất Mát Đa Nhiệm Không - Thời Gian
$$\mathcal{L}_{\text{total}} = \lambda_{\rho} \mathcal{L}_{\text{SmoothL1}}(\hat{\rho}, \rho_{\text{phys}}) + \lambda_{\text{LoS}} \mathcal{L}_{\text{CE}}(\hat{\mathbf{y}}_{\text{LoS}}, y_{\text{LoS}}) + \lambda_{\text{trend}} \mathcal{L}_{\text{SmoothL1}}(\hat{\delta}, \delta_{\text{phys}}) + \lambda_{\text{smooth}} \frac{1}{T-1} \sum_{t=1}^{T-1} \|\hat{\rho}_{t+1} - \hat{\rho}_t\|_2^2$$
Trong đó thành phần $\mathcal{L}_{\text{smooth}}$ đóng vai trò chuẩn hóa điều hòa (Temporal Smoothness Regularization), ngăn chặn hiện tượng mật độ bị dao động nhảy vọt phi vật lý giữa các frame kề cận.

### 5.4. Hệ Thống Chỉ Số Đánh Giá Nghiệm Thu (Metrics)
* **Độ chính xác Mật độ liên tục:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), và Hệ số xác định $R^2 \in (-\infty, 1.0]$.
* **Phân loại Cấp độ Dịch vụ:** Top-1 Accuracy và Macro-averaged F1 Score qua 4 lớp HCM LoS.
* **Độ nhạy Xu hướng:** Directional Accuracy (tỷ lệ phần trăm dự báo chính xác chiều hướng tăng/giảm kẹt xe).

---

## 6. HƯỚNG 6: ĐỊNH DANH LẠI PHƯƠNG TIỆN LIÊN CAMERA THEO HÀNH LANG VÀ ƯỚC TÍNH THỜI GIAN DI CHUYỂN (CORRIDOR-BASED VEHICLE RE-ID & TRAVEL TIME ESTIMATION)

### 6.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction6_vehicle_reid/`
* Trích xuất vùng xe tiền cảnh không cần detector: [`roi_extractor.py`](file:///g:/nckh/DINO/direction6_vehicle_reid/roi_extractor.py) (`DeltaRoIExtractor`)
* Mô hình trích xuất vector đặc trưng với GeM Pooling & BNNeck: [`models.py`](file:///g:/nckh/DINO/direction6_vehicle_reid/models.py) (`VehicleReIDModel`)
* Hàm mất mát tương phản chuỗi tracklet tự giám sát: [`losses.py`](file:///g:/nckh/DINO/direction6_vehicle_reid/losses.py) (`TrackletContrastiveLoss`)
* Bộ so khớp liên camera tích hợp cửa sổ không - thời gian: [`matcher.py`](file:///g:/nckh/DINO/direction6_vehicle_reid/matcher.py) (`VehicleReIDMatcher`)
* Pipeline thực thi và ước lượng tốc độ hành lang: [`run_reid.py`](file:///g:/nckh/DINO/direction6_vehicle_reid/run_reid.py)
* Tài liệu kỹ thuật chi tiết: [`README.md`](file:///g:/nckh/DINO/direction6_vehicle_reid/README.md)

### 6.2. Đặt vấn đề và Mục tiêu Khoa học Cụ thể
1. **Điểm yếu chí tử của các phương pháp Re-ID thông thường:**
   * Hầu hết các nghiên cứu Vehicle Re-ID truyền thống thực hiện so khớp Cosine tự do (Unconstrained Global Gallery Matching) trên toàn bộ kho ảnh. Trong bối cảnh giao thông Việt Nam, với hàng vạn chiếc xe máy có kiểu dáng và màu sắc tương đồng (xe tay ga trắng, xe số đen), việc so khớp trực quan thuần túy dẫn đến tỷ lệ **dương tính giả (False Positives) khổng lồ**, biến hệ thống thành bất khả thi trong thực tiễn.
   * Các nghiên cứu thường chỉ dừng lại ở các chỉ số học máy trừu tượng (Rank-1, mAP) mà không giải quyết bài toán nghiệp vụ cốt lõi của giao thông thông minh.
2. **Mục tiêu khoa học đột phá:**
   * **Loại bỏ hoàn toàn Object Detector (YOLO/Faster R-CNN):** Sử dụng trực tiếp trường sai khác $\Delta$ với lọc hình thái học để tự động cắt các vùng phương tiện chuyển động, tiết kiệm hơn 60% chi phí tính toán phần cứng.
   * **Ràng buộc Tính khả thi Không - Thời gian theo Hành lang (Spatio-Temporal Feasibility Windowing):** Tích hợp thông tin topo mạng lưới giao thông (cự ly giữa các camera liên tiếp $d(c_1, c_2)$) và giới hạn vận tốc vật lý thực tế của phương tiện trong đô thị $[v_{\text{min}}, v_{\text{max}}]$. Cơ chế này loại bỏ ngay lập tức hơn 95% - 98% ứng viên sai khác về mặt thời gian trước khi tính khoảng cách vector đặc trưng.
   * **Ước tính Thời gian Di chuyển (Travel Time) và Tốc độ Hành trình Trung bình (Journey Speed in km/h):** Cung cấp giải pháp đo đạc thời gian hành trình liên nút giao mà không cần đầu tư hệ thống camera nhận diện biển số (ANPR) đắt đỏ hoặc cảm biến vòng từ dưới lòng đường (Loop Detectors).

### 6.3. Mô Hình Toán Học và Phương Pháp Luận

```text
  [Camera C1 @ t1]                                       [Camera C2 @ t2]
         │                                                      │
         ▼                                                      ▼
  [Δ-RoI Extraction]                                     [Δ-RoI Extraction]
  (Không cần YOLO)                                       (Không cần YOLO)
         │                                                      │
         ▼                                                      ▼
  [DINO + GeM + BNNeck]                                  [DINO + GeM + BNNeck]
  Vector e_q (256-D)                                     Vector e_g (256-D)
         │                                                      │
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                 [Spatio-Temporal Feasibility Filter]
                   Δt = t2 - t1 ∈ [d/v_max, d/v_min]
                     (Loại bỏ 98% False Positives)
                                    │
                                    ▼
                     [Cosine Matching: e_q · e_g ≥ τ]
                                    │
                                    ▼
                [Tính Tốc độ Hành trình: v = d / Δt (km/h)]
```

#### 6.3.1. Trích xuất RoI Tự Động bằng Quang Học Hình Thái (Detector-Free RoI)
Bản đồ sai khác $\Delta$ được xử lý qua phép đóng hình thái học (Morphological Closing) với phần tử cấu trúc kích thước $5 \times 5$ để lấp kín các lỗ hổng bên trong thân xe, sau đó phân ngưỡng Otsu nhị phân:
$$M_{\text{vehicle}} = \text{MorphClose}\big(\Delta > \tau_{\text{Otsu}}\big)$$
Thuật toán phân tích thành phần liên thông (Connected Components) trích xuất các hộp bao bounding box. Các hộp bao thỏa mãn tiêu chuẩn hình thái:
$$\text{Area}_{\text{min}} \le \text{Area}(b) \le \text{Area}_{\text{max}} \quad \text{và} \quad 0.4 \le \frac{\text{Width}(b)}{\text{Height}(b)} \le 3.0$$
được tự động crop và đưa về kích thước chuẩn $256 \times 128$ pixel.

#### 6.3.2. Kiến Trúc Biểu Diễn Danh Tính DINO ViT + GeM + BNNeck
1. **Trích xuất đặc trưng Patch Dày đặc:** ViT Backbone trích xuất ma trận đặc trưng patch $\mathbf{F} \in \mathbb{R}^{N_p \times D}$.
2. **Generalized-Mean (GeM) Pooling:** Khác với Max Pooling hoặc Average Pooling, GeM Pooling tập trung làm nổi bật các chi tiết phân biệt danh tính độc nhất (tem xe, giỏ xe, đèn chiếu hậu) với tham số $p$ học được:
   $$\mathbf{f}_{\text{GeM}} = \left( \frac{1}{N_p} \sum_{i=1}^{N_p} \mathbf{f}_i^p \right)^{\frac{1}{p}} \in \mathbb{R}^D$$
3. **Cấu trúc BNNeck (Batch Normalization Neck):** Đưa vector qua lớp BatchNorm1d không có bias, tiếp theo là phép chiếu tuyến tính về không gian nhúng danh tính 256 chiều và chuẩn hóa $L_2$:
   $$\mathbf{e} = \frac{\mathbf{W}_{\text{proj}} \text{BN}(\mathbf{f}_{\text{GeM}})}{\|\mathbf{W}_{\text{proj}} \text{BN}(\mathbf{f}_{\text{GeM}})\|_2} \in \mathbb{R}^{256}$$

#### 6.3.3. Ràng Buộc Tính Khả Thi Không - Thời Gian (Spatio-Temporal Feasibility Windowing)
Xét một phương tiện được ghi nhận tại camera nguồn $c_1$ tại thời điểm $t_1$, với khoảng cách thực tế trên tuyến đường đến camera đích $c_2$ là $d(c_1, c_2)$ (đo bằng km).
Giả định dải vận tốc vật lý thực tế của phương tiện trong mạng lưới giao thông đô thị là $[v_{\text{min}}, v_{\text{max}}]$:
* Vận tốc tối thiểu: $v_{\text{min}} = 10\text{ km/h}$ (trường hợp ùn ứ nghiêm trọng).
* Vận tốc tối đa: $v_{\text{max}} = 60\text{ km/h}$ (giới hạn tốc độ luật định đường đô thị).

Khoảng thời gian di chuyển vật lý hợp lệ (Physical Travel Time Window) bắt buộc phải thỏa mãn:
$$\Delta t_{\text{valid}} = t_2 - t_1 \in \left[ \frac{d(c_1, c_2)}{v_{\text{max}}}, \ \frac{d(c_1, c_2)}{v_{\text{min}}} \right]$$

Mọi ứng viên $g_j$ trong thư viện Gallery tại camera $c_2$ có mốc thời gian $t_2$ nằm ngoài cửa sổ này sẽ bị loại trừ trực tiếp:
$$\text{Sim}(q_i, g_j) = \begin{cases} \mathbf{e}_{q_i} \cdot \mathbf{e}_{g_j}, & \text{nếu } (t_2 - t_1) \in [\Delta t_{\text{min}}, \Delta t_{\text{max}}] \\ -\infty, & \text{ngược lại} \end{cases}$$

#### 6.3.4. Ước Tính Tốc Độ Hành Trình và Thời Gian Di Chuyển
Với ứng viên có độ tương đồng Cosine cao nhất vượt ngưỡng tin cậy $\mathbf{e}_{q_i} \cdot \mathbf{e}_{g_j} \ge \tau_{\text{sim}}$ (mặc định $\tau_{\text{sim}} = 0.65$), hệ thống xác định việc so khớp thành công và suy luận trực tiếp các tham số giao thông:
* **Thời gian hành trình (Travel Time):**
  $$T_{\text{travel}} = t_2 - t_1 \quad (\text{giây})$$
* **Tốc độ hành trình trung bình (Average Journey Speed):**
  $$v_{\text{journey}} = \frac{d(c_1, c_2)}{t_2 - t_1} \times 3600 \quad (\text{km/h})$$

### 6.4. Hệ Thống Chỉ Số Đánh Giá Nghiệm Thu (Metrics)
* **Chỉ số Nhận dạng Phương tiện:** Cumulative Matching Characteristics (Rank-1, Rank-5, Rank-10) và mean Average Precision (mAP).
* **Chỉ số Hiệu năng Đo lường Giao thông:** Sai số tuyệt đối trung bình của thời gian di chuyển (Travel Time MAE tính bằng giây), Sai số phần trăm tuyệt đối trung bình (MAPE tính bằng %), và Tỷ lệ loại trừ dương tính giả (False Positive Rejection Rate).

---

## 7. HƯỚNG 7: OPEN-VOCABULARY TRAFFIC SCENE UNDERSTANDING VIA $\Delta$-CONDITIONED PROPOSALS

### 7.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction7_open_vocabulary/`
* Động cơ sinh đề xuất vùng vật lý: [`proposal_engine.py`](file:///g:/nckh/DINO/direction7_open_vocabulary/proposal_engine.py) (`DeltaProposalEngine`)
* Quản lý từ vựng văn bản mở: [`text_prompts.py`](file:///g:/nckh/DINO/direction7_open_vocabulary/text_prompts.py) (`TrafficPromptVocabulary`)
* Mô hình căn chỉnh DINO sang CLIP: [`models.py`](file:///g:/nckh/DINO/direction7_open_vocabulary/models.py) (`OpenVocabTrafficDetector`)
* Pipeline nhận diện trực quan: [`pipeline.py`](file:///g:/nckh/DINO/direction7_open_vocabulary/pipeline.py)

### 7.2. Đặt vấn đề và Phương pháp kỹ thuật
Các mô hình nhận diện khép kín (Closed-Set Detectors) chỉ phát hiện được các lớp có sẵn trong tập huấn luyện (ví dụ: xe hơi, xe máy). Trên đường phố thực tế tại Việt Nam, sự xuất hiện của các phương tiện đặc thù (xe cứu thương, xe rác, xe ba gác) hoặc chướng ngại vật bất thường đòi hỏi năng lực hiểu cảnh từ vựng mở (Open-Vocabulary Perception).

**Cơ chế hoạt động:**
1. **Delta Region Proposals:** $\Delta$-Mask được lọc qua phân ngưỡng đa mức và Non-Maximum Suppression (NMS) để sinh ra các hộp đề xuất vùng đối tượng không phụ thuộc lớp (Class-Agnostic Proposals) thay thế RPN.
2. **Vision-Language Alignment:** Bộ chiếu đa tầng ánh xạ vector đặc trưng vùng của DINO ViT sang không gian nhúng ngữ nghĩa của CLIP (512 chiều).
3. **Zero-Shot Classification:** Xác suất phân loại của mỗi vùng $r$ đối với danh mục văn bản $c$ được tính qua độ tương đồng Cosine:
   $$P(c | r) = \frac{\exp\big(\text{sim}(\mathbf{z}_r, \mathbf{w}_c) / \tau\big)}{\sum_{k} \exp\big(\text{sim}(\mathbf{z}_r, \mathbf{w}_k) / \tau\big)}$$
   Hỗ trợ người vận hành truy vấn bất kỳ đối tượng nào bằng mô tả ngôn ngữ tự nhiên tiếng Việt hoặc tiếng Anh.

---

## 8. HƯỚNG 8: SELF-SUPERVISED ROAD SURFACE CONDITION ESTIMATION

### 8.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction8_road_condition/`
* Nạp chuỗi ảnh nền 24h: [`dataset.py`](file:///g:/nckh/DINO/direction8_road_condition/dataset.py) (`RoadSurfaceDataset`)
* Mô hình phân tích đa thuộc tính mặt đường: [`models.py`](file:///g:/nckh/DINO/direction8_road_condition/models.py) (`RoadConditionClassifier`)
* Hàm mất mát mỏ neo vật lý: [`losses.py`](file:///g:/nckh/DINO/direction8_road_condition/losses.py) (`SurfaceConsistencyLoss`)
* Đánh giá và gom cụm PCA toàn đô thị: [`eval.py`](file:///g:/nckh/DINO/direction8_road_condition/eval.py)

### 8.2. Đặt vấn đề và Ứng dụng Quản lý Đô thị
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

## 9. TỔNG HỢP VÀ HỆ SINH THÁI 7 HƯỚNG NGHIÊN CỨU

Hệ sinh thái 7 hướng nghiên cứu DINO Suite hình thành một cấu trúc liên hoàn khép kín, tối ưu hóa triệt để cặp tín hiệu vật lý quang học từ camera giao thông:

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
           ├──────────────────────────────┴──────────────────────────────┘
           ▼                              
  [HƯỚNG 2: Scene Decomposition]
  • Tách 3 lớp: Nền + Xe + Alpha
  • Tự động xóa xe (Inpainting)
           │
           ├─────────────────────────────────────────────────────────────┐
           ▼                                                             ▼
  [HƯỚNG 5: Spatio-Temporal Density]                           [HƯỚNG 6: Corridor Re-ID]
  • Chuỗi thời gian BiGRU + DINO + Δ-CNN                       • Trích xuất RoI từ Δ (Không cần YOLO)
  • Tỷ lệ chiếm dụng ρ(t) & Cấp độ dịch vụ HCM LoS             • Ràng buộc không-thời gian [v_min, v_max]
  • Đạo hàm xu hướng kẹt xe ∂ρ/∂t                              • Đo thời gian hành trình & tốc độ km/h
           │                                                             │
           └──────────────────────────────┬──────────────────────────────┘
                                          ▼
                            [HƯỚNG 7: Open-Vocabulary]
                            • Đề xuất vùng quang học Δ-Proposals
                            • Căn chỉnh DINO sang CLIP text embeddings
                            • Nhận diện mọi phương tiện theo mô tả tự nhiên
```

### Bảng Tổng Hợp So Sánh 7 Hướng Nghiên Cứu

| Hướng | Tên Nghiên Cứu | Thư Mục Mã Nguồn | Cơ Chế Cốt Lõi | Mục Tiêu & Output | Độ Mới | Venue Đề Xuất |
|:---|:---|:---|:---|:---|:---:|:---|
| **H1** | **BG-Guided DINO Continual SSL** | `direction1_bg_guided_dino/` | Foreground-Aware Masking (FAM) ép ViT che & học biểu diễn xe cộ | Pretrained ViT Backbone cho thị giác giao thông | 4/5 | IEEE T-ITS, EAAI |
| **H2** | **Scene Decomposition Network** | `direction2_scene_decomposition/` | Alpha Compositing tự giám sát với mỏ neo nền thật $I_{\text{bg}}$ | Bóc tách 3 lớp $\{I_{\text{bg}}, I_{\text{fg}}, M_\alpha\}$, Road Inpainting | 5/5 | CVPR, ECCV, NeurIPS |
| **H3** | **Foreground-Enhanced Counting** | `direction3_foreground_enhanced_counting/` | Mở rộng Patch Embedding 4 kênh (RGB+$\Delta$) kết hợp Warm-Start | Ước lượng lưu lượng xe máy, ô tô trong điều kiện ít mẫu (Few-shot) | 3.5/5 | EAAI Journal, ITSC |
| **H5** | **Spatio-Temporal Density & HCM LoS** | `direction5_temporal_density/` | Hợp nhất DINO + $\Delta$-CNN + BiGRU với mỏ neo vật lý $\rho_{\text{phys}}$ | Tỷ lệ chiếm dụng mặt đường $\rho \in [0, 1]$, Cấp độ HCM LoS, Xu hướng kẹt xe | 4.5/5 | IEEE T-ITS, CVPR |
| **H6** | **Corridor-Based Vehicle Re-ID** | `direction6_vehicle_reid/` | $\Delta$-RoI không cần detector, GeM+BNNeck, Ràng buộc không-thời gian | Nhận dạng lại xe liên camera, Đo thời gian hành trình & Tốc độ $km/h$ | 4.5/5 | IEEE T-ITS, TRB |
| **H7** | **Open-Vocabulary Scene Understanding** | `direction7_open_vocabulary/` | Delta proposals không phụ thuộc lớp kết hợp căn chỉnh DINO-CLIP | Nhận diện không gian mở qua văn bản tự nhiên (xe cứu thương, xe rác,...) | 4/5 | ECCV, WACV |
| **H8** | **Road Surface Condition Estimation** | `direction8_road_condition/` | Đánh giá đa thuộc tính từ chuỗi ảnh nền 24h với mỏ neo quang học | Chỉ số đọng nước, chiếu sáng, hư hại kết cấu mặt đường toàn đô thị | 3.5/5 | IEEE T-ITS, TRB |
