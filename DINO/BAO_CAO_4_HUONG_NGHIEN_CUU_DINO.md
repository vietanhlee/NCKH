# BÁO CÁO KHOA HỌC: PHƯƠNG PHÁP LUẬN VÀ THIẾT KẾ KIẾN TRÚC 4 HƯỚNG NGHIÊN CỨU KHAI THÁC CẶP ẢNH NỀN TĨNH VÀ PHƯƠNG TIỆN TRONG THỊ GIÁC GIAO THÔNG

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

Thay vì bỏ phí ảnh nền tĩnh chỉ để xem trực quan, hệ thống DINO Suite được xây dựng gồm 4 hướng nghiên cứu độc lập nhưng bổ trợ lẫn nhau, tận dụng $\Delta$ và $I_{\text{bg}}$ để giải quyết triệt để các bài toán cốt lõi của thị giác máy tính trong giao thông thông minh (Intelligent Transportation Systems - ITS).

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

## 3. HƯỚNG 2: ZERO-SHOT VEHICLE SEMANTIC SEGMENTATION

### 3.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction2_zero_shot_segmentation/`
* Trích xuất đặc trưng PCA từ ViT: [`pca_extractor.py`](file:///g:/nckh/DINO/direction2_zero_shot_segmentation/pca_extractor.py) (`DINOPCAExtractor`)
* Động cơ dung hợp nhãn giả: [`fusion.py`](file:///g:/nckh/DINO/direction2_zero_shot_segmentation/fusion.py) (`MaskFusionEngine`)
* Kiến trúc phân đoạn Student nhẹ và Dice Loss: [`segmentor.py`](file:///g:/nckh/DINO/direction2_zero_shot_segmentation/segmentor.py) (`VehicleSegmentor`, `LightweightSegDecoder`, `DiceLoss`)
* Pipeline suy luận và trực quan hóa bài báo: [`run_segmentation.py`](file:///g:/nckh/DINO/direction2_zero_shot_segmentation/run_segmentation.py)

### 3.2. Đặt vấn đề và Mục tiêu
Việc gán nhãn mặt nạ phân đoạn pixel (Pixel-level Segmentation Mask) cho dữ liệu giao thông thực tế đòi hỏi chi phí nhân công khổng lồ (trung bình mất 15–30 phút cho một bức ảnh đông đúc xe cộ tại TP.HCM). 

**Mục tiêu:** Tự động tạo ra mặt nạ phân đoạn pixel chính xác cho phương tiện mà **hoàn toàn không cần con người can thiệp (Zero-Shot)**, bằng cách khai thác sự giao thoa bù trừ giữa tín hiệu quang học cấp thấp (Low-level Physics) và tín hiệu ngữ nghĩa cấp cao của DINOv3 (High-level Deep Semantics). Dùng nhãn giả chất lượng cao này để huấn luyện một mạng Student SegHead siêu nhẹ (>60 FPS) phục vụ camera giám sát tại biên.

### 3.3. Phương pháp kỹ thuật

```
                     ┌───────────────────────────────┐
                     │ Ảnh Origin + Ảnh Background   │
                     └───────────────┬───────────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
   [Nhánh Vật Lý: Δ-Map]                         [Nhánh Ngữ Nghĩa: DINOv3 ViT]
   • Trừ nền LAB                                 • Trích xuất Patch Tokens
   • Ngưỡng tự động Otsu                         • Chiếu PCA: Phân tích PC1
   • Phép toán hình thái học                     • Nội suy song tuyến tính
             │                                               │
             │ (Mặt nạ nhạy biên, dính bóng đổ)              │ (Mặt nạ ngữ nghĩa xe, biên thô)
             └───────────────────────┬───────────────────────┘
                                     │
                                     ▼
                        [Động cơ Dung Hợp (Fusion Engine)]
                        • Giao thoa: M_fused = M_Δ ∩ M_PC1
                        • Khử bóng đổ mặt đường
                        • Lọc viền: Guided/Bilateral Filtering
                                     │
                                     ▼
                     [Pseudo Ground-Truth Segmentation Mask]
```

#### 3.3.1. Nhánh 1: Tín hiệu sai khác vật lý ($\Delta$-Mask)
* Tính bản đồ sai khác $\Delta$ qua phép trừ nền trong không gian LAB.
* Phân ngưỡng nhị phân tự động thích nghi theo thuật toán Otsu:
  $$\tau^* = \arg\max_{\tau} \sigma_B^2(\tau)$$
* Áp dụng phép toán hình thái học:
  * Phép mở (Morphological Open) với kernel hình elip $5 \times 5$ nhằm triệt tiêu nhiễu hạt cảm biến camera.
  * Phép đóng (Morphological Close) với kernel $11 \times 11$ để lấp đầy các khoảng trống bên trong thân xe đồng màu.
* **Đặc tính:** Cho đường biên bao quanh đối tượng rất sắc nét theo từng pixel, nhưng nhược điểm chí mạng là **luôn bị dính bóng đổ của xe in xuống mặt đường** vì bóng đổ cũng làm thay đổi giá trị điểm ảnh so với nền tĩnh.

#### 3.3.2. Nhánh 2: Tín hiệu ngữ nghĩa sâu (DINOv3 PCA Segmentation)
* Đưa ảnh origin qua mô hình DINO ViT để lấy ma trận đặc trưng patch tokens ở tầng Transformer cuối cùng:
  $$X \in \mathbb{R}^{N_{\text{patches}} \times D}$$
  với $N_{\text{patches}} = 196$, số chiều biểu diễn $D = 384$ (ViT-Small) hoặc $768$ (ViT-Base).
* Thực hiện phân tích thành phần chính (PCA) trên tập vector patch đã chuẩn hóa trung bình:
  $$(X - \bar{X}) = U \Sigma V^T$$
* Chiếu ma trận đặc trưng lên vector riêng thứ nhất $v_1$ (thành phần giải thích phương sai lớn nhất):
  $$\mathbf{z}_{\text{PC1}} = (X - \bar{X}) v_1 \in \mathbb{R}^{N_{\text{patches}}}$$
* **Cơ sở khoa học:** Theo khám phá của Caron et al. và Oquab et al., cơ chế Self-Attention trong DINO tự động gom cụm các patch có chung thuộc tính ngữ nghĩa trừu tượng. Vector riêng PC1 phân tách rõ rệt giữa hai phân phối: phân phối của "vật thể tiền cảnh" và phân phối của "bối cảnh nền".
* **Đặc tính:** PC1 **hoàn toàn không bị đánh lừa bởi bóng đổ mặt đường** (bởi vì bóng đổ về bản chất ngữ nghĩa sâu vẫn là bề mặt nhựa đường, không chứa cấu trúc cơ khí hay hình thái của xe). Nhược điểm là độ phân giải không gian thô ($14 \times 14$ patches).

#### 3.3.3. Cơ chế Dung hợp (Mask Fusion Engine)
Sự kết hợp giữa hai nhánh triệt tiêu hoàn toàn nhược điểm của nhau:
$$M_{\text{fused}} = \text{BilateralFilter}\Big( M_{\Delta} \cap \text{Threshold}(\text{Resize}(\mathbf{z}_{\text{PC1}})) \Big)$$
* Phép giao $\cap$ loại bỏ 100% bóng đổ trên mặt đường (vì nhánh PCA loại bỏ bóng đổ).
* Phép giao $\cap$ loại bỏ các dao động giả của nền tĩnh như cành cây đung đưa, mặt đường loang nước (vì nhánh PCA xác định đó không phải xe).
* Bộ lọc song phương (Bilateral Filter) khôi phục lại đường biên chính xác đến từng pixel của vỏ xe dựa trên gradient độ sáng cục bộ của ảnh gốc.

---

## 4. HƯỚNG 3: SELF-SUPERVISED SCENE DECOMPOSITION NETWORK (TRAFFIC-DECOMPOSE)

### 4.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction3_scene_decomposition/`
* Bộ nạp dữ liệu cặp ảnh đồng bộ: [`dataset.py`](file:///g:/nckh/DINO/direction3_scene_decomposition/dataset.py) (`DecompositionDataset`)
* Kiến trúc phân rã cảnh 3 nhánh: [`models.py`](file:///g:/nckh/DINO/direction3_scene_decomposition/models.py) (`TrafficDecompositionNet`)
* Hàm mất mát giám sát vật lý đa mục tiêu: [`losses.py`](file:///g:/nckh/DINO/direction3_scene_decomposition/losses.py) (`DecompositionLoss`)
* Huấn luyện mạng: [`train.py`](file:///g:/nckh/DINO/direction3_scene_decomposition/train.py)
* Suy luận Inpainting xóa xe tự động: [`infer.py`](file:///g:/nckh/DINO/direction3_scene_decomposition/infer.py)

### 4.2. Đặt vấn đề và Mục tiêu
Trong đồ họa máy tính và thị giác vật lý, một khung cảnh quan sát được mô hình hóa theo công thức hòa trộn Alpha (Alpha Compositing Formulation):
$$I_{\text{origin}} = M_{\alpha} \odot I_{\text{fg}} + (1 - M_{\alpha}) \odot I_{\text{bg}}$$
Trong đó:
* $I_{\text{bg}} \in \mathbb{R}^{H \times W \times 3}$: Lớp nền đường sạch bóng xe.
* $I_{\text{fg}} \in \mathbb{R}^{H \times W \times 3}$: Lớp chứa các thực thể phương tiện cô lập.
* $M_{\alpha} \in [0, 1]^{H \times W \times 1}$: Mặt nạ mờ trong suốt (Alpha Matte) biểu diễn mức độ hiện diện của phương tiện tại từng tọa độ không gian.
* $\odot$: Phép nhân Hadamard (nhân từng phần tử).

Thông thường, việc phân rã một bức ảnh duy nhất $I_{\text{origin}}$ thành cả 3 thành phần $\{I_{\text{bg}}, I_{\text{fg}}, M_{\alpha}\}$ là một bài toán nghịch đảo vô nghiệm xác định (Ill-posed Inverse Problem) vì số lượng ẩn số gấp 3 lần số lượng phương trình quan sát.

**Mục tiêu:** Xây dựng một mạng nơ-ron sâu tự giám sát hoàn toàn mang tên **TrafficDecompositionNet**. Bằng cách sử dụng ảnh nền thực tế $I_{\text{bg\_real}}$ làm mỏ neo giám sát vật lý, mạng học cách tự động bóc tách bất kỳ bức ảnh giao thông nào thành 3 lớp vật lý độc lập. Ứng dụng trực tiếp cho bài toán: **Tự động xóa sạch xe cộ trên đường (Inpainting) từ một frame duy nhất** mà không cần thuật toán vá ảnh truyền thống.

### 4.3. Thiết kế Kiến trúc và Hàm Mất Mát Đa Mục Tiêu

```
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

#### 4.3.1. Thành phần 1: Mất mát tái tạo khung cảnh ($\mathcal{L}_{\text{recon}}$)
Đảm bảo rằng khi hòa trộn lớp xe $I_{\text{fg}}$ và lớp nền $I_{\text{bg}}$ thông qua mặt nạ $M_{\alpha}$, kết quả phải tái tạo hoàn hảo bức ảnh ban đầu:
$$\mathcal{L}_{\text{recon}} = \|I_{\text{origin}} - \hat{I}_{\text{origin}}\|_{1} + \big(1 - \text{SSIM}(I_{\text{origin}}, \hat{I}_{\text{origin}})\big)$$
Việc kết hợp khoảng cách $L_1$ và chỉ số tương đồng cấu trúc SSIM (Structural Similarity Index) giúp bảo toàn cả độ chính xác màu sắc lẫn độ sắc nét của các cạnh đối tượng.

#### 4.3.2. Thành phần 2: Mất mát giám sát nền thực tế ($\mathcal{L}_{\text{bg}}$)
Đây là **đóng góp khoa học then chốt (Core Novelty)** giúp giải quyết tính chất vô nghiệm xác định của bài toán. Nhánh dự đoán nền $I_{\text{bg}}$ bị cưỡng chế phải hội tụ về ảnh nền thật $I_{\text{bg\_real}}$ tương ứng của khung giờ đó:
$$\mathcal{L}_{\text{bg}} = \|\hat{I}_{\text{bg}} - I_{\text{bg\_real}}\|_{1}$$
Nhờ có mỏ neo này, Decoder nền bị khóa chặt vào cấu trúc lòng đường tĩnh, buộc Decoder phương tiện ($I_{\text{fg}}$) và Decoder Alpha ($M_{\alpha}$) phải hấp thụ toàn bộ phần dư thừa chuyển động (chính là xe cộ).

#### 4.3.3. Thành phần 3: Ràng buộc thưa diện tích phương tiện ($\mathcal{L}_{\text{sparsity}}$)
Nếu không có ràng buộc này, mạng nơ-ron có thể chọn một nghiệm suy biến tầm thường: gán $M_{\alpha} = 1$ ở khắp mọi nơi và sao chép toàn bộ ảnh gốc vào $I_{\text{fg}}$. Để ngăn chặn điều này, hàm mất mát chuẩn hóa $L_1$ trên mặt nạ Alpha được áp dụng:
$$\mathcal{L}_{\text{sparsity}} = \frac{1}{H \cdot W} \sum_{u=1}^{H} \sum_{v=1}^{W} M_{\alpha}(u, v)$$
Ép diện tích xe cộ phải thưa thớt, phản ánh đúng thực tế phương tiện chỉ chiếm một phần diện tích mặt đường.

#### 4.3.4. Thành phần 4: Ràng buộc làm mịn Total Variation ($\mathcal{L}_{\text{tv}}$)
Khử hiện tượng nhiễu đốm hạt tiêu và viền răng cưa quanh thân xe trên mặt nạ $M_{\alpha}$:
$$\mathcal{L}_{\text{tv}} = \frac{1}{H \cdot W} \sum_{u=1}^{H} \sum_{v=1}^{W} \left( |M_{\alpha}(u+1, v) - M_{\alpha}(u, v)| + |M_{\alpha}(u, v+1) - M_{\alpha}(u, v)| \right)$$

---

## 5. HƯỚNG 4: FOREGROUND-ENHANCED TRAFFIC COUNTING (STAGE 1 UPGRADE)

### 5.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction4_foreground_enhanced_counting/`
* Nạp nhãn đếm và tạo tensor 4 kênh: [`dataset.py`](file:///g:/nckh/DINO/direction4_foreground_enhanced_counting/dataset.py) (`FGCountingDataset`)
* Kiến trúc ViT mở rộng 4 kênh và Warm-start: [`models.py`](file:///g:/nckh/DINO/direction4_foreground_enhanced_counting/models.py) (`DINOv3FGCountingModel`, `adapt_patch_embed_to_4ch`, `RegressionHead`)
* Huấn luyện mô hình đếm xe: [`train.py`](file:///g:/nckh/DINO/direction4_foreground_enhanced_counting/train.py)
* Đánh giá hiệu năng và tự động xuất bảng $\text{\LaTeX}$: [`evaluate.py`](file:///g:/nckh/DINO/direction4_foreground_enhanced_counting/evaluate.py)

### 5.2. Đặt vấn đề và Mục tiêu
Bài toán ước lượng lưu lượng phương tiện (Đếm số lượng xe máy, ô tô và tổng lưu lượng) từ một camera CCTV duy nhất gặp rất nhiều thách thức lớn:
1. **Hiện tượng che khuất nghiêm trọng (Heavy Occlusion):** Trong các khung giờ cao điểm tại TP.HCM, xe máy ken đặc che lấp lẫn nhau, mô hình 3 kênh RGB truyền thống rất khó nhận diện ranh giới từng phương tiện.
2. **Biến động chiếu sáng khắc nghiệt:** Ban đêm đèn pha xe gây lóa camera, ban ngày bóng đổ của nhà cao tầng và cây cối làm biến dạng đặc trưng thị giác.
3. **Bài toán học ít mẫu (Few-Shot Regime):** Việc gán nhãn số lượng xe cho từng khung hình rất tốn kém; cần một mô hình hoạt động vượt trội khi chỉ có $5\%$, $10\%$ hoặc $20\%$ dữ liệu có nhãn.

**Mục tiêu:** Mở rộng tầng Patch Embedding của DINOv3 từ 3 kênh tiêu chuẩn (RGB) lên 4 kênh (RGB + $\Delta$), trực tiếp tiêm tín hiệu chuyển động vật lý vào tầng sâu của mạng. Cải thiện rõ rệt chỉ số sai số tuyệt đối trung bình (MAE) và sai số bình phương trung bình căn (RMSE).

### 5.3. Phương pháp kỹ thuật

#### 5.3.1. Cấu trúc Tensor Đầu Vào 4 Kênh
Mỗi mẫu dữ liệu huấn luyện được biểu diễn dưới dạng tensor 4 chiều không gian:
$$X_{\text{4ch}} = [\text{R}, \text{G}, \text{B}, \Delta_{\text{norm}}] \in \mathbb{R}^{4 \times H \times W}$$
Trong đó kênh thứ 4 được chuẩn hóa về cùng phân phối động với các kênh màu:
$$\Delta_{\text{norm}} = \frac{\Delta - 0.5}{0.5} \in [-1.0, 1.0]$$

#### 5.3.2. Kỹ thuật Khởi tạo Thích ứng Ấm (Warm-Start Patch Embedding Adaptation)
Trong Vision Transformer chuẩn, lớp Patch Embedding là một phép tích chập 2D biến đổi patch ảnh kích thước $P \times P$ thành vector không gian $D$ chiều:
$$W_{\text{3ch}} \in \mathbb{R}^{D \times 3 \times P \times P}, \quad b \in \mathbb{R}^{D}$$

Khi mở rộng thêm kênh thứ 4 ($\Delta$), trọng số biến thành $W_{\text{4ch}} \in \mathbb{R}^{D \times 4 \times P \times P}$. Nếu ta khởi tạo ngẫu nhiên trọng số của kênh thứ 4 (ví dụ lấy mẫu Gaussian ngẫu nhiên), các giá trị kích hoạt (Activation) ở tầng Transformer đầu tiên sẽ bị nhiễu loạn nghiêm trọng, phá hủy toàn bộ tri thức tiền huấn luyện (Pretrained Representation) của DINOv3.

Để giải quyết vấn đề này, thuật toán **Warm-Start Weight Adaptation** được áp dụng:
$$W_{\text{4ch}}[:, 0:3, :, :] = W_{\text{pretrained}}$$
$$W_{\text{4ch}}[:, 3, :, :] = \frac{1}{3} \sum_{c=0}^{2} W_{\text{pretrained}}[:, c, :, :]$$

**Ý nghĩa toán học và nghiệp vụ:**
* Ba kênh đầu giữ nguyên $100\%$ các bộ lọc trích xuất cạnh, kết cấu và hình khối màu sắc đã được Meta AI huấn luyện trên hàng trăm triệu bức ảnh.
* Trọng số kênh thứ 4 được gán bằng trung bình cộng cường độ của 3 kênh RGB. Vì bản đồ sai khác $\Delta$ biểu diễn cường độ tương phản thị giác, việc khởi tạo này đảm bảo tại epoch 0, kênh $\Delta$ đóng góp vào hàm kích hoạt với thang đo năng lượng hoàn toàn đồng nhất với các kênh màu, không gây sốc gradient cho mạng nơ-ron.

#### 5.3.3. Cơ chế Đánh giá Khách quan Chống Rò rỉ Bối cảnh (Spatial Disjoint Splitting)
* Tuyệt đối không chia tập Train/Test ngẫu nhiên theo từng frame ảnh (vì hai ảnh kế tiếp nhau từ cùng một camera sẽ có nền đường giống hệt nhau, dẫn tới hiện tượng rò rỉ dữ liệu - Data Leakage).
* Dữ liệu được phân chia theo danh sách Camera ID độc lập: Camera dùng để huấn luyện sẽ không bao giờ xuất hiện trong tập kiểm thử. Điều này chứng minh năng lực tổng quát hóa của kênh $\Delta$ đối với các tuyến đường hoàn toàn mới.
* Tự động tính toán các chỉ số thống kê chuẩn khoa học:
  $$\text{MAE} = \frac{1}{M} \sum_{i=1}^{M} |y_i - \hat{y}_i|, \quad \text{RMSE} = \sqrt{\frac{1}{M} \sum_{i=1}^{M} (y_i - \hat{y}_i)^2}$$
  và tự động xuất bảng so sánh giữa mô hình 3 kênh RGB và 4 kênh RGB+$\Delta$ ra định dạng mã nguồn $\text{\LaTeX}$ chuẩn IEEE Transactions.

---

## 6. TỔNG KẾT VÀ BẢN ĐỒ TIẾN TRÌNH THỰC NGHIỆM

Bốn hướng nghiên cứu trên tạo thành một hệ sinh thái khoa học khép kín và có tính kế thừa chặt chẽ:

```
[Dữ Liệu Thô: Camera TP.HCM]
              │
              ▼
[Cặp Ảnh Vật Lý: Origin + Background]
              │
              ├────────────────────────────────────────┐
              ▼                                        ▼
   [HƯỚNG 1: BG-Guided DINO]                [HƯỚNG 4: FG Counting]
   • FAM: Ép ViT học biểu diễn xe cộ        • Thêm kênh thứ 4 (RGB + Δ)
   • Tạo ViT Backbone chuyên biệt ITS       • Cải thiện trực tiếp MAE/RMSE
              │                                        │
              ▼                                        ▼
   [HƯỚNG 2: Zero-Shot Segmentation]        [Báo Cáo Nghiệm Thu / Bài Báo]
   • DINO PCA ∩ Δ-Mask
   • Tự sinh nhãn pixel không cần người
              │
              ▼
   [HƯỚNG 3: Scene Decomposition]
   • Phân rã vật lý: Mặt đường + Xe + Alpha
   • Đỉnh cao Novelty (CVPR / ECCV)
```

1. **Hướng 1** giải quyết bài toán biểu diễn nền tảng (Foundation SSL Representation for Traffic).
2. **Hướng 2** giải quyết bài toán thiếu hụt nhãn phân đoạn pixel (Data Annotation Bottleneck).
3. **Hướng 3** mở rộng biên giới lý thuyết về phân rã cảnh quang học tự giám sát (Physics-guided Scene Decomposition).
4. **Hướng 4** mang lại giá trị ứng dụng thực tiễn ngay lập tức cho bài toán giám sát lưu lượng giao thông đô thị.
