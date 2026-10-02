# BÁO CÁO KHOA HỌC: PHƯƠNG PHÁP LUẬN VÀ THIẾT KẾ KIẾN TRÚC 5 HƯỚNG NGHIÊN CỨU TRỌNG TÂM KHAI THÁC CẶP ẢNH NỀN TĨNH VÀ PHƯƠNG TIỆN TRONG THỊ GIÁC GIAO THÔNG

**Dự án:** Khai thác tự giám sát cặp ảnh Background tĩnh và Origin phương tiện phục vụ bài toán thị giác máy tính giám sát giao thông đô thị  
**Dữ liệu thực nghiệm:** Hệ thống camera giao thông đô thị TP.HCM (IC4SD-Traffic-HCM)  
**Địa chỉ mã nguồn:** Thư mục `g:/nckh/DINO/`  
**Cấu trúc tinh gọn:** Tập trung vào **5 hướng nghiên cứu trọng tâm** (H1, H2, H3, H5, H8).  
*(Lược bỏ Hướng 4, Hướng 6 và Hướng 7 nhằm tinh gọn hệ thống, loại bỏ các bài toán thiếu mỏ neo vật lý hoặc dễ bị nhiễu nhận dạng trong giao thông hỗn hợp xe máy, tập trung tối đa nguồn lực vào các bài toán đo lường động học, tách rã cảnh và hạ tầng có tính ứng dụng cao nhất).*

---

## 1. TỔNG QUAN HỆ THỐNG VÀ BẢN CHẤT DỮ LIỆU ĐẦU VÀO

Trong giám sát giao thông qua camera cố định (CCTV), hệ thống có hai nguồn tín hiệu hình ảnh tự nhiên cùng góc quan sát:
1. **Ảnh nguồn có phương tiện ($I_{\text{origin}} \in \mathbb{R}^{H \times W \times 3}$):** Chứa các đối tượng động (xe máy, ô tô, xe tải, xe buýt, người đi bộ) di chuyển trên nền đường đô thị phức tạp. Được thu thập theo định dạng `output/{stt}_{timestamp}.jpg`.
2. **Ảnh nền tĩnh sạch bóng xe ($I_{\text{bg}} \in \mathbb{R}^{H \times W \times 3}$):** Được tổng hợp thông qua thuật toán lọc trung vị theo thời gian (Temporal Median Filtering) kết hợp chuẩn hóa độ sáng không gian màu HSV trên từng camera và từng slot giờ ($00\text{h} - 23\text{h}$ theo giờ Việt Nam UTC+7). Được lưu trữ tại `traffic_backgrounds/route_{stt}/background_slot_{slot}h.jpg`.

Sự kết hợp giữa $I_{\text{origin}}$ và $I_{\text{bg}}$ cung cấp một tín hiệu vật lý quang học tiên nghiệm (Physical Prior) vô cùng mạnh mẽ:
$$\Delta(u, v) = \|I_{\text{origin}}(u, v) - I_{\text{bg}}(u, v)\|$$

Thay vì bỏ phí ảnh nền tĩnh chỉ để xem trực quan hoặc sử dụng các thuật toán trừ nền cổ điển dễ bị nhiễu do thời tiết và bóng đổ, hệ thống DINO Suite được tinh gọn thành **5 hướng nghiên cứu cốt lõi**, tích hợp $\Delta$ và $I_{\text{bg}}$ vào không gian biểu diễn sâu của Vision Transformer (ViT) để giải quyết các bài toán đo lường, bóc tách và giám sát hạ tầng trong Giao thông Thông minh (Intelligent Transportation Systems - ITS).

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

**Ý nghĩa:** 75% ngân sách che tập trung khai thác các vùng phương tiện (nơi chứa thông tin đặc trưng giao thông), và 25% ngân sách che được phân bổ ngẫu nhiên để duy trì khả năng biểu diễn tổng thể toàn khung hình.

### 2.4. Bản chất của Khái niệm "Global Views" và "Local Views" trong DINO
* **Global Views (2 góc nhìn $224 \times 224$):** Cắt tỷ lệ $[40\%, 100\%]$ diện tích. Teacher nhận view không che để làm mỏ neo ngữ nghĩa chuẩn; Student nhận view bị áp mặt nạ FAM.
* **Local Views (4 góc nhìn $96 \times 96$):** Cắt tỷ lệ $[5\%, 40\%]$ diện tích, chỉ đưa vào Student để ép mạng học quan hệ từ chi tiết bộ phận (bánh xe, biển số) suy ra tổng thể xe.
* **Hàm mất mát chưng cất tự thân:**
  $$\mathcal{L}_{\text{DINO}} = -\sum_{x \in \{V^g, V^l\}} \sum_{x' \in \{V_1^g, V_2^g\}, x' \neq x} P_t(x') \log P_s(x)$$

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
Trong đó $I_{\text{bg}}$ là lớp nền đường sạch bóng xe, $I_{\text{fg}}$ là lớp chứa phương tiện cô lập, và $M_{\alpha} \in [0, 1]^{H \times W \times 1}$ là mặt nạ mờ trong suốt (Alpha Matte).

**Mục tiêu:** Xây dựng mạng nơ-ron sâu tự giám sát hoàn toàn **TrafficDecompositionNet**. Bằng cách sử dụng ảnh nền thực tế $I_{\text{bg\_real}}$ làm mỏ neo giám sát vật lý, mạng học cách tự động bóc tách bất kỳ bức ảnh giao thông nào thành 3 lớp vật lý độc lập. Ứng dụng trực tiếp cho bài toán: **Tự động xóa sạch xe cộ trên đường (Inpainting) từ một frame duy nhất**.

### 3.3. Thiết kế Kiến trúc và Hàm Mất Mát Đa Mục Tiêu
$$\mathcal{L}_{\text{total}} = \lambda_{\text{rec}} \mathcal{L}_{\text{recon}} + \lambda_{\text{bg}} \mathcal{L}_{\text{bg}} + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparsity}} + \lambda_{\text{tv}} \mathcal{L}_{\text{tv}}$$
* $\mathcal{L}_{\text{recon}} = \|I_{\text{origin}} - \hat{I}_{\text{origin}}\|_{1} + \big(1 - \text{SSIM}(I_{\text{origin}}, \hat{I}_{\text{origin}})\big)$: Ép tái tạo đúng ảnh gốc.
* $\mathcal{L}_{\text{bg}} = \|\hat{I}_{\text{bg}} - I_{\text{bg\_real}}\|_{1}$: Mỏ neo nền thật khóa chặt lòng đường.
* $\mathcal{L}_{\text{sparsity}} = \frac{1}{HW} \sum_{u, v} M_\alpha(u, v)$: Ràng buộc thưa diện tích phương tiện.
* $\mathcal{L}_{\text{tv}} = \text{TotalVariation}(M_\alpha)$: Khử nhiễu đốm, làm mịn đường biên thân xe.

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
Đảm bảo tại epoch 0 kênh $\Delta$ đóng góp năng lượng đồng mức mà không gây sốc gradient. Đánh giá chuẩn qua phân chia phân tách không gian (Spatial Disjoint Splitting theo Camera ID) để triệt tiêu hoàn toàn hiện tượng rò rỉ dữ liệu (Data Leakage).

---

## 5. HƯỚNG 5: DỰ ĐOÁN MẬT ĐỘ KHÔNG-THỜI GIAN VÀ CẤP ĐỘ DỊCH VỤ GIAO THÔNG (SPATIO-TEMPORAL DENSITY & HCM LoS ESTIMATION)

### 5.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction5_temporal_density/`
* Nạp chuỗi thời gian & tính mỏ neo vật lý $\rho_{\text{phys}}$: [`dataset.py`](file:///g:/nckh/DINO/direction5_temporal_density/dataset.py) (`TemporalDensityDataset`, `compute_physical_density`, `discretize_los`)
* Mô hình Không - Thời Gian (ViT + Delta-CNN + BiGRU): [`models.py`](file:///g:/nckh/DINO/direction5_temporal_density/models.py) (`SpatioTemporalDensityModel`, `DeltaSpatialCNN`)
* Hàm mất mát đa nhiệm không - thời gian: [`losses.py`](file:///g:/nckh/DINO/direction5_temporal_density/losses.py) (`TemporalDensityMultiTaskLoss`)
* Pipeline huấn luyện chuỗi thời gian với AMP & Multi-GPU: [`train.py`](file:///g:/nckh/DINO/direction5_temporal_density/train.py)

### 5.2. Đặt vấn đề và Mục tiêu Khoa học Cụ thể
1. **Hạn chế của các phương pháp cũ:** Đếm từng chiếc xe máy trong điều kiện ùn tắc đặc nghẹt ở Việt Nam (hàng trăm xe máy đè lên nhau) dẫn đến sai số rất lớn. Các phương pháp InfoNCE cửa sổ thời gian ngẫu nhiên không có mỏ neo vật lý dễ bị phân kỳ khi luồng giao thông biến động đột ngột.
2. **Mục tiêu khoa học:**
   * Ước lượng **Tỷ lệ Chiếm dụng Lòng đường Liên tục (Continuous Road Space Occupancy Ratio $\rho(t) \in [0, 1]$)**.
   * Phân loại trực tiếp **Cấp độ Dịch vụ Giao thông (HCM LoS)** (Free-flow, Moderate, Slow, Gridlock).
   * Dự đoán **Đạo hàm Xu hướng Biến thiên $\frac{\partial \rho}{\partial t}$** để cảnh báo sớm nguy cơ kẹt xe.

### 5.3. Mô Hình Toán Học và Phương Pháp Luận
* **Mỏ neo vật lý tự thân:** $\rho_{\text{phys}}(t) = \frac{1}{HW} \sum_{u, v} \mathbb{I}(\Delta_t(u, v) > \tau)$.
* **Cấp độ dịch vụ chuẩn HCM LoS:**
  $$\text{LoS}(t) = \begin{cases} 
  0 \quad (\text{Free-Flow}), & \rho(t) < 0.15 \\ 
  1 \quad (\text{Moderate}), & 0.15 \le \rho(t) < 0.35 \\ 
  2 \quad (\text{Slow}), & 0.35 \le \rho(t) < 0.60 \\ 
  3 \quad (\text{Gridlock}), & \rho(t) \ge 0.60 
  \end{cases}$$
* **Hợp nhất không-thời gian:** Vector ngữ nghĩa DINO $[CLS]$ ($D$-dim) kết hợp vector hình thái $\Delta$-CNN (128-dim) qua tầng Linear chiếu về 256 chiều, nạp vào mạng nơ-ron hồi quy hai chiều 2 tầng (2-layer Bi-GRU) để sinh ra biểu diễn động học $\mathbf{h}_t \in \mathbb{R}^{256}$.
* **Hàm mất mát đa nhiệm không - thời gian:**
  $$\mathcal{L}_{\text{total}} = \lambda_{\rho} \mathcal{L}_{\text{SmoothL1}}(\hat{\rho}, \rho_{\text{phys}}) + \lambda_{\text{LoS}} \mathcal{L}_{\text{CE}}(\hat{\mathbf{y}}_{\text{LoS}}, y_{\text{LoS}}) + \lambda_{\text{trend}} \mathcal{L}_{\text{SmoothL1}}(\hat{\delta}, \delta_{\text{phys}}) + \lambda_{\text{smooth}} \frac{1}{T-1} \sum_{t=1}^{T-1} \|\hat{\rho}_{t+1} - \hat{\rho}_t\|_2^2$$

---

## 6. HƯỚNG 8: TỰ GIÁM SÁT ĐA THUỘC TÍNH ĐIỀU KIỆN MẶT ĐƯỜNG ĐÔ THỊ TỪ CHUỖI ẢNH NỀN 24 GIỜ (SELF-SUPERVISED ROAD SURFACE CONDITION ESTIMATION)

### 6.1. Mã nguồn tham chiếu
* Thư mục triển khai: `g:/nckh/DINO/direction8_road_condition/`
* Bộ nạp chuỗi ảnh nền 24h và tính toán mỏ neo vật lý quang học: [`dataset.py`](file:///g:/nckh/DINO/direction8_road_condition/dataset.py) (`RoadSurfaceDataset`, `compute_physical_surface_metrics`)
* Mô hình phân tích đa thuộc tính (Tri-Head Multi-Task): [`models.py`](file:///g:/nckh/DINO/direction8_road_condition/models.py) (`RoadConditionClassifier`)
* Hàm mất mát tự giám sát kết hợp mỏ neo vật lý: [`losses.py`](file:///g:/nckh/DINO/direction8_road_condition/losses.py) (`SurfaceConsistencyLoss`)
* Pipeline huấn luyện Multi-GPU & AMP: [`train.py`](file:///g:/nckh/DINO/direction8_road_condition/train.py) (`train_road_condition`)
* Pipeline đánh giá, trực quan hóa PCA 2D và xuất báo cáo toàn đô thị: [`eval.py`](file:///g:/nckh/DINO/direction8_road_condition/eval.py) (`plot_road_condition_pca`)
* Tài liệu kỹ thuật: [`README.md`](file:///g:/nckh/DINO/direction8_road_condition/README.md)

### 6.2. Đặt vấn đề và Động lực Khoa học
Trong toàn bộ hệ sinh thái giám sát giao thông thông minh (ITS), hơn 95% các nghiên cứu thị giác máy tính truyền thống chỉ tập trung vào việc nhận diện đối tượng tiền cảnh (xe máy, ô tô, người đi bộ) và coi mặt đường là "vùng nền vô giá trị".

Tuy nhiên, trong bối cảnh thực tế tại các đô thị nhiệt đới gió mùa như TP.HCM:
1. **Hiện tượng ngập úng và mưa cục bộ:** Các cơn mưa rào nhiệt đới gây ngập tức thời các tuyến đường huyết mạch. Việc lắp đặt cảm biến mực nước vật lý trên hàng trăm tuyến đường đòi hỏi chi phí đầu tư và bảo trì phần cứng cực lớn.
2. **Xuống cấp hạ tầng mặt đường:** Mặt đường nhựa chịu tải trọng nặng và mưa ngập liên tục dẫn tới bong tróc nhựa, nứt nẻ và hình thành ổ gà nguy hiểm. Hiện nay công tác tuần tra chủ yếu dựa trên con người ghi nhận thủ công rất chậm trễ.
3. **Biến thiên điều kiện chiếu sáng:** Hệ thống đèn cao áp ban đêm có camera đủ sáng, có camera thiếu sáng hoặc bị chói lóa.

Kho dữ liệu ảnh nền tĩnh 24 giờ $\{I_{\text{bg}}^{h=0}, \dots, I_{\text{bg}}^{h=23}\}$ trên toàn bộ 608 camera cung cấp góc nhìn toàn cảnh về **lòng đường sạch bóng xe** ở mọi khung giờ trong ngày. Đây chính là mỏ vàng dữ liệu quang học chưa từng được khai thác để xây dựng hệ thống **Tự Động Giám Sát Sức Khỏe Mặt Đường Đô Thị (Urban Road Surface Health Monitoring)** hoàn toàn không cần nhãn thủ công.

---

### 6.3. Cơ Sở Vật Lý Quang Học và Phương Pháp Luận Trích Xuất Mỏ Neo

Khác với các phương pháp học sâu thông thường phụ thuộc vào việc con người gán nhãn "đường ướt" hay "đường hỏng", Hướng 8 xây dựng một **Cơ chế Mỏ neo Vật lý Tự thân (Physics-Based Optical Anchoring)** kết hợp 3 đặc tính quang học trực tiếp từ ảnh nền tĩnh:

```text
                                [Ảnh Nền Tĩnh I_bg (h)]
                                           │
                 ┌─────────────────────────┼─────────────────────────┐
                 ▼                         ▼                         ▼
        [Không Gian Màu CIE-LAB]   [Mô Hình Phản Xạ Gương]   [Toán Tử Vi Phân Sobel]
         Luminance L ∈ [0, 255]     L(u, v) > 215             |∇L| = √(gx² + gy²)
                 │                         │                         │
                 ▼                         ▼                         ▼
      [Chu Kỳ Chiếu Sáng Nhật Khóa] [Chỉ Số Ẩm Ướt Mỏ Neo]   [Chỉ Số Hư Hại Mỏ Neo]
         0=Đêm, 1=Hoàng hôn, 2=Ngày  y_wet = clip(20*s_spec)   y_deg = clip(5*s_rough)
                 │                         │                         │
                 └─────────────────────────┼─────────────────────────┘
                                           │
                                           ▼
                       [Mỏ Neo Giám Sát Tự Thân: Targets Vật Lý]
```

#### 6.3.1. Phân tách Cường độ Chiếu sáng trong Không gian CIE-LAB
Ảnh nền $I_{\text{bg}}$ được chuyển đổi từ không gian màu RGB sang không gian màu không tương quan CIE-LAB. Kênh độ chói $L(u, v) \in [0, 255]$ phản ánh năng lượng chiếu sáng bề mặt mà không bị nhiễu bởi các sắc độ màu vạch kẻ đường:
$$\mu_L = \frac{1}{H \times W} \sum_{u=1}^H \sum_{v=1}^W \frac{L(u, v)}{255}$$

#### 6.3.2. Mô hình hóa Phản xạ Gương (Specular Reflection) phục vụ Đo lường Ẩm ướt / Ngập nước
Theo nguyên lý quang học bề mặt (Mô hình Cook-Torrance và Phong Reflection Model):
* **Mặt đường khô ráo:** Nhựa đường có độ nhám vi mô cao, tán xạ ánh sáng theo cơ chế phản xạ khuếch tán đồng đều (Diffuse Lambertian Reflection). Độ chói phân bố tương đối đồng nhất.
* **Mặt đường ẩm ướt / đọng nước mưa:** Lớp nước lấp đầy các khe vi mô của nhựa đường tạo thành một màng điện môi nhẵn bóng. Lúc này, phản xạ gương (Specular Reflection) chiếm ưu thế tuyệt đối. Dưới tác động của đèn đường cao áp hoặc ánh sáng bầu trời, bề mặt nước sinh ra các điểm phản xạ chói lóa với cường độ sáng cực đại ($L(u, v) > 215$):
  $$M_{\text{specular}}(u, v) = \mathbb{I}\big(L(u, v) > 215\big)$$
* Tỷ lệ diện tích phản chiếu gương sơ cấp được tính bằng:
  $$s_{\text{specular}} = \frac{1}{H \times W} \sum_{u=1}^H \sum_{v=1}^W M_{\text{specular}}(u, v)$$
* Mỏ neo vật lý cho mức độ ẩm ướt / ngập nước được chuẩn hóa về đoạn $[0, 1]$:
  $$y_{\text{wet\_anchor}} = \text{clip}\big(20.0 \cdot s_{\text{specular}}, \ 0.0, \ 1.0\big)$$
  Hệ số khuếch đại $20.0$ bảo đảm khi diện tích đọng nước đạt $\ge 5\%$ bề mặt khung hình, chỉ số ngập ướt đạt cực đại $1.0$.

#### 6.3.3. Mô hình hóa Độ nhám Kết cấu (Surface Roughness) qua Toán tử Gradient Sobel
Mặt đường bị suy giảm chất lượng (ổ gà, nứt nẻ mai rùa, bong tróc bê tông nhựa) có đặc trưng là sự đứt gãy kết cấu quang học, sinh ra các biến thiên độ sáng cục bộ rất cao theo cả hai phương ngang và dọc:
$$g_x(u, v) = \mathbf{S}_x * L, \quad g_y(u, v) = \mathbf{S}_y * L, \quad \|\nabla L(u, v)\| = \sqrt{g_x(u, v)^2 + g_y(u, v)^2}$$
trong đó $\mathbf{S}_x, \mathbf{S}_y$ là ma trận tích chập Sobel $3 \times 3$.
* Độ nhám bề mặt trung bình (Texture Roughness):
  $$s_{\text{roughness}} = \frac{1}{H \times W} \sum_{u=1}^H \sum_{v=1}^W \frac{\|\nabla L(u, v)\|}{255}$$
* Mỏ neo vật lý cho chỉ số suy giảm kết cấu mặt đường:
  $$y_{\text{deg\_anchor}} = \text{clip}\big(5.0 \cdot s_{\text{roughness}}, \ 0.0, \ 1.0\big)$$

#### 6.3.4. Chu kỳ Chiếu sáng Nhật khóa (Diurnal Illumination Phase Partitioning)
Thời gian ghi hình $h \in \{0, 1, \dots, 23\}$ trên mỗi camera được tự động gắn nhãn vào 3 pha chiếu sáng tự nhiên:
$$y_{\text{illum}} = \begin{cases} 
0 \quad (\text{Đêm: } 00\text{h}-05\text{h}, \ 19\text{h}-23\text{h} \text{ — Chiếu sáng nhân tạo hoặc thiếu sáng}), \\ 
1 \quad (\text{Chạng vạng: } 06\text{h}, \ 18\text{h} \text{ — Bình minh và Hoàng hôn}), \\ 
2 \quad (\text{Ngày: } 07\text{h}-17\text{h} \text{ — Chiếu sáng tự nhiên mặt trời}) 
\end{cases}$$

---

### 6.4. Thiết Kế Kiến Trúc Mạng Phân Tích Đa Thuộc Tính (RoadConditionClassifier)

```text
                         [Ảnh Nền Đường I_bg (224x224)]
                                        │
                                        ▼
                         [DINO Vision Transformer ViT]
                                        │
                                        ▼
                         Vector [CLS] Token z_road ∈ R^D
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
      [Wetness Head]             [Illum Head]             [Degradation Head]
     Linear(D → 128)            Linear(D → 128)            Linear(D → 128)
       LayerNorm                      ReLU                    LayerNorm
         ReLU                    Linear(128 → 3)                ReLU
    Linear(128 → 1)                    │                  Linear(128 → 1)
        Sigmoid                        ▼                      Sigmoid
             │                    Logits ∈ R^3                     │
             ▼                          │                          ▼
     y_hat_wet ∈ [0, 1]                Softmax             y_hat_deg ∈ [0, 1]
    (Chỉ số ẩm ướt)                     │                  (Chỉ số hư hại)
                                        ▼
                                 y_hat_illum ∈ {0,1,2}
                                 (Phân loại chiếu sáng)
```

1. **Trích xuất Đặc trưng Nền tảng:** Mạng tiếp nhận ảnh nền đường tĩnh $I_{\text{bg}}$ kích thước $224 \times 224 \times 3$. Vision Transformer trích xuất vector đại diện toàn cục $[CLS]$:
   $$\mathbf{z}_{\text{road}} = \text{ViT}(I_{\text{bg}})_{[CLS]} \in \mathbb{R}^D$$
   Với ViT-Small ($D=384$) hoặc ViT-Base ($D=768$). Trọng số của ViT có thể được đóng băng (`freeze_backbone=True`) để khai thác không gian biểu diễn tổng quát của DINO mà không sợ bị trôi dạt tham số (Catastrophic Forgetting).
2. **Ba Đầu Giải Mã Đa Nhiệm (Tri-Head Architecture):**
   * **Đầu ước lượng độ ẩm ướt / đọng nước (`wetness_head`):**
     $$\hat{y}_{\text{wet}} = \sigma\Big(\mathbf{W}_{w2} \text{ReLU}\big(\text{LayerNorm}(\mathbf{W}_{w1} \mathbf{z}_{\text{road}} + \mathbf{b}_{w1})\big) + b_{w2}\Big) \in [0, 1]$$
   * **Đầu phân loại điều kiện chiếu sáng (`illum_head`):**
     $$\hat{\mathbf{y}}_{\text{illum}} = \mathbf{W}_{i2} \text{ReLU}(\mathbf{W}_{i1} \mathbf{z}_{\text{road}} + \mathbf{b}_{i1}) + \mathbf{b}_{i2} \in \mathbb{R}^3$$
   * **Đầu ước lượng suy giảm kết cấu mặt đường (`degradation_head`):**
     $$\hat{y}_{\text{deg}} = \sigma\Big(\mathbf{W}_{d2} \text{ReLU}\big(\text{LayerNorm}(\mathbf{W}_{d1} \mathbf{z}_{\text{road}} + \mathbf{b}_{d1})\big) + b_{d2}\Big) \in [0, 1]$$

---

### 6.5. Hàm Mục Tiêu Tối Ưu Hóa (Surface Consistency Multi-Task Loss)

Hàm mất mát toàn cục tích hợp mỏ neo vật lý đa mục tiêu:
$$\mathcal{L}_{\text{surface}} = \lambda_{\text{illum}} \mathcal{L}_{\text{CE}}(\hat{\mathbf{y}}_{\text{illum}}, y_{\text{illum}}) + \lambda_{\text{wet}} \|\hat{y}_{\text{wet}} - y_{\text{wet\_anchor}}\|_2^2 + \lambda_{\text{deg}} \|\hat{y}_{\text{deg}} - y_{\text{deg\_anchor}}\|_2^2$$
Trong đó:
* $\mathcal{L}_{\text{CE}}$ là hàm mất mát Entropy chéo đa lớp (Cross-Entropy Loss).
* $\|\cdot\|_2^2$ là sai số toàn phương trung bình (Mean Squared Error).
* Trọng số cân bằng thực nghiệm: $\lambda_{\text{illum}} = 1.0$, $\lambda_{\text{wet}} = 2.0$ (ưu tiên cao cho việc phát hiện đường ngập/ướt), $\lambda_{\text{deg}} = 1.0$.

Mô hình học cách ánh xạ các mẫu hình phản chiếu quang học phức tạp trên không gian biểu diễn trừu tượng của DINO khớp với các chỉ số vật lý thực tế.

---

### 6.6. Phân Tích Đa Tạp Không Gian PCA 2D và Chẩn Đoán Hạ Tầng Toàn Đô Thị

Sau khi huấn luyện, hệ thống trích xuất vector đặc trưng mặt đường chuẩn hóa $L_2$:
$$\mathbf{e}_{\text{road}} = \frac{\mathbf{z}_{\text{road}}}{\|\mathbf{z}_{\text{road}}\|_2} \in \mathbb{R}^D$$

Áp dụng giải thuật Phân tích Thành phần Chính (PCA) để chiếu $\mathbf{e}_{\text{road}}$ xuống không gian 2 chiều ($PC_1, PC_2$):
* **Trục $PC_1$ (Chiếu sáng & Chu kỳ ngày/đêm):** Tự động phân tách rành mạch các trạng thái chiếu sáng của đô thị (Ban ngày ở góc phần tư dương, Ban đêm ở góc phần tư âm, Hoàng hôn/Bình minh nằm ở dải chuyển tiếp trung gian).
* **Trục $PC_2$ (Trạng thái ẩm ướt & Kết cấu nhựa đường):** Mặt đường khô ráo phân bố ở vùng dưới, trong khi các khung hình có đường ướt, đọng nước mưa và phản chiếu gương mạnh bị đẩy vọt lên vùng trên của đa tạp.

#### Báo cáo Hạ tầng Đô thị Tự động (`road_surface_citywide_report.csv`)
Pipeline [`eval.py`](file:///g:/nckh/DINO/direction8_road_condition/eval.py) tự động quét toàn bộ chuỗi ảnh nền của 608 camera trên địa bàn TP.HCM và xuất ra bảng chẩn đoán định lượng:
* `cam_id` / `route_id`: Mã định danh camera và nút giao thông.
* `slot_h`: Khung giờ quan sát.
* `wetness_score`: Nguy cơ đọng nước / ngập úng $\in [0, 1]$.
* `illum_class`: Mức độ chiếu sáng (0=Đêm, 1=Chạng vạng, 2=Ngày).
* `degradation_score`: Mức độ nứt nẻ / hư hại bề mặt $\in [0, 1]$.
* `luminance`: Độ sáng trung bình của mặt đường.

Dữ liệu này cung cấp công cụ đắc lực cho Trung tâm Quản lý Điều hành Giao thông Đô thị để phát hiện sớm các điểm đen ngập úng sau mưa và lên kế hoạch bảo trì thảm nhựa đường kịp thời.

---

## 7. TỔNG HỢP VÀ HỆ SINH THÁI 5 HƯỚNG NGHIÊN CỨU TRỌNG TÂM

Năm hướng nghiên cứu hình thành một hệ thống bổ trợ khép kín từ **biểu diễn nền tảng**, **phân rã quang học**, **đo lường động học dòng xe**, đến **giám sát chất lượng hạ tầng**:

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
  • Pretrained Backbone ITS      • Ước lượng lưu lượng ít mẫu • Giám sát ngập ướt, nứt nẻ
           │                              │                              │
           └──────────────────────────────┼──────────────────────────────┘
                                          ▼
                             [HƯỚNG 2: Scene Decomposition]
                             • Alpha Compositing tự giám sát
                             • Road Inpainting: Xóa sạch xe từ 1 frame
                                          │
                                          ▼
                             [HƯỚNG 5: Spatio-Temporal Density]
                             • BiGRU + DINO + Δ-CNN
                             • Tỷ lệ chiếm dụng lòng đường ρ(t)
                             • Cấp độ dịch vụ HCM LoS & Xu hướng ∂ρ/∂t
```

### Bảng Tổng Hợp So Sánh 5 Hướng Nghiên Cứu

| Hướng | Tên Nghiên Cứu | Thư Mục Mã Nguồn | Cơ Chế Cốt Lõi | Mục Tiêu & Output | Độ Mới | Venue Đề Xuất |
|:---|:---|:---|:---|:---|:---:|:---|
| **H1** | **BG-Guided DINO Continual SSL** | `direction1_bg_guided_dino/` | Foreground-Aware Masking (FAM) ép ViT che & học biểu diễn xe cộ | Pretrained ViT Backbone chuyên biệt cho giao thông | 4/5 | IEEE T-ITS, EAAI |
| **H2** | **Scene Decomposition Network** | `direction2_scene_decomposition/` | Alpha Compositing tự giám sát với mỏ neo nền thật $I_{\text{bg}}$ | Bóc tách 3 lớp $\{I_{\text{bg}}, I_{\text{fg}}, M_\alpha\}$, Road Inpainting | 5/5 | CVPR, ECCV, NeurIPS |
| **H3** | **Foreground-Enhanced Counting** | `direction3_foreground_enhanced_counting/` | Mở rộng Patch Embedding 4 kênh (RGB+$\Delta$) kết hợp Warm-Start | Ước lượng lưu lượng xe máy, ô tô trong điều kiện ít mẫu (Few-shot) | 3.5/5 | EAAI Journal, ITSC |
| **H5** | **Spatio-Temporal Density & HCM LoS** | `direction5_temporal_density/` | Hợp nhất DINO + $\Delta$-CNN + BiGRU với mỏ neo vật lý $\rho_{\text{phys}}$ | Tỷ lệ chiếm dụng mặt đường $\rho \in [0, 1]$, Cấp độ HCM LoS, Xu hướng kẹt xe $\partial\rho/\partial t$ | 4.5/5 | IEEE T-ITS, CVPR |
| **H8** | **Road Surface Condition Estimation** | `direction8_road_condition/` | Đánh giá đa thuộc tính từ chuỗi ảnh nền 24h với mỏ neo quang học | Chỉ số đọng nước, chiếu sáng, hư hại kết cấu mặt đường toàn đô thị | 4/5 | IEEE T-ITS, TRB |
