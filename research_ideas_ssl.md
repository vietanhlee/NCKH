# Nghiên cứu Học Tự Giám Sát (Self-Supervised Learning) cho Dữ Liệu Thị Giác và Đồ Thị Giao Thông Đô Thị Quy Mô Lớn

Tài liệu này hệ thống hóa đề xuất nghiên cứu theo chuẩn mực học thuật, phân tích chi tiết hai hướng tiếp cận tự giám sát đột phá (**STG-MAE** và **Traffic-JEPA**), cùng thiết kế thực nghiệm đánh giá trên các bài toán hạ nguồn (**Downstream Tasks**) dựa trên bộ dữ liệu mạng lưới camera giao thông TP.HCM.

---

## 1. Động lực nghiên cứu và Đặt vấn đề

Trong bài toán phân tích và dự báo giao thông đô thị dựa trên mạng lưới camera giám sát:
1. **Sự mất cân đối về dữ liệu:** Việc gán nhãn thủ công (bounding box, đếm phương tiện phân loại) đòi hỏi chi phí nhân lực rất lớn, đặc biệt trong bối cảnh giao thông hỗn hợp với mật độ xe máy dày đặc và độ che khuất cao ($5.012$ ảnh được gán nhãn). Ngược lại, dữ liệu thị giác thô thu nhận liên tục $24/7$ từ $608$ camera trong $86$ ngày là nguồn tài nguyên dồi dào nhưng chưa được khai thác.
2. **Hạn chế của mô hình hai giai đoạn có giám sát:** Mô hình hai giai đoạn truyền thống gặp hiện tượng "nghẽn cổ chai thông tin" (Information Bottleneck) khi toàn bộ đặc trưng thị giác bị nén thành đại lượng vô hướng (scalar count), đồng thời gây ra hiện tượng tích lũy và lan truyền sai số nhận diện sang bài toán đồ thị hạ nguồn.

**Giải pháp đề xuất:** Áp dụng **Học tự giám sát (Self-Supervised Learning - SSL)** trong giai đoạn tiền huấn luyện (Pre-training) trên nguồn dữ liệu video thô quy mô lớn nhằm trích xuất các biểu diễn không-thời gian tổng quát, trước khi tinh chỉnh (Fine-tuning) cho các nhiệm vụ chuyên biệt.

---

## 2. Ý tưởng 1: Spatio-Temporal Graph Masked Autoencoder (STG-MAE)

### 2.1. Bản chất kiến trúc
STG-MAE mở rộng nguyên lý của Masked Autoencoder (MAE) sang miền dữ liệu đa phương thức (Ảnh + Đồ thị không-thời gian). Mô hình hoạt động dựa trên cơ chế che khuất ngẫu nhiên một tỷ lệ lớn các quan sát theo cả chiều không gian và thời gian, sau đó huấn luyện mạng học cách tái tạo lại các quan sát bị thiếu dựa trên sự tương quan topo mạng lưới và quy luật chuyển dịch dòng xe.

### 2.2. Cơ chế hoạt động
1. **Spatio-Temporal Tokenization:** Tại mỗi nút camera $i$ và thời điểm $t$, ảnh quan sát $I_{i,t}$ được chuyển đổi thành chuỗi vector đặc trưng (Patch Tokens) thông qua một Vision Backbone (ví dụ: ViT hoặc ConvNeXt gọn nhẹ).
2. **Chiến lược che khuất (Masking Strategy):** Áp dụng tỷ lệ che khuất cao ($50\% - 70\%$):
   - *Spatial Masking:* Che khuất ngẫu nhiên một tập hợp các nút camera trong mạng lưới.
   - *Temporal Masking:* Che khuất các khung hình tại các bước thời gian liên tiếp.
3. **Mã hóa bất đối xứng (Asymmetric Encoder):** Chỉ những token **không bị che khuất** mới được đưa vào Spatio-Temporal Graph Encoder. Mạng học cách nắm bắt sự tương quan giữa các tuyến đường thông qua các tầng Spatio-Temporal Attention và Chebyshev Graph Convolution.
4. **Giải mã tái tạo (Lightweight Decoder):** Nhận toàn bộ chuỗi token (bao gồm các token được học và các learnable mask token) để khôi phục lại đặc trưng thị giác hoặc các patch ảnh tại các vị trí bị giấu.
5. **Mục tiêu tối ưu:**
   $$\mathcal{L}_{\text{recon}} = \frac{1}{|M|} \sum_{(i,t) \in M} \| \hat{X}_{i,t} - X_{i,t} \|_2^2$$
   *(Trong đó $M$ là tập hợp các chỉ số vị trí bị che khuất).*

### 2.3. Ưu và nhược điểm
- **Ưu điểm:** Khả năng ép mô hình học sự phụ thuộc không-thời gian vật lý rất mạnh; cung cấp tính trực quan cao khi công bố (hiển thị trực tiếp kết quả tái tạo khung hình bị che).
- **Nhược điểm:** Việc tái tạo ở mức điểm ảnh (pixel-level) đòi hỏi chi phí tính toán lớn và dễ bị phân tán tài nguyên vào các chi tiết nền không mang thông tin dòng xe (bóng cây, màu sắc mặt đường).

### 2.4. Các bài toán đánh giá hạ nguồn (Downstream Tasks)

#### Task 1: Khôi phục dữ liệu mạng lưới bị khuyết thiếu (Missing Network Data Imputation)
- **Bối cảnh:** Trong thực tế vận hành hệ thống camera giám sát đô thị, hiện tượng suy hao đường truyền, lỗi phần cứng hoặc góc quay bị che khuất do thời tiết diễn ra thường xuyên.
- **Phương pháp đánh giá:** Thiết lập kịch bản mất mát tín hiệu ngẫu nhiên ($10\%, 20\%, 30\%$) trên $608$ nút mạng. Sử dụng mô hình STG-MAE đã tiền huấn luyện để nội suy và khôi phục chuỗi dữ liệu lưu lượng tại các trạm quan sát bị gián đoạn.
- **Chỉ số đo lường:** MAE, RMSE, WAPE giữa dữ liệu khôi phục và dữ liệu thực tế tại các nút bị che.

#### Task 2: Ước lượng lưu lượng trong điều kiện khan hiếm nhãn (Few-Shot Vehicle Counting)
- **Bối cảnh:** Đánh giá khả năng chuyển giao biểu diễn của mô hình khi triển khai tại các nút giao mới với chi phí gán nhãn tối thiểu.
- **Phương pháp đánh giá:** Cố định (freeze) hoặc tinh chỉnh nhẹ (fine-tune) Encoder đã tiền huấn luyện với các tỷ lệ nhãn hạn chế ($5\%, 10\%, 20\%$ tập nhãn $5.012$ ảnh). So sánh đường cong hội tụ và độ chính xác với mô hình huấn luyện có giám sát từ đầu (Supervised from scratch).
- **Chỉ số đo lường:** MAE, RMSE phân loại theo danh mục phương tiện (Ô tô, Xe máy).

#### Task 3: Dự báo lưu lượng không gian - thời gian đa bước (Multi-Horizon Forecasting)
- **Phương pháp đánh giá:** Tích hợp đầu dự báo đa bước (Multi-Horizon Forecasting Head) vào cấu trúc Encoder để dự báo lưu lượng dòng xe tại các mốc thời gian $t+1 \dots t+6$ ($5 - 30$ phút).
- **Chỉ số đo lường:** MAE, RMSE, MAPE so sánh với các baseline chuẩn (STGCN, GraphWaveNet, ASTGCN, MegaCRN).

#### Task 4: Phát hiện dị thường và sự cố giao thông không giám sát (Unsupervised Anomaly Detection)
- **Bối cảnh:** Phát hiện các sự cố bất thường (tai nạn giao thông, phương tiện chết máy gây xung đột dòng xe, ngập nước cục bộ) mà không cần dữ liệu sự cố gán nhãn trước.
- **Phương pháp đánh giá:** Dựa trên sai số tái tạo (Reconstruction Error). Khi xảy ra sự cố vi phạm tính quy luật thông thường của mạng lưới, mô hình sẽ không thể khôi phục chính xác trạng thái tại nút đó, dẫn đến sai số $\mathcal{L}_{\text{recon}}$ tăng đột biến và kích hoạt cảnh báo ngưỡng.
- **Chỉ số đo lường:** AUROC, AUPRC trên tập dữ liệu sự cố giao thông.

---

## 3. Ý tưởng 2: Traffic-JEPA (Joint Embedding Predictive Architecture)

### 3.1. Bản chất kiến trúc
Lấy cảm hứng từ kiến trúc Joint Embedding Predictive Architecture (JEPA), Traffic-JEPA loại bỏ hoàn toàn tầng giải mã điểm ảnh (Pixel Decoder). Thay vì cố gắng tái tạo từng chi tiết ảnh bề mặt, mô hình tập trung **dự đoán biểu diễn ngữ nghĩa trừu tượng trong không gian tiềm ẩn (Latent Representation)** của mạng lưới giao thông tại các thời điểm tương lai.

### 3.2. Cơ chế hoạt động
Kiến trúc vận hành dựa trên cơ chế hai nhánh bất đối xứng:
1. **Context Encoder:** Xử lý khung hình hiện tại tại các nút quan sát để tạo ra vector ngữ cảnh $s_{i,t} \in \mathbb{R}^d$.
2. **Graph Spatio-Temporal Predictor:** Nhận vector trạng thái ngữ cảnh kết hợp cấu trúc đồ thị topo mạng lưới đường $\tilde{L}$ để dự đoán vector biểu diễn tiềm ẩn của các nút giao lân cận ở các bước tiếp theo: $\hat{s}_{j, t+k} = \text{Predictor}(s_{i,t}, \tilde{L})$.
3. **Target Encoder:** Trích xuất vector đặc trưng thực tế $s_{j, t+k}$ từ khung hình tương lai. Trọng số của Target Encoder được cập nhật theo cơ chế trung bình động hàm mũ (Exponential Moving Average - EMA) từ Context Encoder nhằm ngăn chặn hiện tượng sụp đổ biểu diễn (Representation Collapse).
4. **Mục tiêu tối ưu:**
   $$\mathcal{L}_{\text{JEPA}} = \mathcal{D}(\hat{s}_{j, t+k}, \text{sg}(s_{j, t+k}))$$
   *(Trong đó $\mathcal{D}$ là khoảng cách chuẩn hóa L1/L2 hoặc Cosine Distance; $\text{sg}(\cdot)$ là phép toán chặn dòng gradient - stop-gradient).*

### 3.3. Ưu và nhược điểm
- **Ưu điểm:**
  - **Tối ưu hóa tài nguyên tính toán:** Tốc độ huấn luyện nhanh hơn STG-MAE từ $3 - 5$ lần; giảm thiểu đáng kể chi phí bộ nhớ VRAM do không phải lưu trữ và đạo hàm qua các tầng giải mã ảnh lớn.
  - **Khả năng tự kháng nhiễu:** Do không tối ưu hóa trên không gian điểm ảnh, mô hình tự động triệt tiêu các thành phần nhiễu tần số cao (chói sáng ống kính, vệt nước mưa, bóng râm) và chỉ giữ lại động lực học cốt lõi của dòng phương tiện.
- **Nhược điểm:** Đòi hỏi việc lựa chọn cẩn trọng các siêu tham số cập nhật EMA và kích thước không gian tiềm ẩn để đảm bảo tính ổn định hội tụ.

### 3.4. Các bài toán đánh giá hạ nguồn (Downstream Tasks)

#### Task 1: Dự báo lưu lượng từ không gian biểu diễn tiềm ẩn (Latent-to-Volume Flow Forecasting)
- **Phương pháp đánh giá:** Sử dụng vector đặc trưng tương lai $\hat{s}_{j, t+k}$ do Predictor sinh ra, đưa qua một mạng giải mã tuyến tính đa tầng (MLP Projection Head) để dự báo trực tiếp số lượng phương tiện theo từng danh mục.
- **Chỉ số đo lường:** MAE, RMSE, MAPE trên toàn bộ $608$ nút mạng lưới. Chứng minh tính ưu việt của việc dự báo trên không gian tiềm ẩn so với dự báo chuỗi số đơn thuần.

#### Task 2: Phân loại trạng thái ùn tắc mạng lưới (Congestion Regime Classification)
- **Bối cảnh:** Phân định tự động trạng thái vận hành của các trục giao thông thành các mức độ: Thông thoáng, Mật độ trung bình, Ùn ứ cục bộ, Tắc nghẽn nghiêm trọng.
- **Phương pháp đánh giá:** Áp dụng phương thức Linear Probing (cố định toàn bộ tham số của Encoder, chỉ huấn luyện một tầng phân loại Softmax trên vector đặc trưng $s_{i,t}$).
- **Chỉ số đo lường:** Macro F1-Score, Top-1 Accuracy.

#### Task 3: Truy vấn và đối sánh bối cảnh giao thông tương đồng (Traffic Scene & Anomaly Retrieval)
- **Bối cảnh:** Hỗ trợ các trung tâm điều hành giao thông thông minh (TMC) nhanh chóng truy xuất các kịch bản ùn tắc trong lịch sử có đặc tính tương đồng để áp dụng phương án phân luồng tối ưu.
- **Phương pháp đánh giá:** Đo khoảng cách tương đồng Cosine giữa vector đặc trưng của nút quan sát mục tiêu với cơ sở dữ liệu vector toàn mạng lưới.
- **Chỉ số đo lường:** Mean Average Precision at K (mAP@K), Precision@K.

#### Task 4: Đánh giá khả năng chuyển giao miền liên đô thị (Cross-City Domain Transferability)
- **Bối cảnh:** Đánh giá tính tổng quát hóa của biểu diễn khi áp dụng mô hình đã huấn luyện tại TP.HCM sang các đô thị có hạ tầng và phân bổ phương tiện tương đồng (Hà Nội, Đà Nẵng, Bangkok) mà không cần huấn luyện lại từ đầu.
- **Phương pháp đánh giá:** Đánh giá hiệu năng theo hai kịch bản: Zero-Shot Transfer và Few-Shot Adaptation.
- **Chỉ số đo lường:** So sánh mức độ suy giảm hiệu năng ($\Delta \text{MAE}$, $\Delta \text{RMSE}$) so với mô hình được huấn luyện nội tại.

---

## 4. Bảng phân tích đối sánh kỹ thuật

| Tiêu chuẩn so sánh | Ý tưởng 1: STG-MAE | Ý tưởng 2: Traffic-JEPA |
| :--- | :--- | :--- |
| **Không gian tối ưu** | Điểm ảnh & Đặc trưng bị che khuất (Pixel & Feature Space) | Không gian tiềm ẩn trừu tượng (Latent Embedding Space) |
| **Chi phí tính toán / VRAM** | Cao (yêu cầu bộ giải mã điểm ảnh) | **Thấp (nhanh hơn $3 - 5$ lần, tối ưu bộ nhớ)** |
| **Khả năng triệt tiêu nhiễu** | Trung bình (vẫn chịu ảnh hưởng bởi nhiễu bề mặt ảnh) | **Cao (loại bỏ tự nhiên các nhiễu tần số cao)** |
| **Tính minh họa trực quan** | **Rất tốt (hiển thị trực tiếp ảnh tái tạo)** | Cần phân tích qua không gian vector (t-SNE, Attention Maps) |
| **Độ mới về mặt học thuật** | Đã định hình rõ (phù hợp hội nghị Q1/A) | **Xu hướng dẫn đầu hiện nay (2024–2025, phù hợp A\*)** |
| **Phần cứng khuyến nghị** | Cụm GPU $\ge 24\text{GB}$ (RTX 3090 / 4090 / A100) | **Tối ưu trên GPU phổ thông ($12 - 16\text{GB}$ VRAM)** |

---

## 5. Khuyến nghị lộ trình nghiên cứu

1. **Giai đoạn tiền xử lý (Feature Extraction Phase):** Trích xuất và lưu trữ đặc trưng thị giác (Feature Caching) bằng backbone gọn nhẹ (DINOv2-Small hoặc ConvNeXt-Tiny) trên tập dữ liệu $50 - 100$ trạm camera trọng điểm để kiểm chứng giải thuật (Proof-of-Concept).
2. **Giai đoạn phát triển lõi (Core SSL Development):** 
   - Nếu ưu tiên hiệu năng tính toán và tính đột phá lý thuyết: **Chọn Traffic-JEPA**.
   - Nếu ưu tiên tính trực quan hóa và kiểm chứng khôi phục dữ liệu: **Chọn STG-MAE**.
3. **Giai đoạn thẩm định (Benchmark Evaluation):** Thực hiện thẩm định chéo trên hai bài toán then chốt: **Khôi phục dữ liệu khuyết thiếu (Task 1)** và **Dự báo đa bước trong điều kiện ít nhãn (Task 2 & 3)**.
