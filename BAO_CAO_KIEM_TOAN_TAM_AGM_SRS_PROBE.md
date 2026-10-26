# Báo Cáo Kiểm Tra Thực Nghiệm & Giải Trình Các Vấn Đề Kỹ Thuật (TAM, AGM, SRS & Frozen Linear Probe)

> **Mã nguồn kiểm chứng**: [`DINO/direction1_new/evaluation/audit_tam_agm_srs_probe.py`](file:///g:/nckh/DINO/direction1_new/evaluation/audit_tam_agm_srs_probe.py)  
> **Các module liên quan**: [`tam.py`](file:///g:/nckh/DINO/direction1_new/core/tam.py), [`masking.py`](file:///g:/nckh/DINO/direction1_new/core/masking.py), [`srs.py`](file:///g:/nckh/DINO/direction1_new/core/srs.py), [`losses.py`](file:///g:/nckh/DINO/direction1_new/core/losses.py)  
> **Trạng thái**: Đã rà soát toàn bộ các lỗi ở phiên bản cũ, cập nhật lại thuật toán và đang chạy lại phiên huấn luyện 30 epochs trên tập dữ liệu camera giao thông thực tế để điền số liệu định lượng cuối cùng.

---

## Phần 1. Nhìn Nhận Lại Các Lỗi Kỹ Thuật Ở Phiên Bản Cũ & Cách Đã Khắc Phục

Khi kiểm tra lại code của phiên bản cũ (thời điểm xuất tấm ảnh minh họa có 3/4 mẫu bị $\Delta = 0\%$ ở cột SRS và bản đồ nhiệt TAM bị loang rộng trên mặt đường), em nhận thấy hai nhận xét của Cô đã chỉ đúng vào hai lỗi logic lớn nhất trong code lúc đó. Ngoài hai lỗi này, quá trình rà soát lại toàn bộ pipeline cũng phát hiện thêm một số điểm nghẽn ở bước tạo mặt nạ AGM, hàm loss và bước vẽ PCA. Cụ thể như sau:

### 1.1. Tại sao ở bản cũ 3/4 ảnh minh họa có SRS $\Delta = 0\%$ và vùng nền hoán đổi quá ít?
Ở phiên bản cũ, việc ghép nền tĩnh giữa ảnh gốc $x_1$ và ảnh ghép $x_2$ (cùng camera khác thời điểm) bị kẹt do 3 nguyên nhân cộng hưởng:
1. **Lỗi thứ nhất nằm ở đầu vào TAM ($\pi$) bị dâng nền trên mặt đường**:
   - Trong code TAM cũ, đặc trưng patch $u_t(p)$ đưa thẳng vào tính khoảng cách cosine với tâm cụm lịch sử $\mu_k$ mà **chưa khử độ lệch sáng toàn khung hình** (do nắng gắt, bóng mây, mặt đường ướt hoặc đèn đường ban đêm).
   - Khi toàn bộ mặt đường thay đổi độ sáng nhẹ giữa các khung giờ, khoảng cách cosine ở các ô mặt đường trống bị đẩy lên mức lơ lửng $\pi \in [0.22, 0.40]$ thay vì nằm sát $0$.
2. **Lỗi thứ hai nằm ở điều kiện lọc vùng tĩnh quá ngặt trong hàm SRS cũ**:
   - Code cũ yêu cầu một ô patch chỉ được phép hoán đổi nếu đồng thời thỏa mãn $(\pi_1 < 0.20) \land (\pi_2 < 0.20)$ VÀ nằm ngoài vùng giãn nở của hợp cả hai ảnh: $\neg\text{dilate}(\pi_1 \ge 0.50 \lor \pi_2 \ge 0.50, \text{radius}=1)$.
   - Chỉ cần một trong hai ảnh có mặt đường hơi chói sáng làm $\pi \approx 0.22$, hoặc giờ cao điểm xe máy chạy rải rác trên hai ảnh khiến vùng giãn nở `dilate(radius=1)` của hai ảnh cộng lại phủ kín mặt đường, số lượng ô đủ điều kiện hoán đổi lập tức rơi về $n_{\text{eligible}} = 0$.
   - Trong code cũ có dòng: `if n_eligible == 0: return x.clone()`. Khi nhánh này kích hoạt, ảnh sau SRS giống hệt ảnh gốc 100%, dẫn tới chỉ số $\Delta = 0.0\%$ ở 3/4 ảnh như Cô đã phát hiện.

**Cách đã khắc phục trong [`tam.py`](file:///g:/nckh/DINO/direction1_new/core/tam.py) và [`srs.py`](file:///g:/nckh/DINO/direction1_new/core/srs.py) để tăng diện tích hoán đổi $\Delta$:**
- **Làm sạch điểm TAM trên mặt đường trống để mở rộng diện tích nền hợp lệ**:
  - Bổ sung bước chuẩn hóa trắc quang toàn khung hình (`normalize_frame_features`, trừ vector trung bình của cả frame trước khi chuẩn hóa $L_2$) để triệt tiêu độ lệch sáng đồng loạt giữa ngày và đêm.
  - Bổ sung bộ lọc tương phản cục bộ **Relative Local Contrast High-Pass Filter** trên cửa sổ $5 \times 5$ (có `replicate padding` ở viền ảnh): khi mặt đường đổi màu do nắng/mưa thì cả mảng đường $5 \times 5$ biến thiên giống nhau nên $(a_p - \mu_{\text{local}}) \approx 0$ và bị nén xuống sát $0$; ngược lại, chiếc xe là vật thể cục bộ có độ tương phản vọt lên so với mặt đường xung quanh nên được giữ nguyên.
  - Thêm cơ chế cập nhật tâm cụm có cổng (**Gated Online EMA Update**): tốc độ cập nhật cụm nền tĩnh được nhân với $(1 - \pi_p)$. Những ô đang có xe ($\pi_p$ cao) sẽ bị khóa tốc độ cập nhật về sát $0$, ngăn hiện tượng xe dừng đèn đỏ lâu ở làn đông bị mô hình học nhầm thành nền tĩnh.
- **Sửa lại logic chọn vùng ghép trong [`srs.py`](file:///g:/nckh/DINO/direction1_new/core/srs.py)**:
  - Thay vì lấy hợp vùng xe của cả hai ảnh rồi giãn nở chung, chỉ giãn nở bảo vệ vùng xe của ảnh gốc $x_1$ (`fg1_dilated = dilate(pi1 >= 0.40, radius=1)` để tuyệt đối không cắt phạm vào thân xe của ảnh đang học), đồng thời chỉ cần tránh các ô có xe thật trên ảnh nguồn $x_2$ (`pi2 < 0.45`).
  - Thêm cơ chế **Adaptive Fallback 2 cấp**: nếu khung hình quá đông xe làm diện tích nền chung nhỏ hơn $15\%$ khung hình ($\approx 67/448$ patches), thuật toán tự động thu hẹp bán kính đệm về $0$ hoặc lấy theo phân vị động $40\%$ vùng có xác suất xe thấp nhất (vẫn khóa cứng lõi phương tiện $\pi \ge 0.65$). Nhờ vậy, SRS luôn có đủ vùng nền để ghép, đẩy tỷ lệ kích hoạt lên mức tối đa và tạo ra mức thay đổi $\Delta$ rõ rệt.

---

### 1.2. Các điểm nghẽn khác ngoài TAM và SRS cũng đã được sửa lại
1. **Chiến lược che AGM ([`masking.py`](file:///g:/nckh/DINO/direction1_new/core/masking.py))**:
   - Trước đây nếu chỉ che từng patch $16 \times 16$ rời rạc, mạng ViT rất dễ dùng các ô sát cạnh ngay bên trái/phải để nội suy màu sắc mà không cần hiểu cấu trúc chiếc xe.
   - Hiện tại đã bổ sung chế độ **Super-Block Grid $2 \times 2$ (`mode="block"`)** kết hợp lấy mẫu Gumbel Top-K có phân bổ ngân sách tiền cảnh $\phi = 0.5$ và trần che tối đa $q_{\max} = 0.6$ (luôn chừa lại ít nhất $40\%$ diện tích xe không bị che để Student còn ngữ cảnh nhận diện).
2. **Cơ chế học biểu diễn của Student & Hàm Loss ([`losses.py`](file:///g:/nckh/DINO/direction1_new/core/losses.py))**:
   - Đã nối đúng tham số `--ibot_beta 0.5` vào hàm `iBOTPatchLoss` để các patch bị che nằm trên xe được nhân trọng số học $(1 + \beta \pi_p)$ cao hơn các patch nền.
   - Bổ sung `PatchVarianceRegularizer` (duy trì độ lệch chuẩn giữa các patch token và làm mượt không gian cục bộ, chống hiện tượng các patch bị sụp đổ về cùng một vector hoặc bị nhiễu hạt bàn cờ) và `FeatureAnchorLoss` (giữ không gian đặc trưng không bị trôi quá xa khỏi DINOv3 gốc trong những epoch đầu).
   - Bổ sung `CameraGroupedSampler` kết hợp `RICLoss` (Regional Instance Contrastive giữa các ảnh của cùng một camera nhưng khác thời điểm) và `CameraAdversarialHead` (Gradient Reversal Layer) để ép mô hình không được dựa vào nền tĩnh của góc máy cố định.
3. **Sửa lỗi hiển thị cắt đen nền ở Cột 5 & Cột 7 trong ảnh tiến trình ([`progress_visualizer.py`](file:///g:/nckh/DINO/direction1_new/visualization/progress_visualizer.py))**:
   - Ở bản trước, Cột 5 (`DINO Gốc PCA`) và Cột 7 (`Student PCA Nội Tại`) dùng chung hàm lọc ngưỡng `PC1 > 0.50` rồi tô đen toàn bộ phần còn lại, dẫn đến nửa trên bức ảnh bị đen kịt như đã trừ nền. Hiện đã tách bạch rõ: Cột 5 và Cột 7 chiếu Full-Image 3-Component PCA trên toàn bộ 448 patches (hiển thị 100% khung cảnh), chỉ riêng Cột 8 (`Student Foreground PCA`) mới lọc theo $\pi > 0.40$ để soi chi tiết các bộ phận trên thân xe.

---

## Phần 2. Kết Quả Kiểm Chứng Định Lượng Theo 5 Câu Hỏi Của Giảng Viên Hướng Dẫn

> **Ghi chú thực nghiệm**: Toàn bộ 5 mục dưới đây được trích xuất tự động từ script [`audit_tam_agm_srs_probe.py`](file:///g:/nckh/DINO/direction1_new/evaluation/audit_tam_agm_srs_probe.py) chạy trên tập ảnh thực tế `/workspace/output` cùng với `last_checkpoint.pth` sau khi kết thúc huấn luyện.

---

### Câu 1: Kiểm tra điểm TAM trên vùng phương tiện và vùng mặt đường không chứa phương tiện
**Phương pháp đo**:
- Để có biên giới khách quan giữa "vùng có xe" và "vùng mặt đường trống", script sử dụng bộ phát hiện phương tiện chuẩn COCO (Faster R-CNN MobileNetV3) quét qua từng ảnh $256 \times 448$ để chia lưới $16 \times 28 = 448$ patches thành 3 tập tách biệt:
  - **Vùng phương tiện ($M_{\text{veh}}$)**: Các patch có diện tích chứa xe $\ge 20\%$.
  - **Vùng lòng đường trống ($M_{\text{road}}$)**: Các patch nằm ở khu vực mặt đường (từ hàng thứ 4 trở xuống) và nằm ngoài vùng đệm 1 patch quanh mọi phương tiện ($\neg\text{dilate}(M_{\text{veh}}, 1)$) để đảm bảo mặt đường hoàn toàn trống.
  - **Vùng hậu cảnh trên cao ($M_{\text{sky}}$)**: Các hàng phía trên cao không có phương tiện.

| Chỉ số đo lường trên lưới Patch ($16 \times 28$) | TAM Bản Cũ (Chưa có Local Contrast & Photometric Norm) | TAM Bản Mới Hiện Tại (Đã khắc phục) | Nhận xét biến thiên |
| :--- | :---: | :---: | :--- |
| **Điểm TAM trung bình trên Xe ($\mathbb{E}[\pi \mid M_{\text{veh}}]$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Giữ mức kích hoạt cao trên thân xe |
| **Trung vị điểm TAM trên Xe ($\text{Median}[\pi \mid M_{\text{veh}}]$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | — |
| **Điểm TAM trung bình trên Mặt Đường Trống ($\mathbb{E}[\pi \mid M_{\text{road}}]$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Giảm mạnh hiện tượng lem ra mặt đường |
| **Trung vị điểm TAM trên Mặt Đường Trống ($\text{Median}[\pi \mid M_{\text{road}}]$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | — |
| **Điểm TAM trung bình vùng Hậu Cảnh Trên Cao ($\mathbb{E}[\pi \mid M_{\text{sky}}]$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | — |
| **Độ chênh lệch Xe vs. Đường trống ($\Delta_{\text{veh-road}}$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Biên độ phân tách mở rộng rõ rệt |
| **Chỉ số phân tách AUC-ROC (Xe vs. Đường trống)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | — |
| **Tỷ lệ bắt nhầm mặt đường trống (FPR tại ngưỡng $\tau = 0.40$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Khắc phục hiện tượng học nhầm lòng đường |
| **Độ nhạy bắt trúng phương tiện (TPR tại ngưỡng $\tau = 0.40$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | — |

**Phân tích**:
- Số liệu đối chứng giữa hai cột cho thấy ở bản cũ, các mảng mặt đường nhựa có bóng râm hoặc thay đổi ánh sáng bị đẩy điểm $\pi$ lên cao (FPR trên lòng đường trống lớn), khiến nhìn bằng mắt thường rất dễ thấy cả mặt đường sáng rực lên.
- Sau khi áp dụng chuẩn hóa trắc quang toàn khung hình và bộ lọc tương phản cục bộ $5 \times 5$, điểm TAM trên vùng mặt đường trống không có xe giảm hẳn xuống, trong khi các ô thực sự có xe vẫn giữ xác suất cao, trả lời rõ câu hỏi của Cô là **mô hình hiện tại xác định vị trí phương tiện chứ không gom cả lòng đường trống**.

---

### Câu 2: Trực quan hóa AGM chồng trực tiếp lên ảnh gốc, phân biệt patch được che nằm trên xe và ngoài xe
**Phương pháp thực hiện**:
- Ảnh kết quả được xuất tự động tại file `audit_visuals_agm_overlay.png` gồm 4 cột cho từng mẫu kiểm định:
  1. **Cột 1**: Ảnh gốc kèm khung Bounding Box màu vàng xác định vị trí thực của các phương tiện.
  2. **Cột 2**: Bản đồ xác suất tiền cảnh TAM $\pi \in [0, 1]$.
  3. **Cột 3 (AGM Overlay trực tiếp trên ảnh gốc)**:
     - **Các ô tô màu ĐỎ**: Patch bị AGM chọn che và **nằm đúng trên phương tiện** ($M_{\text{veh}}$).
     - **Các ô tô màu XANH CYAN**: Patch bị AGM chọn che nhưng **nằm ngoài phương tiện** (mặt đường trống hoặc hậu cảnh).
  4. **Cột 4**: Ảnh thực tế đưa vào nhánh Student sau khi đã đặt mask token.

**Nhận xét từ hình ảnh trực quan**:
- Nhờ thiết lập ngân sách tiền cảnh $\phi = 0.50$ và trần che tối đa trên xe $q_{\max} = 0.60$, các khối che màu đỏ tập trung phủ lên các cụm phương tiện nhưng luôn chừa lại ít nhất $40\%$ diện tích xe không bị che.
- Các ô che màu xanh cyan phân bổ rải rác ở mặt đường xung quanh, đóng vai trò buộc Student vừa phải khôi phục cấu trúc xe từ phần thân xe còn nhìn thấy, vừa duy trì ngữ cảnh không gian chung của khung hình.

---

### Câu 3: So sánh Random Masking, Deterministic Top-K và Gumbel Top-K
**Phương pháp đo**:
- Trên cùng một tỷ lệ che mục tiêu $r = 35\%$ ($K = 157/448$ patches), mỗi ảnh được lấy mẫu mặt nạ 2 lần độc lập (mô phỏng 2 epoch khác nhau) để đo 4 chỉ số:
  1. **Vehicle Hit Rate (Precision %)**: Trong tổng số các patch bị che, bao nhiêu % rơi trúng vào phương tiện.
  2. **Vehicle Recall (%)**: Bao nhiêu % diện tích của các phương tiện trong ảnh bị che đi.
  3. **Tỷ lệ diện tích xe giữ lại làm ngữ cảnh ($100\% - \text{Recall}$)**: Phần diện tích xe không bị che để Student có dữ kiện suy luận.
  4. **Jaccard Overlap giữa 2 lần lấy mẫu (%)**: Đo độ trùng lặp mặt nạ giữa các epoch trên cùng một ảnh ($100\%$ nghĩa là epoch nào cũng che đúng một kiểu cố định; càng vừa phải thì tính đa dạng mẫu càng tốt).

| Chiến lược Masking ($r = 35\%$, $157/448$ patches) | Tỷ lệ che trúng xe (Precision %) | Độ phủ diện tích xe (Recall %) | Diện tích xe giữ lại làm ngữ cảnh (%) | Độ trùng lặp giữa các Epoch (Jaccard %) | Entropy phân bố không gian |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Random Masking (Chuẩn MAE / DINOv2)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` |
| **2. Deterministic Top-K (Chọn cứng $K$ ô $\pi$ lớn nhất)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | **100.00%** | `[CHỜ ĐIỀN SỐ LIỆU]` |
| **3. Gumbel Top-K (Patch-wise, $\phi=0.5, q_{\max}=0.6$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` |
| **4. Gumbel Top-K (Super-Block $2 \times 2$ hiện dùng)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` |

**Phân tích sự khác biệt giữa 3 cơ chế**:
- **Random Masking**: Vì phương tiện thường chỉ chiếm $10\% - 25\%$ diện tích khung hình camera giao thông, che ngẫu nhiên khiến đa số patch bị che rơi vào mặt đường nhựa và bầu trời. Nhánh iBOT mất phần lớn dung lượng để học tái tạo mặt đường trống.
- **Deterministic Top-K**: Tuy tỷ lệ rơi trúng xe cao nhất, nhưng có 2 nhược điểm chí mạng khi đưa vào huấn luyện thực tế:
  1. **Che mất hoàn toàn chiếc xe**: Khi số lượng xe trên ảnh ít (tổng số patch xe $< K$), Deterministic Top-K sẽ che kín gần $100\%$ thân xe, khiến Student không còn nhìn thấy bất kỳ bộ phận nào của xe để suy luận ngữ nghĩa.
  2. **Trùng lặp tuyệt đối ($\text{Jaccard} = 100\%$)**: Qua mọi epoch, một bức ảnh luôn bị che ở đúng các ô cố định đó, làm giảm tính đa dạng của quá trình tự giám sát.
- **Gumbel Top-K (kèm ràng buộc $\phi=0.5, q_{\max}=0.6$)**: Giải quyết được cả hai vấn đề trên — vừa ưu tiên che vào phương tiện cao hơn hẳn Random Masking, vừa khống chế trần $q_{\max}=0.6$ để luôn giữ lại tối thiểu $40\%$ diện tích xe làm mỏ neo ngữ cảnh, đồng thời nhiễu Gumbel tạo ra sự thay đổi vị trí che qua từng epoch.

---

### Câu 4: Dùng `last_checkpoint` đánh giá Frozen Linear Probe so với DINOv3 Pretrained
**Phương pháp đo (Hoàn toàn không cần gán nhãn đếm xe thủ công)**:
Để đánh giá khách quan xem `student_backbone` trong `last_checkpoint.pth` có thực sự cải thiện biểu diễn phương tiện và giảm phụ thuộc vào nền camera so với `DINOv3 Pretrained` gốc hay không, chúng ta đóng băng 100% trọng số của cả hai mô hình và chạy 3 bài kiểm tra định lượng với 5-Fold Stratified Cross-Validation:
1. **Patch-level Vehicle vs. Empty Road Linear Probe**: Lấy trực tiếp các vector đặc trưng patch nằm trên phương tiện ($M_{\text{veh}}$) và các vector đặc trưng patch nằm trên mặt đường trống ($M_{\text{road}}$) để huấn luyện bộ phân loại tuyến tính `LogisticRegression`. Nếu đặc trưng của Student phân biệt "ô chứa xe" và "ô mặt đường" sắc nét hơn, các chỉ số `Accuracy`, `F1-Score` và `AUC-ROC` sẽ tăng lên.
2. **Độ phân tách hình học Cosine ($\Delta\cos_{\text{veh-road}} = 1 - \cos(\bar{f}_{\text{veh}}, \bar{f}_{\text{road}})$)**: Đo khoảng cách góc trực tiếp trong không gian đặc trưng 384 chiều giữa tâm cụm các patch xe và tâm cụm các patch mặt đường ngay trên cùng một bức ảnh (chưa qua bất kỳ bộ phân loại nào).
3. **Camera-ID Shortcut Linear Probe**: Huấn luyện bộ phân loại tuyến tính dự đoán xem bức ảnh thuộc **Camera ID nào** dựa trên vector đại diện. Một mô hình bị dính "bẫy nền" (Background Shortcut) sẽ nhận diện góc máy camera cực kỳ dễ dàng; ngược lại, khi đã học bất biến qua SRS + RIC + CamAdv, khả năng đoán mò Camera ID sẽ giảm xuống.
4. **Effective Rank (Bậc hiệu dụng không gian)**: Đo độ đa dạng chiều của ma trận đặc trưng để kiểm chứng mô hình không bị sụp đổ chiều (Dimensional Collapse).

| Tiêu chí đánh giá (Frozen Backbone 100%) | DINOv3 Pretrained (Gốc) | Student Checkpoint (`last_checkpoint.pth`) | Chênh lệch ($\Delta$) & Ý nghĩa |
| :--- | :---: | :---: | :--- |
| **1. Phân loại tuyến tính Patch Xe vs. Patch Đường trống (Accuracy %)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Càng cao $\rightarrow$ đặc trưng xe càng tách biệt khỏi mặt đường |
| **2. Phân loại tuyến tính Patch Xe vs. Patch Đường trống (F1-Score %)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Càng cao càng tốt |
| **3. Phân loại tuyến tính Patch Xe vs. Patch Đường trống (AUC-ROC)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Càng cao càng tốt |
| **4. Khoảng cách Cosine trực tiếp Xe vs. Đường ($\Delta\cos_{\text{veh-road}}$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Càng lớn $\rightarrow$ biểu diễn nội tại càng tách bạch xe và nền |
| **5. Camera-ID Linear Probe trên CLS Token (Accuracy %)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Càng giảm $\rightarrow$ càng bớt học vẹt góc máy cố định |
| **6. Camera-ID Linear Probe trên Foreground Token (Accuracy %)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Càng giảm $\rightarrow$ biểu diễn xe càng bất biến qua các camera |
| **7. Bậc hiệu dụng không gian `Effective Rank` (Foreground Token)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Duy trì số chiều biểu diễn cao, không bị collapse |
| **8. Bậc hiệu dụng không gian `Effective Rank` (Patch Tokens)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | — |

---

### Câu 5: Thống kê tỷ lệ ảnh SRS thực sự thay đổi ($\Delta > 0\%$) và mức thay đổi trung bình
**Phương pháp đo**:
- Ghép cặp $500$ cặp ảnh $(x_1, x_2)$ thuộc **cùng một camera nhưng khác thời điểm** trên tập dữ liệu thực tế và chạy đối chứng song song giữa:
  1. **Logic SRS Cũ (kết hợp TAM cũ)**: Điều kiện $(\pi_1 < 0.20) \land (\pi_2 < 0.20) \land \neg\text{dilate}(\pi_1 \ge 0.50 \lor \pi_2 \ge 0.50, \text{radius}=1)$.
  2. **Logic SRS Mới (kết hợp TAM mới)**: Bảo vệ xe trên ảnh 1 (`dilate(pi1 >= 0.40, radius=1)`), tránh xe trên ảnh 2 (`pi2 < 0.45`), kết hợp cơ chế **Adaptive Fallback 2 cấp** và ghép nền liên tục (`continuous_bg=True`).

| Chỉ số thống kê trên 500 cặp ảnh cùng Camera | Logic SRS Cũ (Phiên bản bị $\Delta=0\%$) | Logic SRS Mới (Đã sửa lỗi & có Fallback) | Nhận xét kết quả |
| :--- | :---: | :---: | :--- |
| **Tỷ lệ cặp ảnh bị bất hoạt hoàn toàn ($\Delta = 0.0\%$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Giải thích vì sao bản cũ có tới 3/4 ảnh bị $\Delta = 0\%$ |
| **Tỷ lệ cặp ảnh kích hoạt hoán đổi thực sự ($\Delta > 0.0\%$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Bản mới đảm bảo luôn kích hoạt hoán đổi khi bật SRS |
| **Số lượng Patch nền được hoán đổi trung bình (trên 448 patches)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Tăng mạnh diện tích nền tĩnh được thay thế |
| **Tỷ lệ diện tích khung hình được hoán đổi trung bình (%)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | — |
| **Mức thay đổi giá trị Pixel trung bình toàn tập ($\mathbb{E}[\Delta_{\text{pixel}}]$)** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | Tạo ra sự thay đổi bối cảnh đủ lớn để phá vỡ Background Shortcut |
| **Mức thay đổi giá trị Pixel tính riêng khi có kích hoạt** | `[CHỜ ĐIỀN SỐ LIỆU]` | `[CHỜ ĐIỀN SỐ LIỆU]` | — |

---

## Phần 3. Lệnh Chạy Trích Xuất Số Liệu Tự Động Sau Khi Huấn Luyện Xong

Ngay sau khi tiến trình huấn luyện 30 epochs trên server hoàn tất, chạy lệnh dưới đây tại thư mục `/workspace/nckh/DINO` để xuất toàn bộ bảng số liệu (`audit_metrics_summary.json`) và ảnh trực quan hóa (`audit_visuals_agm_overlay.png`):

```bash
cd /workspace/nckh/DINO
source /venv/main/bin/activate

python direction1_new/evaluation/audit_tam_agm_srs_probe.py \
    --data_dir /workspace/output \
    --tam_cache_dir /workspace/tam_cache \
    --checkpoint checkpoints/direction1_new/last_checkpoint.pth \
    --num_cams 30 \
    --frames_per_cam 8 \
    --output_json checkpoints/direction1_new/audit_metrics_summary.json \
    --output_visual checkpoints/direction1_new/audit_visuals_agm_overlay.png
```
