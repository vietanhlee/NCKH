# Chi Tiết Thuật Toán Context-Aware Markov Weak Supervision (CAM-WS)
> **Tài liệu Kỹ thuật Chuyên sâu — Hướng 3: Weak Supervision cho Giám sát Giao thông Đô thị**  
> *Đặc tả Toán học, Chuỗi Markov, Hàm Likelihood, Thuật toán Forward-Backward, Tối ưu hóa EM và Cài đặt Production.*

---

## MỤC LỤC
1. [Thuật Toán Con 2: Mã hóa Không gian Ngữ cảnh Đa biến (54 Ngữ cảnh)](#1-thuật-toán-con-2-mã-hóa-không-gian-ngữ-cảnh-đa-biến-54-ngữ-cảnh)
2. [Cơ Sở Lý Thuyết Cốt Lõi: Chuỗi Markov & Hàm Likelihood](#2-cơ-sở-lý-thuyết-cốt-lõi-chuỗi-markov--hàm-likelihood)
   - [2.1. Động học Markov Cấp 1 (First-Order Markovian Dynamics)](#21-động-học-markov-cấp-1-first-order-markovian-dynamics)
   - [2.2. Ma trận Chuyển trạng thái A và Ý nghĩa Vật lý Đô thị](#22-ma-trận-chuyển-trạng-thái-a-và-ý-nghĩa-vật-lý-đô-thị)
   - [2.3. Hàm Likelihood & Marginal Likelihood (Hàm Hợp Lý Tích Lũy)](#23-hàm-likelihood--marginal-likelihood-hàm-hợp-lý-tích-lũy)
   - [2.4. Nghịch lý Tối ưu hóa Likelihood Trực tiếp](#24-nghịch-lý-tối-ưu-hóa-likelihood-trực-tiếp)
3. [Thuật Toán Con 3: Thuật Toán Forward-Backward trong Không Gian Log-Sum-Exp](#3-thuật-toán-con-3-thuật-toán-forward-backward-trong-không-gian-log-sum-exp)
   - [3.1. Định nghĩa Biến Forward & Backward](#31-định-nghĩa-biến-forward--backward)
   - [3.2. Cơ chế Log-Sum-Exp Chống Tràn số Dưới (Numerical Underflow)](#32-cơ-chế-log-sum-exp-chống-tràn-số-dưới-numerical-underflow)
   - [3.3. Tính toán Xác suất Hậu nghiệm Mềm $\gamma_t$ và $\xi_t$](#33-tính-toán-xác-suất-hậu-nghiệm-mềm-gamma_t-và-xi_t)
4. [Thuật Toán Con 4: Tối Ưu Hóa EM (Expectation-Maximization)](#4-thuật-toán-con-4-tối-ưu-hóa-em-expectation-maximization)
   - [4.1. Bản chất Toán học: Bất đẳng thức Jensen và Cận Dưới ELBO](#41-bản-chất-toán-học-bất-đẳng-thức-jensen-và-cận-dưới-elbo)
   - [4.2. E-step (Expectation Step)](#42-e-step-expectation-step)
   - [4.3. M-step (Maximization Step) — Nghiệm Giải Tích Đóng](#43-m-step-maximization-step--nghiệm-giải-tích-đóng)
   - [4.4. Cơ chế Làm trơn Laplace (Laplace Smoothing)](#44-cơ-chế-làm-trơn-laplace-laplace-smoothing)
5. [Thuật Toán Con 5: Chưng Cất End-Model (DINOv3 + Causal GRU)](#5-thuật-toán-con-5-chưng-cất-end-model-dinov3--causal-gru)
   - [5.1. Kiến trúc Causal 1-Way Recurrent Network](#51-kiến-trúc-causal-1-way-recurrent-network)
   - [5.2. Hàm Mất Mát Soft Cross-Entropy Có Ngưỡng Tự Tin](#52-hàm-mất-mát-soft-cross-entropy-có-ngưỡng-tự-tin)
6. [Đối Chiếu Trực Tiếp Từng Dòng Mã Nguồn Production (`label_model.py`)](#6-đối-chiếu-trực-tiếp-từng-dòng-mã-nguồn-production-label_modelpy)

---

## 1. Thuật Toán Con 2: Mã hóa Không gian Ngữ cảnh Đa biến (54 Ngữ cảnh)
> 📂 Mã nguồn: [`context.py`](file:///g:/nckh/DINO/direction3_weak_supervision/context.py)

Trong bài toán giám sát giao thông quy mô thành phố, **không có bất kỳ một nguồn nhãn yếu (LF) nào chính xác tuyệt đối trên mọi môi trường**:
- Thuật toán trích xuất biên nền (LF2) tê liệt khi camera rung lắc hoặc ban đêm lóa đèn pha.
- Mô hình nhận diện đối tượng (LF1) bỏ sót xe khi camera chuyển sang hồng ngoại (IR) hoặc bị che khuất tầm nhìn.
- Mô hình ngôn ngữ thị giác (LF5 - VLM) suy giảm độ chính xác khi ảnh mờ hạt trong mưa.

Do đó, mô hình nhãn yếu bắt buộc phải **phân tách độ tin cậy theo ngữ cảnh (Context-Conditioned Reliability)**.

### Không gian phân rã 54 trạng thái
Không gian ngữ cảnh được rời rạc hóa thành tích Descartes của 4 biến độc lập:

$$\mathcal{C} = \mathcal{X}_{\text{IR}} \times \mathcal{X}_{\text{Hour}} \times \mathcal{X}_{\text{Road}} \times \mathcal{X}_{\text{Density}}$$

1. **Biến quang học ánh sáng $\mathcal{X}_{\text{IR}} \in \{0, 1\}$ (2 mức):**
   - $0$: Ban ngày / Chế độ màu RGB.
   - $1$: Ban đêm / Chế độ cảm biến hồng ngoại đơn sắc (Infrared - IR).
2. **Biến khung giờ sinh hoạt $\mathcal{X}_{\text{Hour}} \in \{0, 1, 2\}$ (3 mức):**
   - $0$: Đêm muộn ($22\text{h}00 - 05\text{h}00$).
   - $1$: Thấp điểm ban ngày ($09\text{h}00 - 16\text{h}00$ và $20\text{h}00 - 21\text{h}00$).
   - $2$: Giờ cao điểm giao thông ($06\text{h}00 - 08\text{h}00$ và $17\text{h}00 - 19\text{h}00$).
3. **Biến phân loại hạ tầng đường $\mathcal{X}_{\text{Road}} \in \{0, 1, 2\}$ (3 mức):**
   - $0$: Đại lộ / Trục xuyên tâm lớn (dung lượng xe cao).
   - $1$: Nút giao vòng xoay / Ngã tư trọng điểm.
   - $2$: Đường nhánh hẹp nội đô.
4. **Biến mật độ quan trắc thô $\mathcal{X}_{\text{Density}} \in \{0, 1, 2\}$ (3 mức):**
   - $0$: Thưa thớt.
   - $1$: Trung bình.
   - $2$: Dày đặc.

### Công thức ánh xạ phẳng (Flat Context Indexing)
Chỉ số ngữ cảnh duy nhất $c_t \in \{0, 1, \dots, 53\}$ được tính bằng công thức giải tích trực tiếp:

$$c_t = \text{is\_ir} \times 27 + \text{hour\_group} \times 9 + \text{road\_type} \times 3 + \text{density\_bin}$$

Công thức này tạo ra một bảng tra cứu tức thời $\mathcal{O}(1)$ giúp ma trận nhầm lẫn của mỗi LF được học độc lập trên từng "lát cắt môi trường" cụ thể.

---

## 2. Cơ Sở Lý Thuyết Cốt Lõi: Chuỗi Markov & Hàm Likelihood

### 2.1. Động học Markov Cấp 1 (First-Order Markovian Dynamics)
Gọi $y_t \in \{0, 1, 2, 3\}$ là **trạng thái ùn tắc thực tế (ẩn số)** tại thời điểm $t$:
- $0$: Thông thoáng (Free-flow)
- $1$: Đông nhẹ (Moderate)
- $2$: Ùn ứ (Slow/Heavy)
- $3$: Kẹt cứng nghiêm trọng (Jammed)

Trạng thái giao thông vật lý **có tính liên tục không gian - thời gian mạnh mẽ**: Dòng xe cộ không thể tức thời biến mất hoặc tắc nghẽn đột ngột từ 0 nhảy thẳng lên 3 chỉ sau vài phút.

#### Giả thuyết Markov Cấp 1 (First-Order Markov Property):
Xác suất trạng thái tại frame $t$ chỉ phụ thuộc duy nhất vào trạng thái ngay trước nó tại frame $t-1$, hoàn toàn độc lập với toàn bộ lịch sử xa hơn:

$$P(y_t \mid y_{t-1}, y_{t-2}, \dots, y_1) = P(y_t \mid y_{t-1})$$

### 2.2. Ma trận Chuyển trạng thái A và Ý nghĩa Vật lý Đô thị
Ma trận chuyển trạng thái $\mathbf{A} \in \mathbb{R}^{K \times K}$ ($K = 4$) định nghĩa xác suất chuyển tiếp giữa các mức độ kẹt xe:

$$A_{ij} = P(y_t = j \mid y_{t-1} = i), \quad \text{với } \sum_{j=0}^{K-1} A_{ij} = 1, \; \forall i$$

#### Ý nghĩa vật lý và Cấu trúc ràng buộc:
1. **Quán tính dòng chảy (Diagonal Dominance):**
   Thời gian lấy mẫu giữa hai frame camera là $\Delta t \approx 5$ phút. Trong $5$ phút, trạng thái giao thông thường có xu hướng giữ nguyên:
   $$A_{ii} \gg A_{ij} \quad (j \neq i)$$
   Trong khởi tạo (warm-start), đường chéo được đặt $A_{ii} = 0.85$.
2. **Chuyển tiếp lân cận (Band-diagonal structure):**
   Giao thông thay đổi tuần tự: $0 \leftrightarrow 1 \leftrightarrow 2 \leftrightarrow 3$. Xác suất nhảy vọt $0 \to 3$ hoặc $3 \to 0$ trong 1 bước thời gian có giá trị rất nhỏ:
   $$A_{03} \approx 0, \quad A_{30} \approx 0$$
   Điều này đóng vai trò như một **bộ lọc thông thấp thời gian (temporal low-pass filter)**, ngăn chặn việc nhãn bị nhảy giật cục do nhiễu tức thời của camera.

---

### 2.3. Hàm Likelihood & Marginal Likelihood (Hàm Hợp Lý Tích Lũy)

Trong thống kê, **Likelihood (Độ hợp lý)** đo lường: *"Nếu giả sử mô hình có bộ tham số $\theta$, thì xác suất sinh ra đúng tập dữ liệu mà ta đã quan sát được là bao nhiêu?"*

Bộ tham số của mô hình gồm:
$$\theta = \Big\{ \boldsymbol{\pi}, \; \mathbf{A}, \; \mathbf{E} \Big\}$$
Trong đó:
- $\boldsymbol{\pi} \in \mathbb{R}^K$: Phân bố xác suất tiên nghiệm ban đầu $P(y_1 = k)$.
- $\mathbf{A} \in \mathbb{R}^{K \times K}$: Ma trận chuyển trạng thái Markov $P(y_t = j \mid y_{t-1} = i)$.
- $\mathbf{E} \in \mathbb{R}^{J \times C \times K \times K}$: Ma trận phát xạ quan sát (Emission / Confusion matrix) phụ thuộc ngữ cảnh:
  $$\mathbf{E}_{j, c, k, v} = P(\lambda_{t, j} = v \mid y_t = k, c_t = c)$$
  với $j \in \{1 \dots J\}$ là LF thứ $j$, $v \in \{0 \dots K-1\}$ là nhãn mà LF đưa ra.

#### 1. Hàm Hợp Lý Đầy Đủ (Complete-Data Likelihood):
Nếu ta **biết trước toàn bộ chuỗi nhãn thật $\mathbf{y}_{1:T} = (y_1, y_2, \dots, y_T)$**, thì xác suất đồng thời sinh ra cả nhãn thật $\mathbf{y}_{1:T}$ và các quan sát của 5 LFs $\boldsymbol{\lambda}_{1:T}$ là:

$$P(\boldsymbol{\lambda}_{1:T}, \mathbf{y}_{1:T} \mid \theta) = P(y_1 \mid \boldsymbol{\pi}) \prod_{t=2}^{T} P(y_t \mid y_{t-1}, \mathbf{A}) \prod_{t=1}^{T} \prod_{j=1}^{J} P(\lambda_{t, j} \mid y_t, c_t, \mathbf{E})$$

Chuyển sang không gian Log (Complete-data Log-Likelihood):

$$\log P(\boldsymbol{\lambda}_{1:T}, \mathbf{y}_{1:T} \mid \theta) = \log \pi_{y_1} + \sum_{t=2}^T \log A_{y_{t-1}, y_t} + \sum_{t=1}^T \sum_{j: \lambda_{t,j} \ge 0} \log \mathbf{E}_{j, c_t, y_t, \lambda_{t,j}}$$

#### 2. Hàm Hợp Lý Biên Quan Sát (Marginal / Incomplete-Data Likelihood):
Thực tế ta **hoàn toàn không có nhãn thật $\mathbf{y}_{1:T}$ (biến ẩn)**! Ta chỉ quan sát được duy nhất $\boldsymbol{\lambda}_{1:T}$.
Do đó, độ hợp lý của dữ liệu quan sát được tính bằng cách **cộng gộp (marginalize) trên toàn bộ mọi chuỗi trạng thái ẩn khả dĩ**:

$$P(\boldsymbol{\lambda}_{1:T} \mid \theta) = \sum_{y_1=0}^{K-1} \sum_{y_2=0}^{K-1} \dots \sum_{y_T=0}^{K-1} P(\boldsymbol{\lambda}_{1:T}, y_1, y_2, \dots, y_T \mid \theta)$$

Mục tiêu học máy là tìm bộ tham số $\theta^*$ sao cho dữ liệu quan sát có độ hợp lý cao nhất:

$$\theta^* = \arg\max_\theta \log P(\boldsymbol{\lambda}_{1:T} \mid \theta) = \arg\max_\theta \log \left( \sum_{\mathbf{y}_{1:T}} P(\boldsymbol{\lambda}_{1:T}, \mathbf{y}_{1:T} \mid \theta) \right)$$

---

### 2.4. Nghịch lý Tối ưu hóa Likelihood Trực tiếp

Tại sao không thể giải bài toán $\max_\theta \log P(\boldsymbol{\lambda}_{1:T} \mid \theta)$ bằng đạo hàm thông thường?
1. **Bùng nổ tổ hợp cấp lũy thừa $\mathcal{O}(K^T)$:**
   Một chuỗi video gồm $T = 300$ frame, mỗi frame có $K = 4$ trạng thái $\implies$ Số lượng chuỗi $\mathbf{y}_{1:T}$ là $4^{300} \approx 10^{180}$ (lớn hơn số lượng nguyên tử trong vũ trụ). Không thể tính tổng trực tiếp.
2. **Kẹt toán tử Log bên ngoài Dấu tổng ($\log \sum$):**
   Trong giải tích, $\log(a + b) \neq \log a + \log b$. Dấu $\log$ bao ngoài dấu $\sum$ khiến các tham số bị liên kết phi tuyến phức tạp, triệt tiêu mọi nghiệm giải tích đóng.

👉 **Giải pháp hoàn hảo gồm 2 vũ khí:**
- Dùng **Thuật toán Forward-Backward** để rút gọn độ phức tạp từ $\mathcal{O}(K^T)$ xuống $\mathcal{O}(T \cdot K^2)$ bằng quy hoạch động.
- Dùng **Thuật toán EM (Expectation-Maximization)** để hóa giải dấu $\log \sum$ thông qua tối ưu hóa cận dưới biến phân (ELBO).

---

## 3. Thuật Toán Con 3: Thuật Toán Forward-Backward trong Không Gian Log-Sum-Exp
> 📂 Mã nguồn: [`label_model.py#L83-L144`](file:///g:/nckh/DINO/direction3_weak_supervision/label_model.py#L83-L144)

Thuật toán Forward-Backward sử dụng quy hoạch động để tính toán chính xác xác suất phân bố trạng thái tại từng thời điểm $t$ mà không cần duyệt vét cạn $K^T$ đường đi.

### 3.1. Định nghĩa Biến Forward & Backward

#### 1. Biến Forward $\alpha_t(j)$:
Là xác suất đồng thời của việc quan sát chuỗi từ đầu đến thời điểm $t$ và hệ thống đang ở trạng thái $j$ tại thời điểm $t$:
$$\alpha_t(j) = P(\boldsymbol{\lambda}_1, \boldsymbol{\lambda}_2, \dots, \boldsymbol{\lambda}_t, \; y_t = j \mid \theta)$$

- **Khởi tạo ($t = 1$):**
  $$\alpha_1(j) = \pi_j \cdot P(\boldsymbol{\lambda}_1 \mid y_1 = j, c_1)$$
- **Đệ quy tiến ($t = 2 \dots T$):**
  $$\alpha_t(j) = \left[ \sum_{i=0}^{K-1} \alpha_{t-1}(i) \cdot A_{ij} \right] \cdot P(\boldsymbol{\lambda}_t \mid y_t = j, c_t)$$

#### 2. Biến Backward $\beta_t(i)$:
Là xác suất có điều kiện của việc quan sát phần còn lại của chuỗi từ $t+1$ đến $T$, biết rằng tại thời điểm $t$ hệ thống đang ở trạng thái $i$:
$$\beta_t(i) = P(\boldsymbol{\lambda}_{t+1}, \dots, \boldsymbol{\lambda}_T \mid y_t = i, \theta)$$

- **Khởi tạo ($t = T$):**
  $$\beta_T(i) = 1.0, \quad \forall i$$
- **Đệ quy lùi ($t = T-1 \dots 1$):**
  $$\beta_t(i) = \sum_{j=0}^{K-1} A_{ij} \cdot P(\boldsymbol{\lambda}_{t+1} \mid y_{t+1} = j, c_{t+1}) \cdot \beta_{t+1}(j)$$

---

### 3.2. Cơ chế Log-Sum-Exp Chống Tràn số Dưới (Numerical Underflow)

Trong máy tính, tích của hàng trăm xác suất nhỏ hơn 1 ($\prod_{t=1}^{300} 0.1$) sẽ dẫn tới $10^{-300}$, tràn giới hạn phần cứng `float64` về đúng số $0$ (Underflow).
Để loại trừ hoàn toàn lỗi này, toàn bộ thuật toán được chuyển sang không gian Log:

Đặt $\log \alpha_t(j)$, $\log \beta_t(i)$, $\log A_{ij}$, $\log \text{emission}$.

Khi cần tính tổng trong không gian log:
$$\log \left( \sum_{i} \exp(v_i) \right)$$
Ta áp dụng đẳng thức biến đổi **Log-Sum-Exp Trick**:

$$m = \max_i(v_i)$$
$$\log \left( \sum_{i} \exp(v_i) \right) = m + \log \left( \sum_{i} \exp(v_i - m) \right)$$

Do đã trừ đi giá trị cực đại $m$, số mũ lớn nhất luôn bằng $e^0 = 1$, các số còn lại nằm trong khoảng $(0, 1]$, triệt tiêu 100% nguy cơ tràn số!

Đoạn mã thực thi trong `label_model.py`:
```python
# Forward Pass trong log space (dòng 105-108)
prev_terms = log_alpha[t - 1] + log_A[:, j]
m = np.max(prev_terms)
log_alpha[t, j] = m + np.log(np.sum(np.exp(prev_terms - m))) + log_emission[t, j]
```

---

### 3.3. Tính toán Xác suất Hậu nghiệm Mềm $\gamma_t$ và $\xi_t$

Sau khi chạy xong hai chiều Forward và Backward:

#### 1. Xác suất Trạng thái Đơn lẻ $\gamma_t(k)$ (Posterior Soft Labels):
Xác suất trạng thái thật tại thời điểm $t$ là mức kẹt xe $k$, dựa trên toàn bộ quan sát của chuỗi từ $1 \to T$:

$$\gamma_t(k) = P(y_t = k \mid \boldsymbol{\lambda}_{1:T}, \theta) = \frac{\alpha_t(k) \beta_t(k)}{P(\boldsymbol{\lambda}_{1:T} \mid \theta)} = \frac{\alpha_t(k) \beta_t(k)}{\sum_{l=0}^{K-1} \alpha_t(l) \beta_t(l)}$$

Trong không gian log:
$$\log \gamma_t(k) = \log \alpha_t(k) + \log \beta_t(k) - \text{LogSumExp}_l(\log \alpha_t(l) + \log \beta_t(l))$$

Vector $\gamma_t = [\gamma_t(0), \gamma_t(1), \gamma_t(2), \gamma_t(3)]$ chính là **Phân phối Nhãn mềm (Soft Label)** chân lý mà ta tìm kiếm!

#### 2. Xác suất Chuyển trạng thái Liên tiếp $\xi_t(i, j)$:
Xác suất tại $t$ là mức $i$ và tại $t+1$ chuyển sang mức $j$:

$$\xi_t(i, j) = P(y_t = i, y_{t+1} = j \mid \boldsymbol{\lambda}_{1:T}, \theta) = \frac{\alpha_t(i) \cdot A_{ij} \cdot P(\boldsymbol{\lambda}_{t+1} \mid y_{t+1}=j, c_{t+1}) \cdot \beta_{t+1}(j)}{\sum_{a} \sum_{b} \alpha_t(a) A_{ab} P(\boldsymbol{\lambda}_{t+1} \mid b, c_{t+1}) \beta_{t+1}(b)}$$

$\xi_t(i, j)$ được dùng để cập nhật lại ma trận chuyển trạng thái $\mathbf{A}$ trong bước M-step.

---

## 4. Thuật Toán Con 4: Tối Ưu Hóa EM (Expectation-Maximization)
> 📂 Mã nguồn: [`label_model.py#L146-L208`](file:///g:/nckh/DINO/direction3_weak_supervision/label_model.py#L146-L208)

Thuật toán EM là phương pháp tối ưu hóa phân rã bài toán Log-Likelihood phức tạp thành hai bước lặp hội tụ đơn giản.

### 4.1. Bản chất Toán học: Bất đẳng thức Jensen và Cận Dưới ELBO

Ta muốn cực đại hóa Log-Likelihood biên:
$$\ell(\theta) = \log P(\boldsymbol{\lambda}_{1:T} \mid \theta) = \log \sum_{\mathbf{y}} P(\boldsymbol{\lambda}_{1:T}, \mathbf{y} \mid \theta)$$

Chèn một phân phối bất kỳ $q(\mathbf{y})$ trên các biến ẩn:
$$\ell(\theta) = \log \sum_{\mathbf{y}} q(\mathbf{y}) \frac{P(\boldsymbol{\lambda}_{1:T}, \mathbf{y} \mid \theta)}{q(\mathbf{y})} = \log \mathbb{E}_{q} \left[ \frac{P(\boldsymbol{\lambda}_{1:T}, \mathbf{y} \mid \theta)}{q(\mathbf{y})} \right]$$

Vì hàm $\log$ là hàm lõm ($\cap$), áp dụng **Bất đẳng thức Jensen** ta đưa dấu $\log$ vào trong dấu kỳ vọng:

$$\ell(\theta) \ge \sum_{\mathbf{y}} q(\mathbf{y}) \log \frac{P(\boldsymbol{\lambda}_{1:T}, \mathbf{y} \mid \theta)}{q(\mathbf{y})} \equiv \text{ELBO}(q, \theta)$$

Biến đổi đại số:
$$\ell(\theta) = \text{ELBO}(q, \theta) + D_{\text{KL}}\Big( q(\mathbf{y}) \;\|\; P(\mathbf{y} \mid \boldsymbol{\lambda}_{1:T}, \theta) \Big)$$

Do khoảng cách Kullback-Leibler $D_{\text{KL}} \ge 0$, ta có:
1. **Bước E (Expectation):** Chọn $q(\mathbf{y}) = P(\mathbf{y} \mid \boldsymbol{\lambda}_{1:T}, \theta^{(m)})$. Khi đó $D_{\text{KL}} = 0 \implies \text{ELBO}$ chạm khít đúng giá trị $\ell(\theta^{(m)})$.
2. **Bước M (Maximization):** Cố định $q(\mathbf{y})$, tìm $\theta^{(m+1)}$ để tối đa hóa $\text{ELBO}$:
   $$\theta^{(m+1)} = \arg\max_\theta \sum_{\mathbf{y}} q(\mathbf{y}) \log P(\boldsymbol{\lambda}_{1:T}, \mathbf{y} \mid \theta)$$
   Điều này đảm bảo $\ell(\theta^{(m+1)}) \ge \ell(\theta^{(m)})$ — **Log-Likelihood luôn luôn tăng và không bao giờ suy giảm!**

---

### 4.2. E-step (Expectation Step)
Với bộ tham số $\theta^{(m)}$ hiện tại:
1. Chạy thuật toán Forward-Backward (Mục 3) trên tất cả các chuỗi video trong tập huấn luyện.
2. Trích xuất ma trận xác suất hậu nghiệm:
   - $\gamma_t(k) = P(y_t = k \mid \boldsymbol{\lambda}_{1:T}, \theta^{(m)})$
   - $\xi_t(i, j) = P(y_t = i, y_{t+1} = j \mid \boldsymbol{\lambda}_{1:T}, \theta^{(m)})$
3. Tính Log-Likelihood của chuỗi:
   $$\log P(\boldsymbol{\lambda}_{1:T}) = \text{LogSumExp}_k(\log \alpha_T(k))$$

---

### 4.3. M-step (Maximization Step) — Nghiệm Giải Tích Đóng
Trong bước M, nhờ dấu $\log$ đã nằm bên trong, hàm mục tiêu tách rời thành các thành phần độc lập. Ta dùng phương pháp nhân tử Lagrange ($\lambda$) để tìm cực trị giải tích:

#### 1. Cập nhật Phân bố Khởi đầu $\boldsymbol{\pi}$:
$$\pi_k^{(m+1)} = \frac{\sum_{s \in \text{Sequences}} \gamma_1^{(s)}(k)}{|\text{Sequences}|}$$

#### 2. Cập nhật Ma trận Chuyển trạng thái Markov $\mathbf{A}$:
Tỷ lệ giữa kỳ vọng số lần chuyển từ $i \to j$ chia cho kỳ vọng số lần rời khỏi $i$:

$$A_{ij}^{(m+1)} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{l=0}^{K-1} \sum_{t=1}^{T-1} \xi_t(i, l)}$$

#### 3. Cập nhật Ma trận Nhầm lẫn theo Ngữ cảnh $\mathbf{E}$ (Emission Matrix):
Xác suất LF $j$ đưa ra nhãn $v$ khi ở ngữ cảnh $c$ và trạng thái thực tế là $k$:

$$\mathbf{E}_{j, c, k, v}^{(m+1)} = \frac{\sum_{t: c_t = c, \lambda_{t, j} = v} \gamma_t(k)}{\sum_{t: c_t = c, \lambda_{t, j} \ge 0} \gamma_t(k)}$$

---

### 4.4. Cơ chế Làm trơn Laplace (Laplace Smoothing)
Trong thực tế, có những ngữ cảnh hiếm (ví dụ: bão lũ ban đêm) khiến một LF chưa bao giờ đưa ra nhãn $v$. Nếu để xác suất bằng $0$, khi tính $\log(0)$ sẽ bị lỗi $-\infty$.
Thuật toán bổ sung tham số làm trơn $\epsilon = 10^{-6}$ (dòng 198 `label_model.py`):

$$\mathbf{E}_{j, c, k, v}^{(m+1)} = \frac{\sum_{t: c_t = c, \lambda_{t, j} = v} \gamma_t(k) + \epsilon}{\sum_{t: c_t = c} \gamma_t(k) + K \cdot \epsilon}$$

Điều này đảm bảo ma trận xác suất luôn xác định dương và ổn định tuyệt đối về mặt số học.

---

## 5. Thuật Toán Con 5: Chưng Cất End-Model (DINOv3 + Causal GRU)
> 📂 Mã nguồn: [`end_model.py`](file:///g:/nckh/DINO/direction3_weak_supervision/end_model.py)

Mô hình nhãn yếu tầng 2 tạo ra nhãn mềm chất lượng cao $q(y_t) = \gamma_t$. Tuy nhiên, để hệ thống chạy được trên thực địa với hàng nghìn camera thời gian thực, ta không thể chạy đồng thời cả 5 LF (đặc biệt là VLM và Background subtraction rất tốn tài nguyên).

Ta chưng cất toàn bộ tri thức của Label Model vào một mạng nơ-ron duy nhất: **End Model**.

### 5.1. Kiến trúc Causal 1-Way Recurrent Network
```
Camera Frame I_t ──► [ DINOv3 ViT Backbone ] ──► z_t (384-d)
                                                    │
                                                    ▼
h_{t-1} ───────────► [ Causal GRU (1-way) ] ───► h_t (256-d)
                                                    │
                                                    ▼
                                            [ MLP Classifier ] ──► p_hat(y_t) (4-d)
```

1. **Spatial Representation (DINOv3 ViT):**
   Trích xuất CLS token $z_t \in \mathbb{R}^{384}$ từ ảnh camera. DINOv3 có tính năng tự giám sát mạnh mẽ, nắm bắt được ngữ nghĩa mặt đường, mật độ xe và phương tiện bị che khuất.
2. **Causal Temporal Modeling (GRU 1 chiều):**
   $$h_t = \text{GRU}(z_t, h_{t-1})$$
   *Yêu cầu bắt buộc:* GRU phải là **1 chiều (Unidirectional / Causal)**, chỉ nhìn về quá khứ. Tuyệt đối không dùng Bidirectional GRU vì khi triển khai thực tế thời gian thực, hệ thống không thể biết trước khung hình của tương lai.
3. **Phân loại:**
   $$\hat{p}(y_t) = \text{Softmax}(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 h_t + \mathbf{b}_1) + \mathbf{b}_2)$$

---

### 5.2. Hàm Mất Mát Soft Cross-Entropy Có Ngưỡng Tự Tin
Nhãn huấn luyện là phân phối xác suất mềm $q(y_t) \in [0, 1]^4$. Hàm mất mát là Cross-Entropy mở rộng:

$$\mathcal{L}_{\text{CE}}(q, \hat{p}) = - \sum_{k=0}^{K-1} q_k(y_t) \log \hat{p}_k(y_t)$$

#### Bộ lọc ngưỡng tự tin (Confidence-Threshold Filtering):
Nếu tại một thời điểm $t$, các LF hoàn toàn bất đồng khiến phân phối nhãn mềm bị phẳng đều ($q \approx [0.25, 0.25, 0.25, 0.25]$), việc ép End Model học theo nhãn này sẽ đưa nhiễu vào trọng số mạng.
Thuật toán đặt ngưỡng lọc tự tin $\tau_{\text{conf}} = 0.40$ (dòng 17–36 `end_model.py`):

$$\mathcal{L}_{\text{End}} = \frac{1}{\sum_t \mathbb{I}(\max_k q_k(y_t) \ge \tau_{\text{conf}})} \sum_{t: \max_k q_k(y_t) \ge \tau_{\text{conf}}} \left( - \sum_{k=0}^{K-1} q_k(y_t) \log \hat{p}_k(y_t) \right)$$

Chỉ những khung hình mà Label Model có độ tự tin rõ ràng mới được tham gia cập nhật gradient cho End Model.

---

## 6. Đối Chiếu Trực Tiếp Từng Dòng Mã Nguồn Production (`label_model.py`)

| Khái niệm Lý thuyết | Biểu thức Toán học | Dòng code trong `label_model.py` |
| :--- | :--- | :--- |
| **Khởi tạo Ma trận Markov** | $A_{ii} = 0.85, \; A_{ij} = \frac{0.15}{K-1}$ | Dòng 38–39: `self.A = np.full(...); np.fill_diagonal(self.A, 0.85)` |
| **Khởi tạo Ma trận Nhầm lẫn** | $\mathbf{E}_{j, c, k, k} = 0.70$ (đường chéo) | Dòng 46–51: `self.emission[j, c, k, k] = 0.70` |
| **Log-Emission có Abstain** | $\sum_{j: \lambda_j \ge 0} \log \mathbf{E}_{j, c, k, \lambda_j}$ | Dòng 76–78: `if val >= 0: prob_prod += np.log(p_val)` |
| **Forward Pass (Log space)** | $\log \alpha_t(j) = \text{LSE}_i(\log \alpha_{t-1} + \log A) + \log e$ | Dòng 106–108: `m = np.max(prev_terms); log_alpha[t, j] = m + np.log(...)` |
| **Backward Pass (Log space)** | $\log \beta_t(i) = \text{LSE}_j(\log A + \log e + \log \beta)$ | Dòng 120–122: `m = np.max(next_terms); log_beta[t, i] = m + np.log(...)` |
| **Tính Nhãn Mềm $\gamma_t$** | $\gamma_t \propto \exp(\log \alpha_t + \log \beta_t)$ | Dòng 125–129: `log_gamma = log_alpha + log_beta; gamma = np.exp(...)` |
| **Tính Chuyển tiếp $\xi_t$** | $\xi_t \propto \exp(\log \alpha_t + \log A + \log e + \log \beta)$ | Dòng 134–142: `term = log_alpha + log_A + ...; xi[t] = ...` |
| **M-step: Soft Counting $\mathbf{E}$** | $\text{Num} += \gamma_t(k), \; \text{Den} += \gamma_t(k)$ | Dòng 185–186: `new_emission_num[...] += gamma[t]` |
| **M-step: Chuẩn hóa Markov $\mathbf{A}$** | $A_{ij} = \sum \xi(i, j) / \sum_l \sum \xi(i, l)$ | Dòng 194–195: `A_row_sums = np.sum(new_A_num...); self.A = new_A_num / ...` |
| **Laplace Smoothing** | $(\text{Num} + \epsilon) / (\text{Den} + K\epsilon)$ | Dòng 198: `self.emission = (new_emission_num + self.eps) / (...)` |
| **Theo dõi Log-Likelihood** | $\ell = \text{LogSumExp}(\log \alpha_T)$ | Dòng 111–112, 174: `total_ll += ll` |
