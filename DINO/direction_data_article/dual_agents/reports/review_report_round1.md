# 🏛️ INDEPENDENT SCIENTIFIC AUDIT & PEER-REVIEW REPORT (ROUND 1)
**Journal Target:** Elsevier *Data in Brief* (Scopus Q1) / Nature *Scientific Data*  
**Auditor / Reviewer:** Senior Scientific Reviewer & Empirical Auditor (Agent 2)  
**Evaluation Status:** **MAJOR REVISION REQUIRED**  

---

## 1. TỔNG QUAN ĐÁNH GIÁ (EXECUTIVE SUMMARY)
Bản thảo bài báo khoa học giới thiệu bộ dữ liệu **IC4SD-TrafficSnap** gồm **608 camera giám sát** và **714,123 ảnh thời gian thực** tại TP.HCM.
Sau khi rà soát độc lập bằng công cụ tự động và kiểm toán trực tiếp trên dữ liệu gốc:
- Tính nhất quán nội bộ: 3 tiêu chí chuẩn, 4 điểm sai lệch.
- Phản biện luận điểm khoa học & giả định chưa kiểm chứng: 5 vấn đề cần giải quyết.

---

## 2. KẾT QUẢ KIỂM TOÁN TÍNH NHẤT QUÁN NỘI BỘ (INTERNAL CONSISTENCY AUDIT)
### Các chỉ số đã xác minh thành công:
- [x] Khớp số trạm camera 608 trạm.
- [x] Khớp thời gian quan sát 93.6 giờ.
- [x] Khớp tỷ lệ vi phạm PII: 0.00%.

### Các sai lệch cần khắc phục ngay:
- [!] **SAI LỆCH SỐ LƯỢNG ẢNH: Cần khớp chính xác 714,123 frames (dữ liệu thật), không dùng số làm tròn.**
- [!] **SAI LỆCH CẠNH ĐỒ THỊ: Cần khớp đúng 2,450 cạnh có hướng (d <= 6.0 km).**
- [!] **SAI LỆCH DUNG LƯỢNG: Phải ghi rõ 44.38 GiB (nhị phân) và 47.66 GB (thập phân).**
- [!] **SAI LỆCH PHÂN LOẠI TOPO: Cần đủ 1,070 cặp một chiều (60.80%), 690 cặp hai chiều (39.20%), 1,760 tổng cặp.**

---

## 3. PHẢN BIỆN PHƯƠNG PHÁP LUẬN & BẮT BẺ GIẢ ĐỊNH (METHODOLOGICAL CRITIQUES)
### Issue #1 [CRIT-01]: Statistical Rigor (PII Anonymity)
- **Luận điểm sơ khởi / Giả định cần bắt bẻ:** Tuyên bố bảo mật 0.00% tuyệt đối chỉ dựa trên mô hình phát hiện tự động (vốn có xác suất bỏ sót sót mẫu).
- **Vấn đề chỉ ra:** Tuyên bố 0.00% PII dựa trên việc không phát hiện mẫu nào mang tính võ đoán nếu không có khoảng tin cậy thống kê.
- **Yêu cầu thực nghiệm bắt buộc từ dữ liệu gốc:** Chạy kiểm định quang học và Quy tắc Thống kê Ba (Rule of Three): Với 0 vi phạm trên N = 714,123 mẫu, chặn trên khoảng tin cậy 95% là 3/N = 4.2 x 10^-6 (< 0.00042%).
- **Mục tiêu số liệu Ground-Truth cần fit vào paper:** `Rule of Three upper bound: 3 / 714,123 = 4.20e-6 (< 0.00042% at 95% CI)`

### Issue #2 [CRIT-02]: Geospatial Modeling (Network Tortuosity)
- **Luận điểm sơ khởi / Giả định cần bắt bẻ:** Giả định khoảng cách đường bộ tỷ lệ thuận đơn giản với khoảng cách không gian mà bỏ qua đặc thù sông nước Đông Nam Á.
- **Vấn đề chỉ ra:** Bài báo chỉ nêu khoảng cách OSRM mà chưa so sánh với khoảng cách trắc địa đường chim bay (Haversine).
- **Yêu cầu thực nghiệm bắt buộc từ dữ liệu gốc:** Bổ sung thực nghiệm tính hệ số uốn khúc mạng lưới đường bộ (Tortuosity index tau = d_network / d_haversine) trên toàn bộ 2,264 cặp khoảng cách >= 100m.
- **Mục tiêu số liệu Ground-Truth cần fit vào paper:** `Tortuosity tau = 1.25 +/- 0.61 (median: 1.12, IQR: [1.01, 1.33], p90: 1.64)`

### Issue #3 [CRIT-04]: Spatial Station Density
- **Luận điểm sơ khởi / Giả định cần bắt bẻ:** Tuyên bố phân bố bao phủ toàn thành phố mà không định lượng mật độ lân cận.
- **Vấn đề chỉ ra:** Mô tả mật độ trạm chưa có chỉ số định lượng về khoảng cách trắc địa giữa các camera liền kề.
- **Yêu cầu thực nghiệm bắt buộc từ dữ liệu gốc:** Chạy thực nghiệm tính phân bố khoảng cách lân cận gần nhất (Nearest Neighbor Euclidean distance) cho 608 trạm.
- **Mục tiêu số liệu Ground-Truth cần fit vào paper:** `Nearest Neighbor Euclidean distance: mean 328.7 +/- 605.5 m, median 170.3 m, IQR [69.6, 339.7] m, min 2.7 m`

### Issue #4 [CRIT-05]: Temporal IoT Sampling Integrity
- **Luận điểm sơ khởi / Giả định cần bắt bẻ:** Lấy mẫu định kỳ 300 giây hoàn hảo không có sự cố đường truyền.
- **Vấn đề chỉ ra:** Chưa giải thích thuyết phục nguyên nhân độ lệch chuẩn Delta T = 239.7s và phân bố đuôi dài.
- **Yêu cầu thực nghiệm bắt buộc từ dữ liệu gốc:** Phân tích chu kỳ làm mới buffer của máy chủ HLS (240-270s) và thống kê 14 trạm ngoại vi bị ngắt kết nối (p99 ~ 1,200s).
- **Mục tiêu số liệu Ground-Truth cần fit vào paper:** `Delta T = 269.0 +/- 239.7 s (median 263.0 s, p90 300 s, p99 1200 s; 512 trạm >=95% coverage, 14 trạm ngoại vi <50%)`

### Issue #5 [CRIT-06]: GNN Mathematical Formulation
- **Luận điểm sơ khởi / Giả định cần bắt bẻ:** Áp dụng công thức DCRNN mà không chứng minh điều kiện hội tụ phổ.
- **Vấn đề chỉ ra:** Cần chứng minh toán học tính ổn định của các toán tử ngẫu nhiên có hướng Pf, Pb và phổ Laplacian.
- **Yêu cầu thực nghiệm bắt buộc từ dữ liệu gốc:** Tính toán bán kính phổ rho(Pf), rho(Pb) và phổ trị riêng trị lớn nhất của Laplacian L_sym.
- **Mục tiêu số liệu Ground-Truth cần fit vào paper:** `rho(Pf) = 1.0, rho(Pb) = 1.0 (stochastic matrices), lambda_max <= 2.0`


---

## 4. GÓI DỮ LIỆU THỰC NGHIỆM CHUẨN XÁC ĐÍNH KÈM (GROUND-TRUTH AUDIT PACKAGE)
Reviewer cung cấp gói số liệu chuẩn từ dữ liệu gốc để Drafter Agent (Agent 1) đưa vào bài báo:

### A. Topo mạng đường bộ & Bất đối xứng hành lang:
- Tổng số nút: **608 trạm**
- Tổng số cạnh có hướng ($d_{ij} \le 6.0$ km): **2450 cạnh**
- Tổng số cặp trạm có liên kết: **1760 cặp**
- Cặp một chiều thuần túy: **1070 cặp (60.8%)**
- Cặp hai chiều: **690 cặp (39.2%)**
- Cặp hai chiều lệch cự ly: **238 cặp (34.49%)** (với 231 cặp $>50$m, 243 cặp $\ge 50$m do 12 cặp chênh đúng 50m, 238 cặp trên ma trận gốc)
- Độ lệch cự ly trung bình: **115.9 $\pm$ 266.2 m** (Trung vị: 24.0 m, Tối đa: 2780.0 m)
- Thành phần liên thông: **6 WCC** (Giant component: 599 nodes, 98.5%), **19 SCC**.
- Bán kính phổ toán tử DCRNN: $\rho(P_f) = 1.0$, $\rho(P_b) = 1.0$ (Ổn định tuyệt đối $\le 1.0$).
- Phổ Laplacian chuẩn hóa: $\lambda_{\min} = -0.0$, $\lambda_{\max} = 2.0 \le 2.0$.

### B. Thực nghiệm Hệ số uốn khúc & Phân bố không gian:
- Hệ số uốn khúc (Network Tortuosity $\tau$): **1.25 $\pm$ 0.61** (Trung vị: **1.12**, IQR: [1.01, 1.33], $p_{90} = 1.64$)
- Khoảng cách lân cận gần nhất (Nearest Neighbor): **328.7 $\pm$ 605.5 m** (Trung vị: **170.3 m**, IQR: [69.6, 339.7] m, Min: 2.7 m)
- Khung tọa độ bao (Bounding Box): Vĩ độ [10.642472$^\circ$N, 10.988295$^\circ$N], Kinh độ [106.452713$^\circ$E, 106.850624$^\circ$E].

### C. Kiểm định Quang học Nyquist & Thống kê PII:
- Ground Sampling Distance (GSD): **2.73 -- 3.25 cm/pixel**.
- Chiếu biển số xe máy: **7.0 x 5.1 px**, nét ký tự **1.79 px** (dưới ngưỡng Nyquist OCR $\ge 16.0$ px).
- Chặn trên khoảng tin cậy 95% theo Rule of Three: **< 0.00042\%** ($3/N = 4.20e-06$).
- Tỷ lệ vi phạm PII ghi nhận: **0.00%**.

### D. Chu kỳ thời gian và tính liên tục IoT:
- Chu kỳ lấy mẫu $\Delta T$: **269.0 $\pm$ 239.7 s** (Trung vị: 263.0 s, $\le 300$ s: 88.4%, $> 600$ s: 3.8%).
- Độ trễ client: **15.0 $\pm$ 4.2 s**, trạm mất đồng bộ NTP: **11 trạm (1.8%)**.

---

## 5. CHỈ THỊ HÀNH ĐỘNG DÀNH CHO DRAFTER AGENT (AGENT 1)
1. Cập nhật ngay các con số Ground-Truth vào `paper/main.tex` và các bảng `.tex`.
2. Vận dụng văn phong học thuật đỉnh cao để giải thích ý nghĩa của các con số thực tế (tại sao Tortuosity đạt 1.25, tại sao Median Asymmetry phản ánh hạ tầng TP.HCM, tại sao PII được bảo vệ vật lý).
3. Đảm bảo cấu trúc các bảng vừa vặn trang giấy, loại bỏ cảnh báo 'Float too large for page'.
4. Trình nộp lại bản thảo ở Round tiếp theo để Reviewer tái kiểm toán!
