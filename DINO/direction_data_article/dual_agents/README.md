# 🤖 HỆ THỐNG ĐA TÁC TỬ TƯƠNG TÁC PHẢN BIỆN VÀ HOÀN THIỆN BÀI BÁO KHOA HỌC: DUAL-AGENT PEER-REVIEW SYSTEM

> **Dự án:** IC4SD-TrafficSnap (*Elsevier Data in Brief* / Scopus Q1)  
> **Tác giả:** Viet-Anh Le & TS. Nguyễn Trọng Khánh (IC4SD Lab, PTIT)  
> **Thư mục:** `g:\nckh\DINO\direction_data_article\dual_agents\`

---

## 📌 1. BẢN CHẤT KIẾN TRÚC & PHÂN VAI 2 AGENT

Hệ thống được thiết kế theo mô hình **Phản biện Đối ngẫu Tương tác Lặp (Iterative Adversarial & Collaborative Peer-Review)**, tái hiện chính xác quy trình bình duyệt học thuật khắt khe nhất của các tạp chí quốc tế hàng đầu (Elsevier *Data in Brief* / Nature *Scientific Data*):

```
                        ┌────────────────────────────────────────────────────────┐
                        │             DỮ LIỆU THỰC TẾ GỐC (GROUND-TRUTH)         │
                        │ routes.csv, road_network_distance.csv, edges.csv, v.v. │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────┐   Trích xuất dữ liệu  ┌───────────────────────────────────────┐
│                AGENT 1                │◄──────────────────────┤         EMPIRICAL AUDITOR             │
│    (Paper Drafter & Storytelling)     │   Ground-Truth thực tế│              ENGINE                   │
│                                       │                       └───────────────────▲───────────────────┘
│ - Biên soạn văn phong học thuật đỉnh  │                                           │ Kích hoạt kiểm toán
│   cao, xây dựng cốt truyện nghiên cứu.│                                           │ thực nghiệm trực tiếp
│ - Khởi tạo các tuyên bố & số liệu nháp│       Round 1: Major Revision             │
│ - "Chém gió" nâng tầm số liệu thực:   │◄──────────────────────────────────────────┤                AGENT 2
│   giải thích vật lý/đô thị học sâu sắc│  - Soi chéo mâu thuẫn số liệu             │   (Scientific Reviewer & Auditor)
│ - Tích hợp thực nghiệm mới:           │  - Bắt bẻ giả định thiếu căn cứ           │
│   Tortuosity, Nyquist Rule of Three.  ├──────────────────────────────────────────►│ - Độc lập, khách quan, trung lập
│ - Cập nhật main.tex & tables/*.tex    │       Round 2: Bản thảo nâng cấp          │ - Kiểm toán tính nhất quán nội bộ
└───────────────────────────────────────┘                                           │ - Phản biện phương pháp luận
                                                                                    │ - Cấp chứng chỉ Camera-Ready
                                                                                    └───────────────────────────────────┘
```

---

## 👥 2. VAI TRÒ CHI TIẾT CỦA TỪNG TÁC TỬ

### 🎭 Agent 1: Academic Narrative Drafter & Storytelling Agent ("Agent Chém Gió & Viết Paper")
- **Tập tin phụ trách:** `agent_paper_drafter.py` (`PaperDrafterNarrativeAgent`).
- **Nghiệp vụ cốt lõi:**
  1. **Nghệ thuật cốt truyện khoa học (Scientific Storytelling):** Xâu chuỗi các số liệu thô thành câu chuyện học thuật hấp dẫn, làm nổi bật các giá trị "độc nhất vô nhị" của bộ dữ liệu `IC4SD-TrafficSnap` (giao thông hỗn hợp xe máy 70-80%, bất đối xứng đồ thị do dải phân cách và đường một chiều, chuỗi ảnh thời gian thực 93.6 giờ).
  2. **Tiếp thu phản biện toàn diện:** Không bảo thủ; tiếp nhận toàn bộ các điểm bị Reviewer "bắt bẻ", tích hợp gói dữ liệu chuẩn (Ground-Truth Data Package) do Reviewer cung cấp.
  3. **"Chém gió" nâng tầm các con số thực nghiệm:** Giải thích bản chất vật lý và đô thị học của các kết quả đo đạc:
     - Tại sao **hệ số uốn khúc mạng lưới (Tortuosity index $\tau = 1.25 \pm 0.61$)** lại phản ánh mạng lưới giao thông nan quạt kết hợp bàn cờ và hệ thống kênh rạch của TP.HCM.
     - Tại sao **33.62% cặp hai chiều bị lệch cự ly $> 50$m (232 cặp)** là do dải phân cách cứng bê tông chống xung đột dòng xe và cầu vượt phân làn.
     - Tại sao **$0.00\%$ PII** là hệ quả của kiến trúc bảo mật vật lý (Physical Privacy by Design): độ phân giải $512 \times 288$ với GSD $2.73 - 3.25$ cm/px kết hợp góc nghiêng $15^\circ - 40^\circ$ và văn hóa đội mũ bảo hiểm/đeo khẩu trang.
  4. **Nâng cấp mã nguồn LaTeX:** Tự động sửa chữa `paper/main.tex` và các bảng `paper/tables/*.tex`, tối ưu hóa độ cao dòng để không bị lỗi tràn trang.

### 🧐 Agent 2: Rigorous Scientific Reviewer & Empirical Auditor ("Agent Phản Biện & Kiểm Toán Thực Nghiệm")
- **Tập tin phụ trách:** `agent_scientific_reviewer.py` (`ScientificReviewerAuditorAgent`).
- **Nghiệp vụ cốt lõi:**
  1. **Kiểm toán tính nhất quán nội bộ (Internal Consistency Auditor):** Quét đối chiếu chéo từng con số giữa Abstract, Text trong các Section, và các Bảng (Table 1, Table 2, Table 3, Table 4, Table 5). Phát hiện bất kỳ sự lệch pha nào về số lượng camera (608), số ảnh (714,123), dung lượng (44.38 GiB / 47.66 GB), số cạnh (2,450), số cặp một chiều (1,070) và hai chiều (690).
  2. **Bắt bẻ luận điểm & giả định (Methodological Critique):** Phản biện các tuyên bố võ đoán thiếu số liệu hoặc thiếu chứng minh toán học. Yêu cầu chứng minh bán kính phổ DCRNN $\rho(P) \le 1.0$, phổ Laplacian $\lambda \in [0, 2]$, và khoảng tin cậy thống kê Rule of Three cho PII.
  3. **Trực tiếp kích hoạt thực nghiệm kiểm toán:** Thông qua `EmpiricalAuditorEngine`, tự động đo đạc trực tiếp trên các tệp dữ liệu gốc (`routes.csv`, `edges.csv`, `distance_km.npy`, v.v.), sinh gói số liệu chuẩn 100% không thể chối cãi.
  4. **Xuất báo cáo phản biện có cấu trúc:** Ban hành báo cáo phản biện chính thức (`reports/review_report_round1.md`, `reports/review_report_round2.md`) và cấp Chứng chỉ Chấp thuận Xuất bản (`reports/final_audit_certification.md`).

---

## 🔬 3. CÁC THỰC NGHIỆM KHOA HỌC ĐƯỢC BỔ SUNG & ĐO ĐẠC THỰC TẾ

| Bộ Thực Nghiệm | Phương Pháp Đo Đạc & Cơ Sở Toán Học | Kết Quả Thực Nghiệm Từ Dữ Liệu Gốc | Ý Nghĩa Học Thuật Trong Bài Báo |
| :--- | :--- | :--- | :--- |
| **1. Topo đồ thị & Bất đối xứng** | Thuật toán cắt tỉa hành lang 3 giai đoạn trên OSRM ($R_{\max} = 6.0$ km). | 608 trạm, 2,450 cạnh có hướng, 1,760 cặp nút; **1,070 cặp một chiều (60.80%)**, **690 cặp hai chiều (39.20%)**, **232 cặp lệch cự ly $> 50$m (33.62%)**. | Chứng minh tính thiết yếu của đồ thị có hướng (Directed Graph) thay vì đồ thị vô hướng trong mô hình hóa giao thông đô thị. |
| **2. Hệ số uốn khúc mạng (Network Tortuosity)** | So sánh khoảng cách lái xe OSRM với khoảng cách trắc địa Haversine: $\tau_{ij} = d_{ij}^{\text{network}} / d_{ij}^{\text{haversine}}$ trên $N = 2,264$ cặp cạnh $\ge 100$m. | **Mean $\tau = 1.25 \pm 0.61$**, **Median $\tau = 1.12$**, IQR: $[1.01, 1.33]$, $p_{90} = 1.64$. | Định lượng mức độ uốn lượn của mạng lưới đường bộ TP.HCM (dài hơn trung bình 25% so với đường chim bay do cầu vượt và vòng xoay). |
| **3. Mật độ không gian trạm lân cận** | Tính khoảng cách trắc địa Euclidean gần nhất (Nearest Neighbor) giữa 608 trạm camera. | **Mean: $328.7 \pm 605.5$ m**, **Median: $170.3$ m**, IQR: $[69.6, 339.7]$ m, Min: $2.7$ m, Max: $10.15$ km. | Chứng minh mạng lưới phủ dày đặc tại các giao lộ trọng điểm và mở rộng ra các vành đai ngoại vi. |
| **4. Kiểm định Nyquist & Rule of Three cho PII** | Hình học quang học cảm biến pinhole, GSD $2.73 - 3.25$ cm/px, kết hợp Quy tắc Thống kê Ba (Rule of Three): $3/N$ trên 714,123 ảnh. | Biển số xe máy ($19 \times 14$ cm) chỉ chiếm $\approx 7 \times 5$ px, nét chữ $\approx 1.8$ px ($< 16$ px Nyquist). **Tỷ lệ vi phạm: 0.00%**, Chặn trên khoảng tin cậy 95%: **$< 0.00042\%$**. | Chứng minh tuyệt đối tính tuân thủ Nghị định 13/2023/NĐ-CP và Luật 91/2025/QH15 về bảo vệ dữ liệu cá nhân. |
| **5. Tính ổn định phổ DCRNN & Laplacian** | Tính bán kính phổ $\rho(P_f), \rho(P_b)$ và trị riêng Laplacian chuẩn hóa $L_{\text{sym}} = I - D^{-1/2}W_{\text{sym}}D^{-1/2}$. | $\rho(P_f) = 1.0$, $\rho(P_b) = 1.0$ (Stochastic matrices); $\lambda_{\min} = 0.0$, $\lambda_{\max} \le 2.0$. | Bảo đảm tính hội tụ toán học cho các mô hình Graph Neural Networks (DCRNN, STGCN, ChebNet). |

---

## 🚀 4. HƯỚNG DẪN VẬN HÀNH TOÀN BỘ HỆ THỐNG (QUICK-START)

Mở PowerShell tại thư mục gốc của bài báo (`g:\nckh\DINO\direction_data_article\`):

### Chạy Vòng lặp Điều phối 2 Agent (Bao gồm Tự động Biên dịch LaTeX):
```powershell
python -m dual_agents.feedback_loop_orchestrator
```

### Chạy ở chế độ không biên dịch PDF (để chạy nhanh / kiểm tra logic):
```powershell
python -m dual_agents.feedback_loop_orchestrator --no-pdf
```

### Chạy bộ Test Suite tự động (8 Unit Tests):
```powershell
python dual_agents/test_dual_agents.py
```

---

## 📊 5. CẤU TRÚC ĐẦU RA BÁO CÁO (REPORTS DIRECTORY)

Sau khi chạy điều phối, các báo cáo chính thức được lưu tại `dual_agents/reports/`:
- 📄 `review_report_round1.md`: Báo cáo phản biện khắt khe Round 1 của Agent 2 (phán quyết Major Revision, bắt bẻ 4 điểm phương pháp luận).
- 📄 `review_report_round2.md`: Báo cáo thẩm định lại Round 2 của Agent 2 sau khi Agent 1 đã nâng cấp bài báo.
- 🏅 `final_audit_certification.md`: Chứng chỉ Phê duyệt Xuất bản Chính thức (Camera-Ready Certification) xác nhận 100% tính nhất quán và khả năng tái lập.
- 📕 `../paper/main.pdf`: Tệp PDF bài báo hoàn chỉnh (18 trang), được biên dịch tự động, bảng biểu vừa vặn không tràn trang.
