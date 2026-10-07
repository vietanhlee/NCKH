# 🚦 IC4SD-TrafficSnap: A City-Scale Multi-Camera Image Time-Series and Geospatial Road Network Dataset for Heterogeneous Urban Traffic

> **Tài liệu kiểm soát tiến độ & Hướng dẫn sử dụng nhanh cho tác giả**  
> **Cập nhật:** Tháng 10/2026 | **Tác giả:** Viet-Anh Le & Dr. Khanh Nguyen-Trong (IC4SD Lab, PTIT)

---

## 📌 1. BẢN CHẤT DỰ ÁN & VỊ TRÍ TRONG HỆ SINH THÁI DINO

Hướng nghiên cứu này đóng vai trò là **Công trình Công bố Dữ liệu Mở Quốc tế (Data Article)** độc lập, nhằm chính thức xuất bản và chuẩn hóa bộ dữ liệu thị giác giao thông đa phương thức làm nền tảng nuôi sống toàn bộ các hướng nghiên cứu trong [`DINO/`](file:///g:/nckh/DINO):

- **Tên bộ dữ liệu chính thức**: **`IC4SD-TrafficSnap`**  
  *IC4SD-TrafficSnap: A City-Scale Multi-Camera Image Time-Series and Geospatial Road Network Dataset for Heterogeneous Urban Traffic*
- **2 Cấu phần Dữ liệu Thực thể Cốt lõi (Primary Modalities):**
  1. **Chuỗi ảnh thời gian thực từ 608 Camera (Surveillance Image Time-Series):**
     - Thu thập liên tục qua **93.6 giờ (trải dài 5 ngày dương lịch: 02/10 đến 06/10/2026)** từ **608 camera giám sát cố định** của TP.HCM (khu vực các quận cũ / vùng lõi đô thị trước đợt sắp xếp hành chính tháng 7/2025). Toàn bộ 608 trạm camera giám sát được số hóa tọa độ chính xác và đồng bộ hoàn chỉnh trên sơ đồ mạng lưới địa không gian.
     - Khoảng cách lấy mẫu thực nghiệm: $\Delta T = 269.0 \pm 239.7\text{ s}$ (Trung vị: $263.0\text{ s}$, mục tiêu polling danh định: 5.0 phút).
     - Tổng lượng ảnh hợp lệ: **714,123 snapshots** (loại bỏ toàn bộ thẻ lỗi placeholder).
     - Độ phân giải: **$512 \times 288$ pixels** (tỷ lệ 16:9 streaming, nén JPEG gốc chất lượng ~75--80).
     - Dung lượng lưu trữ: **44.38 GiB** (tương đương **47.66 GB** thập phân; dung lượng tệp trung bình $65.17 \pm 13.36\text{ KB}$).
     - Độ trễ ingest: $\Delta t_{\text{lag}} = 15.0 \pm 4.2\text{ s}$ so với đồng hồ phần cứng camera trên các trạm đồng bộ NTP.
  2. **Thông tin Bản đồ Địa không gian & Đồ thị Dẫn xuất (Geospatial Road Network & Derived Graph):**
     - Tọa độ GPS trắc địa, độ cao cột (6--15m), phân loại tuyến đường của 608 trạm (`metadata/routes.csv`).
     - Ma trận cự ly lái xe thực tế đo bằng OSRM qua bản đồ OpenStreetMap (`metadata/road_network_distance.csv`).
      - **Đồ thị không gian có hướng dẫn xuất toán học (Derived Representation):** Thiết lập qua nguyên tắc kề cận hành lang giao thông tuyến tính (sequential corridor adjacency) trong bán kính $\le 6.0\text{ km}$, tổng hợp thành các tensor machine-learning (`graph/distance_km.npy`, `graph/direction.npy`, `graph/edges.csv`) gồm **2,450 hành lang có hướng**, ghi nhận 1,070 liên kết một chiều không có cạnh ngược và 238 cặp bất đối xứng $\ge 50\text{ m}$ (cùng 452 cặp đối xứng trong 690 cặp có liên kết hai chiều).
- **Phục vụ trực tiếp cho các hướng trong thư mục `DINO/`**:
  - **Hướng H1 & HG** (`direction1_bg_guided_dino`, `directionG_camera_ssl`): Học tự giám sát bất biến camera (TAM + AGM + SRS).
  - **Hướng H2** (`direction2_scene_decomposition`): Tự bóc tách nền đường tĩnh và tiền cảnh xe cộ động.
  - **Hướng H6** (`direction6_anomaly_detection`): Phát hiện lỗi camera và sự cố giao thông không giám sát.
  - **Hướng H7** (`direction7_traffic_forecasting`): Học biểu diễn đồ thị không gian - thời gian (STGNN/DCRNN/STGCN).
  - **Hướng H8** (`direction8_bg_conditioning`): Đánh giá tính tổng quát hóa trên camera chưa từng thấy.

---

## 🏛️ 2. TẠP CHÍ MỤC TIÊU & TÌNH TRẠNG BẢN THẢO

| Mục Tiêu | Tạp Chí | Phân Hạng | Tình Trạng Hiện Tại |
| :--- | :--- | :--- | :--- |
| **Đề xuất Chính** | **Data in Brief** (Elsevier) | Scopus Q1/Q2, Gold Open Access | **ĐÃ HOÀN TẤT & BIÊN DỊCH PDF THÀNH CÔNG** (26 trang camera-ready, đầy đủ 6 Bảng và 7 Hình) |
| **Đề xuất Dự phòng** | **Scientific Data** (Nature Portfolio) | Q1, Impact Factor ~9.8 | Đầy đủ nội dung theo chuẩn *Data Descriptor* |

### Thông tin Tác giả:
- **Viet-Anh Le** (Tác giả thứ nhất): `anhlv.b23kh002@stu.ptit.edu.vn` | ORCID: [0009-0003-5748-0439](https://orcid.org/0009-0003-5748-0439)
- **Dr. Khanh Nguyen-Trong** (Corresponding Author): `khanhnt@ptit.edu.vn` | ORCID: [0000-0001-5175-8805](https://orcid.org/0000-0001-5175-8805)
- **Đơn vị**: *Intelligent Computing for Sustainable Development Laboratory (IC4SD), PTIT, Hanoi, Vietnam*.

### Vị trí các tệp bản thảo bài báo ([`paper/`](file:///g:/nckh/DINO/direction_data_article/paper)):
- 📄 **File LaTeX chính**: [`paper/main.tex`](file:///g:/nckh/DINO/direction_data_article/paper/main.tex)
- 📚 **File BibTeX trích dẫn**: [`paper/references.bib`](file:///g:/nckh/DINO/direction_data_article/paper/references.bib)
- 📑 **Bảng Specifications (Table 1)**: [`paper/tables/tab_specifications.tex`](file:///g:/nckh/DINO/direction_data_article/paper/tables/tab_specifications.tex) *(Thiết kế 4 khối, vừa khít 1 trang A4)*
- 📈 **Bảng So sánh Benchmark (Table 2)**: [`paper/tables/tab_dataset_comparison.tex`](file:///g:/nckh/DINO/direction_data_article/paper/tables/tab_dataset_comparison.tex)
- 📊 **Bảng Thống kê thực nghiệm (Table 3)**: [`paper/tables/tab_summary_stats.tex`](file:///g:/nckh/DINO/direction_data_article/paper/tables/tab_summary_stats.tex)
- 🌐 **Bảng Topo đồ thị (Table 4)**: [`paper/tables/tab_graph_metrics.tex`](file:///g:/nckh/DINO/direction_data_article/paper/tables/tab_graph_metrics.tex)
- 🔒 **Bảng Kiểm định PII (Table 5)**: [`paper/tables/tab_pii_audit.tex`](file:///g:/nckh/DINO/direction_data_article/paper/tables/tab_pii_audit.tex)
- 🔮 **Bảng Baseline Forecasting Benchmark (Table 6)**: Nhúng trực tiếp tại Section 4.6 (`paper/main.tex`)
- 🖼️ **Thư mục hình vẽ chất lượng xuất bản (300 DPI)**: [`paper/figures/`](file:///g:/nckh/DINO/direction_data_article/paper/figures/)
- 📕 **Tệp PDF bài báo hoàn chỉnh (26 trang)**: [`paper/main_clean.pdf`](file:///g:/nckh/DINO/direction_data_article/paper/main_clean.pdf)

---

## 📦 3. CẤU TRÚC GÓI ĐÓNG GÓI ZENODO ([`zenodo_bundle/`](file:///g:/nckh/DINO/direction_data_article/zenodo_bundle))

```text
DINO/direction_data_article/zenodo_bundle/
├── LICENSE                                    # Giấy phép: CC BY-NC 4.0 (ảnh/nhãn), ODbL (đồ thị), MIT (code)
├── CITATION.cff                               # Định dạng trích dẫn chuẩn Citation File Format
├── .zenodo.json                               # Siêu dữ liệu cấu hình Zenodo tự động (v2.0.0)
├── data_dictionary.md                         # Từ điển dữ liệu chuẩn
├── README.md                                  # Hướng dẫn quốc tế đầy đủ
├── checksums.sha256                           # Mã băm SHA-256 xác thực tính toàn vẹn 70 tệp
│
├── metadata/                                  # Dữ liệu hình học & bản đồ
│   ├── routes.csv                             # 608 trạm: Tọa độ, loại đường, độ cao cột (6-15m)
│   ├── stations.csv                           # Bảng tổng hợp topo, bậc vào/ra và vai trò nút
│   └── road_network_distance.csv              # Ma trận khoảng cách định tuyến 608x608 từ OSM (CSV)
│
├── graph/                                     # Ma trận đồ thị dẫn xuất định dạng chuẩn máy học
│   ├── distance_km.npy                        # Ma trận khoảng cách tính theo km (608x608, float64)
│   ├── direction.npy                          # Ma trận hướng liên kết nhị phân (608x608)
│   └── edges.csv                              # Danh sách 2.450 cạnh định tuyến thực tế
│
├── sample_preview/sample_camera_sequences/    # Tập ảnh mẫu mini & mẫu đếm xe 5,012
│
└── code/                                      # Module Python nạp dữ liệu chuẩn production
    ├── __init__.py                            # Package init
    ├── load_unlabeled_images.py               # Lớp Dataset nạp ảnh
    ├── camera_sampler.py                      # Multi-camera temporal batch sampler
    └── graph_utils.py                         # Dual-transition matrices P_f, P_b & Laplacian
```

---

## ⚡ 4. BIÊN DỊCH BÀI BÁO LATEX & CHẠY BỘ KIỂM ĐỊNH VALIDATION

### 4.1 Biên dịch PDF bài báo (26 trang)
```powershell
cd G:\nckh\DINO\direction_data_article\paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
```

### 4.2 Chạy bộ công cụ kiểm toán kỹ thuật (Technical Validation Suite)
Các script độc lập nằm trong thư mục [`validation/`](file:///g:/nckh/DINO/direction_data_article/validation):
1. **Kiểm chứng tương quan động học không-thời gian mạng lưới (§4.6):**
   ```powershell
   python validation/validate_baseline_forecasting_and_correlation.py
   ```
2. **Kiểm toán quyền riêng tư thị giác & Quy tắc ba mức thống kê (§4.5 & Bảng 5):**
   ```powershell
   python validation/verify_pii_anonymity.py
   ```
3. **Kiểm định cấu trúc topo đồ thị và ma trận chuyển tiếp (§4.3 & Bảng 4):**
   ```powershell
   python validation/validate_graph_topology.py
   ```
4. **Kiểm tra tính toàn vẹn và phân bố chuỗi thời gian (§4.1 - §4.2):**
   ```powershell
   python validation/validate_temporal_coverage.py
   ```

