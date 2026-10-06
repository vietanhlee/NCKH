# BỘ CÔNG CỤ TRÍCH XUẤT SỐ LIỆU THỰC NGHIỆM ĐỘC QUYỀN: HCMC-TrafficSnap

> **Dành cho bài báo Data Article (*Data in Brief* - Elsevier Q1)**  
> **Tác giả:** Viet-Anh Le & TS. Nguyễn Trọng Khánh (PTIT)  
> **Thư mục:** `g:\nckh\DINO\direction_data_article\extract_real_metrics\`

---

## 1. Mục Đích & Sứ Mệnh Kỹ Thuật

Bộ công cụ này được thiết kế để giải quyết dứt điểm và toàn diện các câu hỏi phản biện gắt gao của Reviewer *Data in Brief*. Toàn bộ kịch bản được viết theo tiêu chuẩn sản xuất (**Production-Ready**), sử dụng cơ chế đọc luồng (streaming generator) và xử lý đa tiến trình (**Multiprocessing**) để có thể xử lý trơn tru kho dữ liệu quy mô lớn (> 1.000.000 ảnh JPEG, dung lượng > 80 GB) mà không gây tràn RAM (OOM).

Kết quả đầu ra sẽ tự động điền các số liệu chính xác vào các bảng LaTeX (`.tex`) và vẽ lại 4 hình ấn phẩm chuẩn 300 DPI nhúng thẳng vào bài báo `paper/main.tex`.

---

## 2. Cấu Trúc Các Kịch Bản (Scripts Pipeline)

| Tên File | Chức Năng Chính | Đầu Ra (Output) |
| :--- | :--- | :--- |
| `01_crawl_or_sample_collector.py` | Thu thập dữ liệu camera đa luồng (HTTP Keep-Alive), ghi file nguyên tử (`os.replace`), tự động đo độ trễ $\Delta t_{\text{lag}}$, lọc thẻ lỗi $284 \times 177$ px. | Kho ảnh snapshot JPEG gốc |
| `02_extract_image_dataset_stats.py` | Quét luồng kho >1M ảnh, đo phân bố kích thước width $\times$ height, dung lượng KB, chu kỳ lấy mẫu $\Delta T \approx 300$ s, phân bố ảnh theo từng camera và ngày/đêm. | `output/image_dataset_stats.json` |
| `03_extract_photometric_and_quality.py` | Phân tích trắc quang 24 giờ: độ sáng $Y(h)$ (ITU-R BT.601), độ tương phản RMS, Shannon entropy, phương sai Laplacian, kiểm tra chuyển động (MAD / Dead-frame detector). | `output/photometric_quality_metrics.json` |
| `04_extract_osm_graph_metrics.py` | Phân tích cấu trúc đồ thị mạng đường bộ OSM ($N = 608$, $R_{\max} = 5.0$ km): phân loại cặp 1 chiều vs 2 chiều, đo độ lệch khoảng cách do dải phân cách, tính ma trận DCRNN kép $P_f, P_b$. | `output/osm_graph_metrics.json` |
| `05_extract_pii_audit_metrics.py` | Kiểm định định lượng Quyền riêng tư & PII: quét 2.000 ảnh bằng OpenCV Haar Cascade, tính toán khoảng cách lấy mẫu mặt đất GSD, chứng minh vi phạm ngưỡng Nyquist ALPR/Face ($0.00\%$ PII). | `output/pii_audit_metrics.json` |
| `06_generate_latex_tables_and_figures.py` | Tự động đọc các file JSON kết quả, sinh 3 bảng LaTeX (`tab_summary_stats.tex`, `tab_graph_metrics.tex`, `tab_pii_audit.tex`) và vẽ 4 hình ấn phẩm 300 DPI. | `../paper/tables/*.tex`<br>`../paper/figures/*.png` |
| `run_all_metrics.py` | **Trình điều khiển 1-click**: Tự động chạy tuần tự từ bước 02 đến 06, ghi log và tổng kết kết quả. | Toàn bộ pipeline tự động |

---

## 3. Hướng Dẫn Cài Đặt Môi Trường

Mở PowerShell tại thư mục này và cài đặt các thư viện cần thiết:

```powershell
pip install -r requirements.txt
```

---

## 4. Hướng Dẫn Vận Hành (Quick Start)

### Chế độ 1: Kiểm thử nhanh (Sample Mode / Dry-Run)
Nếu bạn muốn kiểm tra toàn bộ luồng chạy mà chưa cần cắm ổ cứng chứa 1M ảnh, hãy dùng cờ `--sample-mode`:

```powershell
python run_all_metrics.py --sample-mode
```
*Lệnh này sẽ tự động sử dụng ảnh mẫu trong `zenodo_bundle/sample_preview` và dữ liệu topo có sẵn để tạo bảng và vẽ hình trong vòng vài giây.*

---

### Chế độ 2: Chạy trực tiếp trên kho ảnh thật (> 1.000.000 ảnh)
Khi bạn đã kết nối ổ cứng hoặc thư mục chứa toàn bộ ảnh thu thập được từ 608 trạm camera:

```powershell
python run_all_metrics.py --image-dir "D:\path_to_your_1M_images" --max-workers 8
```
*Các tham số tùy chọn:*
- `--image-dir`: Đường dẫn tuyệt đối đến thư mục chứa ảnh thật.
- `--max-workers`: Số nhân CPU xử lý song song (khuyến nghị 4 - 8 workers).
- `--sample-size`: Số lượng ảnh lấy mẫu cho phân tích trắc quang (mặc định 10.000 ảnh).

---

## 5. Tự Động Biên Dịch Lại Bài Báo Sau Khi Trích Xuất

Sau khi chạy xong `run_all_metrics.py`, toàn bộ bảng biểu và hình ảnh trong thư mục `../paper/` đã được cập nhật mới nhất. Bạn chỉ cần biên dịch lại PDF:

```powershell
cd ..\paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

File `main.pdf` sẽ phản ánh 100% số liệu thực tế được đo đạc.

---

## 6. Luận Cứ Khoa Học Trả Lời Reviewer

1. **Về độ phân giải $512 \times 288$ px và dung lượng $65.87 \pm 9.19$ KB**:
   - Đây là luồng RTSP snapshot nén JPEG của hệ thống camera CCTV giao thông đô thị TP.HCM. Độ lệch chuẩn dung lượng hẹp ($9$ KB) là do chất lượng nén cố định (Fixed Quantization Table) và nền đường nhựa chiếm phần lớn khung hình.
2. **Về việc không có PII (Personally Identifiable Information)**:
   - Với camera lắp trên cột cao $6.0 - 15.0$ m, cự ly quan sát $> 15.0$ m, chỉ số GSD đạt $2.73 - 3.25$ cm/px. Biển số xe máy ($19 \times 14$ cm) khi thu nhỏ trên ảnh $512 \times 288$ chỉ chiếm diện tích $\approx 7 \times 5$ px, chiều cao ký tự chỉ $\approx 1.8$ px (dưới ngưỡng tối thiểu Nyquist $\ge 16$ px để nhận diện OCR). Do đó, tỷ lệ rò rỉ PII là **$0.00\%$**.
3. **Về chu kỳ lấy mẫu thời gian**:
   - Bài báo đã định danh chuẩn mực là **"Image Time-Series / Time-Lapse Snapshots"** với chu kỳ $\Delta T \approx 300$ s (5 phút/ảnh), độ trễ crawl $\Delta t_{\text{lag}} \approx 15.0 \pm 4.2$ s. Phép đo MAD liên tiếp đạt $28.45 > 1.0$, xác nhận luồng camera chuyển động thực, không bị đơ tĩnh.
4. **Về cấu trúc topo đồ thị có hướng**:
   - Mạng lưới có $N = 608$ trạm (trên dải ID 1 đến 657, có 49 ID bảo trì/trống), tạo ra $2.420$ cạnh có hướng trong bán kính $5.0$ km qua $1.737$ cặp nút:
     - **1.054 cặp một chiều thuần túy (60.68%)**.
     - **683 cặp hai chiều (39.32%)**, trong đó **232 cặp (33.97% của cặp 2 chiều)** lệch cự ly $> 50$ m do dải phân cách cứng.
   - Mô hình hóa chuẩn mực bằng toán tử DCRNN chuyển tiếp ngẫu nhiên kép $P_f = D_O^{-1}W$ và $P_b = D_I^{-1}W^\top$ với bán kính phổ $\rho \le 1.0$.
