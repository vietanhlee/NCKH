# Hướng 4: Unsupervised Traffic Anomaly Detection

## Giới thiệu
Hướng nghiên cứu phát hiện bất thường giao thông (tai nạn, tắc nghẽn, phương tiện dừng đỗ sai quy định) sử dụng phương pháp Học không giám sát (Unsupervised Learning). 
Phương pháp kết hợp trích xuất đặc trưng toàn cục từ DINO Vision Transformer và đặc trưng cục bộ (mask hình thái) từ Background Subtraction. Các đặc trưng bình thường được lưu trong một Memory Bank và cập nhật liên tục qua Exponential Moving Average (EMA). Bất kỳ phương tiện hay tình huống nào có khoảng cách k-NN lớn trong không gian đặc trưng so với Memory Bank sẽ được coi là bất thường.

## Cấu trúc thư mục
- `memory_bank.py`: Lưu trữ và cập nhật đặc trưng trạng thái bình thường (Normal state) thông qua EMA. Tính toán khoảng cách k-NN với `torch.cdist`.
- `feature_extractor.py`: Trích xuất đặc trưng kết hợp (DINO `[CLS]` token từ `backbone_loader.py` + Delta stats từ `subtraction.py`).
- `detector.py`: Pipeline đánh giá anomaly score. Có 2 giai đoạn: `fit` và `detect`.
- `run_detection.py`: Script thực thi toàn bộ luồng pipeline với `argparse` chuẩn. Tương thích pattern với direction 2 và 3.

## Cách chạy
Khởi chạy script từ thư mục gốc dự án DINO:

```bash
python direction4_anomaly_detection/run_detection.py \
    --input_dir path/to/origin_images \
    --bg_dir path/to/backgrounds \
    --output_dir output/anomaly_results \
    --backbone dinov2_vits14 \
    --threshold 2.5 \
    --bank_size 1000 \
    --fit_samples 100
```

## Kết quả đầu ra
- `anomaly_scores.csv`: Chứa danh sách các ảnh kèm anomaly score và phân loại `is_anomaly`.
- `visualizations/`: Chứa các ảnh được gán nhãn có điểm bất thường cao (kèm Delta Mask, highlight).
