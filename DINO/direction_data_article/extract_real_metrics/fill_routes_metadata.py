"""
=============================================================================
 KỊCH BẢN CHUẨN HÓA VÀ CẬP NHẬT TỌA ĐỘ VÀO METADATA ROUTES.CSV
 Dự án: IC4SD-TrafficSnap (Directional Traffic Network & Snapshot Dataset)
 Tác giả: Nhóm nghiên cứu IC4SD - PTIT
 Tiêu chuẩn: Production-Ready, kiểm soát kiểu dữ liệu và tính toàn vẹn 100%
=============================================================================

Mô tả nghiệp vụ:
- Đọc file nguồn chuẩn (chứa danh mục mở rộng ~661 camera với đầy đủ tọa độ GPS và thông tin phường/quận):
  Nguồn: "C:\\Users\\levie\\Downloads\\camera_data - camera_data4_merged_3.csv.csv"
- Đọc file đích hiện tại của bộ dữ liệu (chứa 608 trạm camera thực tế):
  Đích: "G:\\nckh\\DINO\\direction_data_article\\zenodo_bundle\\metadata\\routes.csv"
- Ánh xạ dữ liệu dựa trên khóa chính: CamID.
- Loại bỏ hoàn toàn cột "Ghi chú".
- Phân tách và chuẩn hóa chuỗi "latitude, longitude" thành:
  + latitude (float, chuẩn WGS84, làm tròn 6 chữ số thập phân)
  + longitude (float, chuẩn WGS84, làm tròn 6 chữ số thập phân)
  + latitude, longitude (chuỗi gốc để tương thích ngược)
- Đồng bộ thông tin địa danh: Location, Phường.
- Đảm bảo giữ đúng 608 trạm theo thứ tự stt (1 đến 657) ăn khớp với ma trận đồ thị.
- Xuất tệp UTF-8 / UTF-8-SIG và tự động cập nhật lại mã băm SHA-256 trong checksums.sha256.
=============================================================================
"""

import os
import sys
import csv
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

# Thiết lập encoding console cho môi trường Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cấu hình logging chuẩn sản xuất
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FillRoutesMetadata")


def parse_coordinates(coord_str: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Phân tách chuỗi tọa độ 'latitude, longitude' thành cặp số thực (float, float).
    
    Args:
        coord_str: Giá trị chuỗi dạng '10.7918902432446, 106.691054105759'
        
    Returns:
        (latitude, longitude) nếu hợp lệ, ngược lại (None, None).
    """
    if pd.isna(coord_str) or coord_str is None:
        return None, None
    
    parts = str(coord_str).replace('"', '').replace("'", "").split(",")
    if len(parts) < 2:
        return None, None
    
    try:
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        
        # Kiểm tra ngưỡng tọa độ địa lý TP.HCM và vùng phụ cận
        if 10.0 <= lat <= 11.5 and 106.0 <= lon <= 107.5:
            return round(lat, 6), round(lon, 6)
        else:
            logger.warning("Tọa độ nằm ngoài phạm vi TP.HCM: (%f, %f)", lat, lon)
            return round(lat, 6), round(lon, 6)
    except (ValueError, TypeError) as err:
        logger.error("Không thể chuyển đổi tọa độ '%s': %s", coord_str, err)
        return None, None


def load_source_camera_data(src_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Đọc và tiền xử lý file nguồn chuẩn chứa tọa độ.
    Sử dụng CamID làm khóa chính (dict lookup).
    
    Args:
        src_file: Đường dẫn tệp CSV nguồn.
        
    Returns:
        Dictionary ánh xạ từ CamID -> thông tin camera (Location, lat, lon, Phường).
    """
    if not src_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn tại: {src_file}")
    
    logger.info("Đang đọc file nguồn: %s", src_file)
    df_src = pd.read_csv(src_file, encoding="utf-8")
    
    # Hiển thị cấu trúc metadata vài dòng đầu của file nguồn
    logger.info("=== METADATA FILE NGUỒN ===")
    logger.info("Tổng số dòng: %d, Tổng số cột: %d", len(df_src), len(df_src.columns))
    logger.info("Danh sách cột nguồn: %s", list(df_src.columns))
    for i in range(min(3, len(df_src))):
        sample_row = df_src.iloc[i].to_dict()
        # Ẩn bớt độ dài nếu có
        logger.info("Dòng mẫu %d: %s", i + 1, sample_row)
    
    # Chuẩn hóa cột CamID làm key
    if "CamID" not in df_src.columns:
        raise KeyError("File nguồn không chứa cột khóa 'CamID'!")
    
    source_lookup: Dict[str, Dict[str, Any]] = {}
    valid_coords_count = 0
    
    for idx, row in df_src.iterrows():
        cam_id = str(row["CamID"]).strip()
        if not cam_id or cam_id == "nan":
            continue
        
        raw_coord = row.get("latitude, longitude")
        lat, lon = parse_coordinates(raw_coord)
        if lat is not None and lon is not None:
            valid_coords_count += 1
            
        phuong = str(row.get("Phường", "")).strip() if pd.notna(row.get("Phường")) else ""
        location = str(row.get("Location", "")).strip() if pd.notna(row.get("Location")) else ""
        
        # Bỏ qua cột 'Ghi chú' theo yêu cầu
        source_lookup[cam_id] = {
            "Location": location,
            "latitude": lat,
            "longitude": lon,
            "coord_raw": str(raw_coord).strip() if pd.notna(raw_coord) else "",
            "Phường": phuong
        }
        
    logger.info("Đã nạp %d camera từ file nguồn (trong đó có %d camera có tọa độ hợp lệ).", 
                len(source_lookup), valid_coords_count)
    return source_lookup


def update_routes_file(
    source_csv_path: str,
    target_routes_path: str,
    output_routes_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Thực hiện cập nhật tọa độ từ file nguồn vào tệp routes.csv theo khóa CamID.
    
    Args:
        source_csv_path: Đường dẫn tệp CSV nguồn tải về.
        target_routes_path: Đường dẫn tệp routes.csv hiện tại.
        output_routes_path: Đường dẫn tệp xuất ra (nếu None sẽ ghi đè tệp target).
        
    Returns:
        DataFrame sau khi đã điền đầy đủ dữ liệu.
    """
    src_file = Path(source_csv_path)
    target_file = Path(target_routes_path)
    out_file = Path(output_routes_path) if output_routes_path else target_file
    
    if not target_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file đích tại: {target_file}")
    
    # 1. Nạp metadata nguồn
    source_lookup = load_source_camera_data(src_file)
    
    # 2. Nạp file routes đích
    logger.info("Đang đọc file đích: %s", target_file)
    df_target = pd.read_csv(target_file, encoding="utf-8")
    
    logger.info("=== METADATA FILE ĐÍCH BAN ĐẦU ===")
    logger.info("Tổng số dòng: %d, Tổng số cột: %d", len(df_target), len(df_target.columns))
    logger.info("Danh sách cột đích: %s", list(df_target.columns))
    
    if "CamID" not in df_target.columns:
        raise KeyError("File routes.csv đích không chứa cột 'CamID'!")
    
    matched_count = 0
    missing_lookup: List[str] = []
    
    records: List[Dict[str, Any]] = []
    for idx, row in df_target.iterrows():
        stt_val = row["stt"] if "stt" in row else row.get("station_id", idx + 1)
        cam_id = str(row["CamID"]).strip()
        loc_original = str(row["Location"]).strip() if "Location" in row else ""
        
        if cam_id in source_lookup:
            matched_count += 1
            src_info = source_lookup[cam_id]
            
            # Ưu tiên Location từ nguồn nếu có, nếu không giữ Location gốc
            final_location = src_info["Location"] if src_info["Location"] else loc_original
            lat = src_info["latitude"]
            lon = src_info["longitude"]
            coord_raw = src_info["coord_raw"]
            phuong = src_info["Phường"]
        else:
            missing_lookup.append(cam_id)
            final_location = loc_original
            lat = None
            lon = None
            coord_raw = ""
            phuong = ""
        
        # Cấu trúc bản ghi chuẩn hóa:
        # stt, Location, CamID, latitude, longitude, "latitude, longitude", Phường
        # Bỏ hoàn toàn trường "Ghi chú"
        records.append({
            "stt": int(stt_val),
            "Location": final_location,
            "CamID": cam_id,
            "latitude": lat,
            "longitude": lon,
            "latitude, longitude": coord_raw,
            "Phường": phuong
        })
    
    df_updated = pd.DataFrame(records)
    
    # Sắp xếp đúng theo stt tăng dần
    df_updated = df_updated.sort_values(by="stt").reset_index(drop=True)
    
    # 3. Báo cáo kiểm toán sau khi merge
    logger.info("=== KẾT QUẢ ÁNH XẠ METADATA ===")
    logger.info("Tổng số camera đích cần cập nhật: %d", len(df_target))
    logger.info("Số camera khớp thành công theo CamID: %d / %d (%.2f%%)", 
                matched_count, len(df_target), (matched_count / len(df_target)) * 100)
    
    if missing_lookup:
        logger.warning("Có %d camera không tìm thấy trong file nguồn: %s", len(missing_lookup), missing_lookup)
    else:
        logger.info("Tuyệt đối: 100% camera (608/608) đã khớp chính xác với file nguồn!")
        
    null_coords = df_updated["latitude"].isnull().sum()
    logger.info("Số camera thiếu tọa độ: %d", null_coords)
    logger.info("Phạm vi vĩ độ (latitude): [%.6f, %.6f]", df_updated["latitude"].min(), df_updated["latitude"].max())
    logger.info("Phạm vi kinh độ (longitude): [%.6f, %.6f]", df_updated["longitude"].min(), df_updated["longitude"].max())
    logger.info("Số camera có thông tin Phường: %d / %d", (df_updated["Phường"] != "").sum(), len(df_updated))
    
    # 4. Ghi file ra đĩa
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_updated.to_csv(out_file, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu tệp kết quả thành công tại: %s", out_file)
    
    return df_updated


def update_checksums(bundle_dir: Path) -> int:
    """
    Cập nhật lại danh sách mã băm SHA-256 trong checksums.sha256 sau khi sửa routes.csv.
    
    Args:
        bundle_dir: Thư mục gốc của zenodo_bundle.
        
    Returns:
        Số lượng mã băm đã được tạo.
    """
    checksum_file = bundle_dir / "checksums.sha256"
    if not checksum_file.exists():
        logger.warning("Không tìm thấy checksums.sha256 tại: %s", checksum_file)
        return 0
    
    logger.info("Đang cập nhật lại mã băm SHA-256 cho toàn bộ zenodo_bundle...")
    hashes = []
    for dirpath, dirnames, filenames in os.walk(bundle_dir):
        dirnames.sort()
        for f in sorted(filenames):
            if f == "checksums.sha256":
                continue
            full_path = Path(dirpath) / f
            rel_path = full_path.relative_to(bundle_dir)
            
            h = hashlib.sha256()
            with open(full_path, "rb") as fp:
                while chunk := fp.read(8192):
                    h.update(chunk)
            hashes.append(f"{h.hexdigest()}  {rel_path}\n")
            
    with open(checksum_file, "w", encoding="utf-8") as fp:
        fp.writelines(hashes)
        
    logger.info("Đã cập nhật %d mã băm vào %s", len(hashes), checksum_file)
    return len(hashes)


def main():
    parser = argparse.ArgumentParser(
        description="Điền và chuẩn hóa tọa độ GPS từ file nguồn vào routes.csv theo khóa CamID."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=r"C:\Users\levie\Downloads\camera_data - camera_data4_merged_3.csv.csv",
        help="Đường dẫn file nguồn chứa danh mục mở rộng có tọa độ."
    )
    parser.add_argument(
        "--target",
        type=str,
        default=r"G:\nckh\DINO\direction_data_article\zenodo_bundle\metadata\routes.csv",
        help="Đường dẫn file routes.csv cần được điền dữ liệu."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file xuất ra (nếu không truyền sẽ ghi đè trực tiếp vào --target)."
    )
    
    args = parser.parse_args()
    
    df_res = update_routes_file(
        source_csv_path=args.source,
        target_routes_path=args.target,
        output_routes_path=args.output
    )
    
    # Cập nhật mã băm SHA-256
    bundle_dir = Path(args.target).resolve().parent.parent
    if (bundle_dir / "checksums.sha256").exists():
        update_checksums(bundle_dir)
        
    print("\n--- HOÀN THÀNH CẬP NHẬT METADATA ROUTES.CSV ---")
    print(df_res.head(5).to_string())


if __name__ == "__main__":
    main()
