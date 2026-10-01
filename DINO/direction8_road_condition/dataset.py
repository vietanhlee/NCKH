"""
=============================================================================
 Hướng 8: Self-Supervised Road Surface Condition Estimation
 Module: Dataset (Nạp chuỗi ảnh nền 24h và tính toán chỉ số vật lý mặt đường)
=============================================================================
"""

import os
import glob
import re
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RoadSurfaceDataset(Dataset):
    """
    Dataset phục vụ bài toán đánh giá tình trạng mặt đường tự giám sát.
    Quét toàn bộ chuỗi ảnh nền 24 giờ (từ 00h đến 23h) trên toàn bộ mạng lưới camera:
      - Trích xuất đặc trưng vật lý sơ cấp: Độ chói mặt đường (Luminance) và
        chỉ số phản chiếu gương (Specular Reflection Index - dấu hiệu đường ướt/ngập nước).
      - Ghép cặp tự thân giữa các khung giờ cùng camera để học tính bất biến của kết cấu nền.
    """

    def __init__(
        self,
        bg_dir: str = "traffic_backgrounds",
        img_size: int = 224,
        max_samples: Optional[int] = None,
    ):
        """
        Khởi tạo RoadSurfaceDataset.

        Args:
            bg_dir: Thư mục chứa các ảnh nền được phân cấp theo route (vd: route_1/background_slot_14h.jpg).
            img_size: Kích thước chuẩn hóa đầu vào ViT (224x224).
            max_samples: Giới hạn số lượng mẫu phục vụ kiểm thử nhanh.
        """
        super().__init__()
        self.bg_dir = bg_dir
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.items: List[Dict[str, Any]] = self._discover_background_images()
        if max_samples and len(self.items) > max_samples:
            self.items = self.items[:max_samples]

        print(f"✅ [RoadSurfaceDataset] Đã tìm thấy {len(self.items)} ảnh nền đường tĩnh phục vụ phân tích mặt đường.")

    def _discover_background_images(self) -> List[Dict[str, Any]]:
        """Quét và trích xuất thông tin ảnh nền từ cấu trúc thư mục."""
        pattern = os.path.join(self.bg_dir, "**", "*.jpg")
        all_files = glob.glob(pattern, recursive=True)
        # Thêm png nếu có
        all_files.extend(glob.glob(os.path.join(self.bg_dir, "**", "*.png"), recursive=True))

        items = []
        for fpath in all_files:
            fname = os.path.basename(fpath)
            parent = os.path.basename(os.path.dirname(fpath))

            # Trích xuất route/camera ID
            m_route = re.search(r"route_(\d+)", parent, re.IGNORECASE)
            cam_id = m_route.group(1) if m_route else parent

            # Trích xuất slot giờ
            m_slot = re.search(r"slot_(\d+)h", fname, re.IGNORECASE)
            slot_h = int(m_slot.group(1)) if m_slot else 12

            items.append({
                "path": fpath,
                "cam_id": cam_id,
                "slot_h": slot_h,
                "filename": fname,
            })

        return items

    @staticmethod
    def compute_physical_surface_metrics(img_np: np.ndarray) -> Dict[str, float]:
        """
        Tính toán các chỉ số quang học vật lý của bề mặt đường:
          - Độ chói trung bình (Luminance).
          - Chỉ số phản chiếu gương (Specular Reflection Index): phản ánh đường ướt có nước phản chiếu ánh sáng.
          - Độ nhám/độ biến thiên gradient cục bộ (Texture Roughness).
        """
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype(np.float32)

        mean_lum = float(np.mean(L)) / 255.0

        # Phản chiếu gương: các điểm có độ sáng rất cao (L > 220) trong điều kiện mặt đường tối
        specular_mask = L > 215
        specular_ratio = float(np.mean(specular_mask))

        # Độ biến thiên bề mặt qua toán tử Sobel
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        roughness = float(np.mean(grad_mag)) / 255.0

        return {
            "luminance": mean_lum,
            "specular_ratio": specular_ratio,
            "roughness": roughness,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.items[idx]
        try:
            pil_img = Image.open(info["path"]).convert("RGB")
        except Exception:
            pil_img = Image.fromarray(np.full((self.img_size, self.img_size, 3), 128, dtype=np.uint8))

        img_np = np.array(pil_img)
        metrics = self.compute_physical_surface_metrics(img_np)

        tensor_img = self.transform(pil_img)

        # Nhãn chiếu sáng ước lượng: 0=Đêm (0-5h, 19-23h), 1=Bình minh/Hoàng hôn (6h, 18h), 2=Ban ngày (7-17h)
        h = info["slot_h"]
        if 7 <= h <= 17:
            illum_class = 2  # Ngày
        elif h in [6, 18]:
            illum_class = 1  # Chạng vạng
        else:
            illum_class = 0  # Đêm

        return {
            "image": tensor_img,
            "cam_id": info["cam_id"],
            "slot_h": h,
            "illum_class": torch.tensor(illum_class, dtype=torch.long),
            "luminance": torch.tensor(metrics["luminance"], dtype=torch.float32),
            "specular_ratio": torch.tensor(metrics["specular_ratio"], dtype=torch.float32),
            "roughness": torch.tensor(metrics["roughness"], dtype=torch.float32),
            "path": info["path"],
        }
