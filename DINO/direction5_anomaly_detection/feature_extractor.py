"""
=============================================================================
 Hướng 5: Unsupervised Traffic Anomaly Detection
 Module: Feature Extractor
 Trích xuất đặc trưng DINO kết hợp thông tin cấu trúc $\\Delta$
=============================================================================
"""

import sys
import os
from typing import Dict, Any
import numpy as np
import torch
from PIL import Image

# Import local common modules
dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dino_root not in sys.path:
    sys.path.insert(0, dino_root)

from common.backbone_loader import get_dino_backbone, extract_tokens
from common.subtraction import BackgroundSubtractor


class DeltaConditionedExtractor:
    """
    Trích xuất đặc trưng hình ảnh giao thông, bao gồm:
      - Đặc trưng toàn cục từ DINO [CLS] token.
      - Đặc trưng không gian hình thái từ bản đồ sai khác $\\Delta$.
    """

    def __init__(self, backbone_name: str = 'dinov2_vits14', device: str = 'cuda'):
        """
        Khởi tạo Feature Extractor.

        Args:
            backbone_name: Tên backbone (ví dụ: 'dinov2_vits14', 'dinov3_vits16').
            device: Thiết bị chạy ('cuda' hoặc 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        # Tải backbone
        self.backbone, self.embed_dim, self.patch_size = get_dino_backbone(
            model_name=backbone_name,
            pretrained=True,
            device=self.device
        )
        self.backbone.eval()
        
        # Trừ nền
        self.subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)
        
        # Normalization cho DINO
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, origin_image: np.ndarray, bg_image: np.ndarray) -> Dict[str, Any]:
        """
        Trích xuất đặc trưng từ ảnh gốc và ảnh nền.

        Args:
            origin_image: Ảnh gốc (H, W, 3).
            bg_image: Ảnh nền (H, W, 3).

        Returns:
            dict chứa:
                - 'dino_cls': Tensor (D,) đặc trưng DINO [CLS].
                - 'delta_stats': Tensor (3,) chứa (mean, std, area_ratio > tau).
                - 'combined': Tensor (D + 3,) ghép nối.
        """
        # 1. Trích xuất thống kê Delta
        delta_norm, _ = self.subtractor.compute_delta(origin_image, bg_image)
        delta_mask = self.subtractor.extract_binary_mask(delta_norm, method="otsu")
        
        # Các thống kê từ delta
        tau = 0.15 # Ngưỡng
        mean_val = np.mean(delta_norm)
        std_val = np.std(delta_norm)
        area_ratio = np.mean(delta_norm > tau)
        
        delta_stats = torch.tensor([mean_val, std_val, area_ratio], dtype=torch.float32, device=self.device)
        
        # 2. Trích xuất DINO [CLS] token
        origin_pil = Image.fromarray(origin_image).convert("RGB")
        input_tensor = self.transform(origin_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cls_token, _ = extract_tokens(self.backbone, input_tensor, patch_size=self.patch_size)
            cls_token = cls_token.squeeze(0) # (D,)
            
        # 3. Kết hợp đặc trưng
        combined = torch.cat([cls_token, delta_stats], dim=0)
        
        return {
            'dino_cls': cls_token,
            'delta_stats': delta_stats,
            'combined': combined
        }
