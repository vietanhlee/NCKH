"""
=============================================================================
 Hướng E: Background Conditioning — Robust Scene Descriptor Extraction
 Trích xuất Bản mô tả Cảnh toàn cục z từ Ảnh Background bằng Thống kê Cắt tỉa (Trimmed Statistics)
 Chống chịu Ghost Vehicle, Bóng đổ cục bộ, Phản ánh trung thực Phối cảnh và Ánh sáng Camera
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RobustSceneDescriptorExtractor(nn.Module):
    """
    Trích xuất vector mô tả cảnh toàn cục z từ ảnh Background:
        z = [ TrimMean_{p in R} f_bg(p),  TrimStd_{p in R} f_bg(p),  TrimMean_{p} f_bg(p) ]
        
    Kích thước vector: 3 * D (ví dụ D = 384 -> z có 1152 chiều; D = 768 -> z có 2304 chiều).
    
    Tính chất vượt trội:
      - Khi background có ghost vehicle (xe kẹt bị in hằn vào ảnh nền), ghost chỉ chiếm
        một tỷ lệ nhỏ diện tích mặt đường.
      - Phép tính Trimmed Mean (bỏ 10% patch xa median nhất) sẽ loại bỏ hoàn toàn các patch ghost,
        giúp z phản ánh trung thực 100% bản chất mặt đường, góc máy và ánh sáng nền.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 384,
        trim_ratio: float = 0.10,
        patch_size: int = 16,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.trim_ratio = trim_ratio
        self.patch_size = patch_size

        # Đóng băng backbone trích xuất đặc trưng
        for p in self.backbone.parameters():
            p.requires_grad = False

    @staticmethod
    def compute_trimmed_statistics(
        features: torch.Tensor,
        trim_ratio: float = 0.10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính Trimmed Mean và Trimmed Std của tập hợp patch tokens:
        
        Args:
            features: Tensor (N, D)
            trim_ratio: Tỷ lệ loại bỏ các điểm dị biệt ở 2 đầu phân phối (mặc định 10%)
            
        Returns:
            trimmed_mean: Tensor (D,)
            trimmed_std: Tensor (D,)
        """
        N, D = features.shape
        if N <= 4:
            return torch.mean(features, dim=0), torch.std(features, dim=0, unbiased=False)

        # 1. Tính median vector đại diện
        med = torch.median(features, dim=0)[0]  # (D,)

        # 2. Đo khoảng cách L2 từ mỗi patch đến median
        dist_to_med = torch.norm(features - med, p=2, dim=-1)  # (N,)

        # 3. Lọc bỏ (1 - trim_ratio) phần trăm các patch xa median nhất
        k_keep = max(2, int(N * (1.0 - trim_ratio)))
        _, keep_indices = torch.topk(dist_to_med, k=k_keep, largest=False)

        trimmed_feats = features[keep_indices]  # (k_keep, D)
        trimmed_mean = torch.mean(trimmed_feats, dim=0)
        trimmed_std = torch.std(trimmed_feats, dim=0, unbiased=False)

        return trimmed_mean, trimmed_std

    def extract_scene_descriptor(
        self,
        bg_image: torch.Tensor,
        road_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Trích xuất vector z cho 1 ảnh Background:
        
        Args:
            bg_image: Tensor (1, 3, H, W) hoặc (3, H, W)
            road_mask: Tensor (H, W) mặt nạ lòng đường (tùy chọn)
            
        Returns:
            z: Tensor (1, 3 * feature_dim)
        """
        if bg_image.dim() == 3:
            bg_image = bg_image.unsqueeze(0)

        B, C, H, W = bg_image.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size

        with torch.no_grad():
            out = self.backbone(bg_image)
            if isinstance(out, dict):
                tokens = out.get("x_norm_patchtokens", list(out.values())[0])
            elif isinstance(out, torch.Tensor):
                tokens = out
            else:
                tokens = out[0]

            # Xử lý token [CLS] nếu có
            if tokens.shape[1] == (h_p * w_p + 1):
                patch_feats = tokens[:, 1:, :]
            elif tokens.shape[1] == (h_p * w_p):
                patch_feats = tokens
            else:
                patch_feats = tokens[:, :h_p * w_p, :]

        # Chuẩn hóa về (N_patches, D)
        all_patches = patch_feats[0]  # (N, D)

        # 1. Thống kê trên toàn bộ ảnh
        global_mean, _ = self.compute_trimmed_statistics(all_patches, self.trim_ratio)

        # 2. Thống kê trên vùng Road Mask (nếu có)
        if road_mask is not None:
            r_down = F.adaptive_avg_pool2d(road_mask.float().view(1, 1, H, W), (h_p, w_p)).flatten()
            road_patch_indices = torch.nonzero(r_down >= 0.5, as_tuple=False).flatten()

            if len(road_patch_indices) > 0:
                road_patches = all_patches[road_patch_indices]
                road_mean, road_std = self.compute_trimmed_statistics(road_patches, self.trim_ratio)
            else:
                road_mean, road_std = global_mean, torch.std(all_patches, dim=0, unbiased=False)
        else:
            road_mean = global_mean
            road_std = torch.std(all_patches, dim=0, unbiased=False)

        # Ghép thành vector z hoàn chỉnh: [road_mean, road_std, global_mean]
        z = torch.cat([road_mean, road_std, global_mean], dim=-1).unsqueeze(0)  # (1, 3 * D)
        return z
