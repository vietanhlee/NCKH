"""
=============================================================================
 Hướng 2: Zero-Shot Vehicle Segmentation — PCA Patch Feature Extractor
 Trích xuất đặc trưng Patch Tokens từ DINOv3/v2 và phân rã thành phần chính (PCA)
 để tự động phát hiện đối tượng phương tiện không cần nhãn giám sát
=============================================================================
"""

from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torchvision import transforms


class DINOPCAExtractor:
    """
    Bộ trích xuất và phân tích thành phần chính (PCA) trên không gian đặc trưng
    của DINOv3/DINOv2. Tận dụng tính chất nổi trội (Emergent Property):
      - PC1: Tách foreground (phương tiện) khỏi background (mặt đường, bóng đổ).
      - PC2, PC3: Mã hóa các phần ngữ nghĩa (thân xe, bánh xe, kính xe).
    """

    def __init__(
        self,
        backbone: nn.Module,
        patch_size: int = 16,
        img_size: int = 224,
        n_components: int = 3,
        device: Union[str, torch.device] = "cpu",
    ):
        self.backbone = backbone.eval()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_components = n_components
        self.device = torch.device(device)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_patch_tokens(self, image: Image.Image) -> Tuple[np.ndarray, int, int]:
        """
        Đưa ảnh qua backbone và trích xuất ma trận patch tokens dạng NumPy (N_patches, embed_dim).
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        H, W = self.img_size, self.img_size
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # Gọi qua forward_features hoặc get_intermediate_layers
        if hasattr(self.backbone, "get_intermediate_layers"):
            out = self.backbone.get_intermediate_layers(x, n=1)
            raw = out[0]
            if isinstance(raw, tuple):
                raw = raw[0]
            tokens = raw[:, 1:] if raw.shape[1] == (h_patches * w_patches + 1) else raw
        elif hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)
            if isinstance(feat, dict):
                tokens = feat.get("x_norm_patchtokens", feat.get("x_prenorm", None))
            else:
                tokens = feat[:, 1:] if feat.shape[1] == (h_patches * w_patches + 1) else feat
        else:
            cls_out = self.backbone(x)
            tokens = cls_out.unsqueeze(1).expand(-1, h_patches * w_patches, -1)

        tokens_np = tokens[0].cpu().numpy()  # (N_patches, embed_dim)
        return tokens_np, h_patches, w_patches

    def compute_pca_maps(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phân rã PCA không gian đặc trưng patch tokens.

        Returns:
            pca_rgb: Bản đồ màu 3 kênh (H_patches, W_patches, 3) đại diện cho [PC1, PC2, PC3] chuẩn hóa [0..1].
            pc1_mask: Mặt nạ phân đoạn thô dựa trên PC1 (H_patches, W_patches) float32 [0..1].
        """
        tokens, h_p, w_p = self.extract_patch_tokens(image)

        # Chuẩn hóa tâm đặc trưng
        tokens_centered = tokens - np.mean(tokens, axis=0, keepdims=True)

        pca = PCA(n_components=min(self.n_components, tokens.shape[1]))
        pca_result = pca.fit_transform(tokens_centered)  # (N_patches, 3)

        # Định dạng lại không gian 2D
        pca_map = pca_result.reshape(h_p, w_p, -1)

        # 1. Phân tích PC1: Xác định hướng dương của Foreground
        pc1 = pca_map[:, :, 0]
        # Heuristic phát hiện hướng foreground: vùng tâm ảnh thường chứa xe nhiều hơn vùng mép viền
        center_region = pc1[h_p // 4 : 3 * h_p // 4, w_p // 4 : 3 * w_p // 4]
        border_region = np.concatenate([pc1[0, :], pc1[-1, :], pc1[:, 0], pc1[:, -1]])
        if np.mean(center_region) < np.mean(border_region):
            pc1 = -pc1

        # Chuẩn hóa PC1 về đoạn [0, 1]
        pc1_norm = (pc1 - np.min(pc1)) / (np.max(pc1) - np.min(pc1) + 1e-8)
        pc1_mask = (pc1_norm > 0.5).astype(np.float32)

        # 2. Chuẩn hóa 3 thành phần chính để hiển thị RGB
        pca_rgb = np.zeros((h_p, w_p, 3), dtype=np.float32)
        for c in range(min(3, pca_map.shape[2])):
            comp = pca_map[:, :, c]
            pca_rgb[:, :, c] = (comp - np.min(comp)) / (np.max(comp) - np.min(comp) + 1e-8)

        return pca_rgb, pc1_mask
