"""
=============================================================================
 Hướng C: Anomaly Detection — Anomaly Scoring & Heatmap Generation
 Tính toán khoảng cách k-NN đến Memory Bank và tổng hợp điểm bất thường
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np


class AnomalyScorer:
    """
    Tính toán điểm bất thường dựa trên khoảng cách Euclidean đến Coreset Memory Bank.
    """

    def __init__(self, k_nearest: int = 1, top_k_ratio: float = 0.05):
        """
        Args:
            k_nearest: Số láng giềng gần nhất để tính khoảng cách (k-NN, mặc định k=1).
            top_k_ratio: Tỷ lệ patch có điểm cao nhất để tính điểm toàn khung hình (mặc định 5%).
        """
        self.k = k_nearest
        self.top_k_ratio = top_k_ratio

    def compute_patch_scores(
        self,
        query_feats: torch.Tensor,
        memory_bank: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính điểm bất thường cho từng patch.
        
        Args:
            query_feats: Tensor (B, N_patches, D) hoặc (N_patches, D)
            memory_bank: Tensor (M, D) từ NormalMemoryBank
            
        Returns:
            scores: Tensor (B, N_patches) - khoảng cách L2 đến láng giềng gần nhất
        """
        if query_feats.dim() == 2:
            query_feats = query_feats.unsqueeze(0)  # (1, N, D)

        B, N, D = query_feats.shape
        M, D_bank = memory_bank.shape
        assert D == D_bank, f"Dimension mismatch: {D} vs {D_bank}"

        # Tính khoảng cách Euclidean ma trận song song
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
        # Do đã chuẩn hóa L2, ||x||=1, ||y||=1 -> ||x - y||^2 = 2 - 2 <x, y>
        q_norm = F.normalize(query_feats, p=2, dim=-1)  # (B, N, D)
        m_norm = F.normalize(memory_bank, p=2, dim=-1)  # (M, D)

        # Tính cosine similarity: (B, N, M)
        cos_sim = torch.matmul(q_norm, m_norm.t())
        
        if self.k == 1:
            max_sim, _ = torch.max(cos_sim, dim=-1)  # (B, N)
        else:
            topk_sim, _ = torch.topk(cos_sim, k=self.k, dim=-1)
            max_sim = torch.mean(topk_sim, dim=-1)

        # Khoảng cách L2 tương đương: sqrt(2 - 2 * cos_sim)
        patch_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * max_sim, min=0.0))
        return patch_dist

    def aggregate_frame_score(
        self,
        patch_scores: torch.Tensor,
        road_patch_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tổng hợp điểm bất thường toàn khung hình phân tách thành 2 vùng:
        1. Vùng Lòng đường (Road Mask) -> Phản ánh sự cố giao thông.
        2. Vùng Ngoại cảnh tĩnh (Static Non-road Mask) -> Phản ánh lỗi camera / góc xoay.

        Args:
            patch_scores: (B, N_patches)
            road_patch_mask: (B, N_patches) binary (1 = lòng đường, 0 = ngoại cảnh)

        Returns:
            road_score: (B,) Điểm bất thường lòng đường (trung bình Top-K patch)
            static_score: (B,) Điểm bất thường ngoại cảnh
        """
        B, N = patch_scores.shape
        road_scores = []
        static_scores = []

        for b in range(B):
            scores_b = patch_scores[b]
            r_mask = road_patch_mask[b] > 0.5
            s_mask = ~r_mask

            # 1. Tính điểm vùng đường (Top-K)
            road_p_scores = scores_b[r_mask]
            if len(road_p_scores) > 0:
                k_road = max(1, int(len(road_p_scores) * self.top_k_ratio))
                top_vals, _ = torch.topk(road_p_scores, k=k_road)
                road_scores.append(top_vals.mean())
            else:
                road_scores.append(torch.tensor(0.0, device=patch_scores.device))

            # 2. Tính điểm vùng ngoại cảnh (Top-K)
            static_p_scores = scores_b[s_mask]
            if len(static_p_scores) > 0:
                k_static = max(1, int(len(static_p_scores) * self.top_k_ratio))
                top_vals_s, _ = torch.topk(static_p_scores, k=k_static)
                static_scores.append(top_vals_s.mean())
            else:
                static_scores.append(torch.tensor(0.0, device=patch_scores.device))

        return torch.stack(road_scores), torch.stack(static_scores)

    @staticmethod
    def generate_heatmap(
        patch_scores: torch.Tensor,
        h_patches: int,
        w_patches: int,
        target_size: Tuple[int, int] = (256, 448),
    ) -> torch.Tensor:
        """
        Tạo Heatmap nội suy song tuyến (B, 1, H, W) từ điểm patch để trực quan hóa.
        """
        B, N = patch_scores.shape
        grid = patch_scores.view(B, 1, h_patches, w_patches)
        heatmap = F.interpolate(grid, size=target_size, mode="bilinear", align_corners=False)
        return heatmap
