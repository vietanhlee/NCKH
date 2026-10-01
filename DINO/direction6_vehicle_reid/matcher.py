"""
=============================================================================
 Hướng 6: Delta-Guided Unsupervised Vehicle Re-Identification Across Cameras
 Module: Matcher & Evaluator (So khớp đặc trưng liên camera và tính CMC / mAP)
=============================================================================
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F


class VehicleReIDMatcher:
    """
    Hệ thống so khớp và truy vấn phương tiện liên camera (Cross-Camera Vehicle Re-ID Matcher).
    Hỗ trợ:
      - Xếp hạng độ tương đồng Cosine giữa ảnh truy vấn (Query) và kho lưu trữ (Gallery).
      - Đánh giá chỉ số học thuật tiêu chuẩn: Cumulative Matching Characteristics (CMC Rank-1, Rank-5)
        và mean Average Precision (mAP).
    """

    def __init__(self, gallery_embeddings: Optional[torch.Tensor] = None, gallery_meta: Optional[List[Dict[str, Any]]] = None):
        """
        Khởi tạo Matcher.

        Args:
            gallery_embeddings: Tensor chứa các vector đặc trưng trong kho Gallery (N, D), đã chuẩn hóa L2.
            gallery_meta: Danh sách metadata tương ứng cho mỗi vector trong Gallery (camera_id, timestamp, bbox).
        """
        self.gallery_embeddings = gallery_embeddings
        self.gallery_meta = gallery_meta or []

    def set_gallery(self, embeddings: torch.Tensor, metadata: List[Dict[str, Any]]) -> None:
        """Cập nhật kho Gallery."""
        self.gallery_embeddings = F.normalize(embeddings, dim=-1, p=2)
        self.gallery_meta = metadata

    def query(
        self,
        query_embedding: torch.Tensor,
        query_cam_id: Optional[str] = None,
        top_k: int = 5,
        filter_same_camera: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm Top-k phương tiện tương đồng nhất trong Gallery.

        Args:
            query_embedding: Vector đặc trưng của xe cần tìm (1, D) hoặc (D,).
            query_cam_id: Camera phát hiện xe query (để lọc xe cùng camera nếu cần).
            top_k: Số lượng kết quả trả về.
            filter_same_camera: Lọc bỏ các kết quả từ cùng một camera (đánh giá cross-camera thực sự).

        Returns:
            Danh sách top_k kết quả gồm: 'similarity', 'meta', 'rank'.
        """
        if self.gallery_embeddings is None or len(self.gallery_embeddings) == 0:
            return []

        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        query_embedding = F.normalize(query_embedding, dim=-1, p=2)

        # Tính độ tương đồng Cosine: sim = Q * G^T
        sims = torch.matmul(query_embedding, self.gallery_embeddings.T).squeeze(0)  # (N,)
        sims_np = sims.detach().cpu().numpy()

        # Sắp xếp giảm dần
        sorted_indices = np.argsort(-sims_np)

        results = []
        rank = 1
        for idx in sorted_indices:
            meta = self.gallery_meta[idx] if idx < len(self.gallery_meta) else {}
            # Loại trừ cùng camera nếu yêu cầu
            if filter_same_camera and query_cam_id is not None and meta.get("cam_id") == query_cam_id:
                continue

            results.append({
                "rank": rank,
                "similarity": float(sims_np[idx]),
                "meta": meta,
            })
            rank += 1
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def compute_cmc_and_map(
        query_feats: torch.Tensor,
        query_ids: np.ndarray,
        query_cams: np.ndarray,
        gallery_feats: torch.Tensor,
        gallery_ids: np.ndarray,
        gallery_cams: np.ndarray,
        top_k_ranks: Tuple[int, ...] = (1, 5, 10),
    ) -> Dict[str, float]:
        """
        Tính toán CMC Rank-k và mAP theo chuẩn học thuật IEEE / CVPR.

        Args:
            query_feats: (N_q, D) embeddings truy vấn.
            query_ids: (N_q,) ID thực của phương tiện truy vấn.
            query_cams: (N_q,) Camera ID của truy vấn.
            gallery_feats: (N_g, D) embeddings kho tìm kiếm.
            gallery_ids: (N_g,) ID thực của phương tiện trong kho.
            gallery_cams: (N_g,) Camera ID của kho.
            top_k_ranks: Các mốc Rank cần tính (vd 1, 5, 10).

        Returns:
            Dict chứa 'rank-1', 'rank-5', 'rank-10', 'mAP'.
        """
        # Chuẩn hóa L2
        q_feats = F.normalize(query_feats, dim=-1, p=2)
        g_feats = F.normalize(gallery_feats, dim=-1, p=2)

        # Ma trận khoảng cách Cosine đảo (càng nhỏ càng gần)
        dist_matrix = 1.0 - torch.matmul(q_feats, g_feats.T).detach().cpu().numpy()

        num_q, num_g = dist_matrix.shape
        all_cmc = []
        all_ap = []

        for i in range(num_q):
            q_id = query_ids[i]
            q_cam = query_cams[i]
            order = np.argsort(dist_matrix[i])

            # Loại bỏ các mẫu cùng ID và cùng camera (giao thức đánh giá chuẩn)
            remove_mask = (gallery_ids[order] == q_id) & (gallery_cams[order] == q_cam)
            keep_indices = np.where(~remove_mask)[0]

            orig_cmc = (gallery_ids[order][keep_indices] == q_id).astype(np.int32)
            if not np.any(orig_cmc):
                continue

            # Tính CMC
            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:max(top_k_ranks)])

            # Tính AP (Average Precision)
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            precisions = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
            ap = (precisions * orig_cmc).sum() / max(1.0, float(num_rel))
            all_ap.append(ap)

        metrics = {}
        if len(all_cmc) > 0:
            mean_cmc = np.mean(np.array(all_cmc), axis=0)
            for k in top_k_ranks:
                metrics[f"rank-{k}"] = float(mean_cmc[k - 1]) if k - 1 < len(mean_cmc) else 0.0
            metrics["mAP"] = float(np.mean(all_ap))
        else:
            for k in top_k_ranks:
                metrics[f"rank-{k}"] = 0.0
            metrics["mAP"] = 0.0

        return metrics
