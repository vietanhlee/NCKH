"""
=============================================================================
 Hướng G: TAM (Temporal Atypicality Map)
 Module tính toán thống kê phân cụm không-thời gian dài hạn theo vị trí patch
 Lưu trữ đa trạng thái tĩnh (K=4), cập nhật trực tuyến và tính xác suất tiền cảnh
 hoàn toàn không cần ảnh nền median và không cần optical flow video.
=============================================================================
"""

import os
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PositionStats(nn.Module):
    """
    Quản lý thống kê đa trạng thái tĩnh theo từng vị trí patch không gian cho các camera.
    
    Kích thước:
        - num_cams (C): Số lượng camera/kênh quan sát.
        - num_patches (P): Số lượng patch tokens (mặc định 448 = 16 x 28).
        - num_states (K): Số trạng thái cụm (mặc định K = 4).
        - feat_dim (d): Số chiều đặc trưng PCA (mặc định d = 64).
    """
    def __init__(
        self,
        num_cams: int = 64,
        num_patches: int = 448,
        num_states: int = 4,
        feat_dim: int = 64,
        w_min: float = 0.15,
        s_max_mult: float = 1.5,
        eta: float = 0.02,
        eta_w: float = 0.005,
        eps: float = 0.01,
        frame_norm_alpha: float = 0.15,
    ):
        super().__init__()
        self.num_cams = num_cams
        self.num_patches = num_patches
        self.num_states = num_states
        self.feat_dim = feat_dim
        self.w_min = w_min
        self.s_max_mult = s_max_mult
        self.eta = eta
        self.eta_w = eta_w
        self.eps = eps
        self.frame_norm_alpha = frame_norm_alpha

        # 1. Tâm cụm mu: (C, P, K, d), chuẩn hóa L2
        # Khởi tạo trực giao ngẫu nhiên
        init_mu = F.normalize(torch.randn(num_cams, num_patches, num_states, feat_dim), p=2, dim=-1)
        self.register_buffer("mu", init_mu)

        # 2. Tần suất quan sát w: (C, P, K), tổng theo K = 1
        init_w = torch.full((num_cams, num_patches, num_states), 1.0 / num_states)
        self.register_buffer("w", init_w)

        # 3. Độ phân tán cosine trung bình s: (C, P, K)
        init_s = torch.full((num_cams, num_patches, num_states), 0.2)
        self.register_buffer("s", init_s)

        # Đánh dấu camera đã được khởi tạo
        self.register_buffer("is_initialized", torch.zeros(num_cams, dtype=torch.bool))

    @staticmethod
    def normalize_frame_features(U: torch.Tensor, alpha: float = 0.15) -> torch.Tensor:
        """
        Chuẩn hóa trắc quang toàn khung hình (Frame-level Photometric Normalization).
        Khử thành phần dịch chuyển đồng biến toàn cục (do bóng mây, sương mù, chế độ hồng ngoại ban đêm)
        trước khi đưa vào thống kê theo vị trí.
        U: (B, P, d) hoặc (N, P, d)
        """
        if alpha <= 0.0:
            return F.normalize(U, p=2, dim=-1)
        frame_mean = U.mean(dim=1, keepdim=True)  # (B, 1, d)
        U_centered = U - alpha * frame_mean
        return F.normalize(U_centered, p=2, dim=-1)

    def set_cam_capacity(self, target_cams: int):
        """Đặt lại số lượng camera chính xác cho PositionStats và tái cấu trúc buffer."""
        if target_cams == self.num_cams:
            return
        device = self.mu.device
        dtype = self.mu.dtype
        self.register_buffer("mu", torch.zeros(target_cams, self.num_patches, self.num_states, self.feat_dim, device=device, dtype=dtype))
        self.register_buffer("w", torch.full((target_cams, self.num_patches, self.num_states), 1.0 / self.num_states, device=device, dtype=dtype))
        self.register_buffer("s", torch.full((target_cams, self.num_patches, self.num_states), 0.2, device=device, dtype=dtype))
        self.register_buffer("is_initialized", torch.zeros(target_cams, dtype=torch.bool, device=device))
        self.num_cams = target_cams

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Tự động điều chỉnh kích thước buffer cho PositionStats khi số lượng camera trong checkpoint
        khác với số camera khởi tạo mặc định (ví dụ checkpoint có 570 camera nhưng init là 16 camera).
        """
        mu_key = prefix + "mu"
        if mu_key in state_dict:
            cams_in_ckpt = state_dict[mu_key].shape[0]
            if cams_in_ckpt != self.num_cams:
                self.set_cam_capacity(cams_in_ckpt)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def ensure_cam_capacity(self, max_cid: int):
        """Mở rộng dung lượng camera động nếu gặp camera_id lớn hơn num_cams hiện tại."""
        if max_cid < self.num_cams:
            return
        new_cams = max(max_cid + 1, self.num_cams * 2)
        device = self.mu.device
        dtype = self.mu.dtype

        new_mu = F.normalize(torch.randn(new_cams, self.num_patches, self.num_states, self.feat_dim, device=device, dtype=dtype), p=2, dim=-1)
        new_mu[:self.num_cams] = self.mu
        self.register_buffer("mu", new_mu)

        new_w = torch.full((new_cams, self.num_patches, self.num_states), 1.0 / self.num_states, device=device)
        new_w[:self.num_cams] = self.w
        self.register_buffer("w", new_w)

        new_s = torch.full((new_cams, self.num_patches, self.num_states), 0.2, device=device)
        new_s[:self.num_cams] = self.s
        self.register_buffer("s", new_s)

        new_init = torch.zeros(new_cams, dtype=torch.bool, device=device)
        new_init[:self.num_cams] = self.is_initialized
        self.register_buffer("is_initialized", new_init)
        self.num_cams = new_cams

    @torch.no_grad()
    def init_kmeans(self, cid: int, U: torch.Tensor, n_iters: int = 15):
        """
        Khởi tạo offline k-means cosine (K=4) ĐƯỢC VECTOR HÓA HOÀN TOÀN song song trên tất cả P vị trí patch
        của camera cid. Tăng tốc 50x-100x so với vòng lặp tuần tự Python.
        U: Tensor (N_frames, P, d), đã chuẩn hóa L2.
        """
        self.ensure_cam_capacity(cid)
        N, P, d = U.shape
        K = self.num_states
        device = self.mu.device
        U = self.normalize_frame_features(U.to(device).float(), alpha=self.frame_norm_alpha)

        # Chuyển vị thành (P, N, d) để tính toán batch matrix multiplication song song cho P patch
        X = U.permute(1, 0, 2).contiguous()  # (P, N, d)

        # Khởi tạo tâm cụm ngẫu nhiên cho từng patch: (P, K, d)
        if N < K:
            rand_idx = torch.randint(0, N, (P, K), device=device)
        else:
            # Sinh ma trận hoán vị ngẫu nhiên cho từng patch
            rand_scores = torch.rand(P, N, device=device)
            rand_idx = torch.topk(rand_scores, k=K, dim=-1)[1]  # (P, K)

        idx_expanded = rand_idx.unsqueeze(-1).expand(-1, -1, d)  # (P, K, d)
        centroids = torch.gather(X, dim=1, index=idx_expanded).clone()
        centroids = F.normalize(centroids, p=2, dim=-1)

        # Vòng lặp K-means cosine vector hóa trên toàn bộ P patch
        for _ in range(n_iters):
            # sim: (P, N, d) x (P, d, K) -> (P, N, K)
            sim = torch.bmm(X, centroids.transpose(1, 2))
            labels = torch.argmax(sim, dim=-1)  # (P, N)
            one_hot = F.one_hot(labels, num_classes=K).float()  # (P, N, K)

            counts = one_hot.sum(dim=1)  # (P, K)
            # Tổng vector thuộc từng cụm: (P, K, N) x (P, N, d) -> (P, K, d)
            sum_x = torch.bmm(one_hot.transpose(1, 2), X)

            # Xử lý cụm rỗng (counts == 0): chọn mẫu ngẫu nhiên từ X để tái khởi tạo
            empty_mask = (counts < 0.5)  # (P, K)
            if empty_mask.any():
                fallback_idx = torch.randint(0, N, (P, K), device=device).unsqueeze(-1).expand(-1, -1, d)
                fallback_x = torch.gather(X, dim=1, index=fallback_idx)
                new_centroids = torch.where(
                    empty_mask.unsqueeze(-1),
                    fallback_x,
                    sum_x / counts.clamp(min=1.0).unsqueeze(-1),
                )
            else:
                new_centroids = sum_x / counts.clamp(min=1.0).unsqueeze(-1)

            centroids = F.normalize(new_centroids, p=2, dim=-1)

        # Tính trọng số tần suất w và độ phân tán s lần cuối
        sim = torch.bmm(X, centroids.transpose(1, 2))  # (P, N, K)
        labels = torch.argmax(sim, dim=-1)            # (P, N)
        one_hot = F.one_hot(labels, num_classes=K).float()  # (P, N, K)
        counts = one_hot.sum(dim=1)                   # (P, K)
        weights = counts / float(max(1, N))           # (P, K)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Độ phân tán cosine: trung bình (1 - cos_sim) của các điểm thuộc cụm k
        dist_matrix = (1.0 - sim).clamp(min=0.0)      # (P, N, K)
        sum_dist = (dist_matrix * one_hot).sum(dim=1) # (P, K)
        dispersions = torch.where(
            counts > 0.5,
            (sum_dist / counts.clamp(min=1.0)).clamp(min=1e-4),
            torch.full_like(counts, 0.2),
        )

        self.mu[cid] = centroids
        self.w[cid] = weights
        self.s[cid] = dispersions
        self.is_initialized[cid] = True

    @torch.no_grad()
    def update(
        self,
        cids: torch.Tensor,
        U: torch.Tensor,
        gate_pi: Optional[torch.Tensor] = None,
    ):
        """
        Cập nhật trực tuyến có cổng (Gated Online EMA Update) các cụm trạng thái không gradient.
        Ngăn chặn phương tiện giao thông ở làn đông đúc làm nhiễm bẩn tâm cụm nền tĩnh.
        cids: (B,) Camera IDs
        U: (B, P, d) Patch features đã chuẩn hóa L2
        gate_pi: Optional (B, P) Xác suất tiền cảnh trong [0, 1]. Nếu vị trí đang có xe (pi cao),
                 tốc độ cập nhật cụm tĩnh sẽ bị khóa hãm tương ứng với (1 - pi).
        """
        B, P, d = U.shape
        max_cid = int(cids.max().item())
        self.ensure_cam_capacity(max_cid)

        U_norm = self.normalize_frame_features(U, alpha=self.frame_norm_alpha)

        # Lấy tâm cụm hiện tại tương ứng với các camera trong batch
        # mu_batch: (B, P, K, d)
        mu_b = self.mu[cids]
        w_b = self.w[cids]
        s_b = self.s[cids]

        # Tính hệ số cổng bảo vệ nền: g in [0.05, 1.0]
        if gate_pi is not None:
            gate = (1.0 - gate_pi.view(B, P).detach().clamp(0.0, 1.0)).clamp(min=0.05, max=1.0)
        else:
            gate = torch.ones((B, P), device=U.device, dtype=U.dtype)

        eta_eff = (self.eta * gate).unsqueeze(-1)       # (B, P, 1)
        eta_w_eff = (self.eta_w * gate).unsqueeze(-1)   # (B, P, 1)

        # Tính tương đồng cosine: U_norm (B, P, 1, d) x mu_b (B, P, K, d) -> cos_sim (B, P, K)
        cos_sim = (U_norm.unsqueeze(2) * mu_b).sum(dim=-1)
        k_star = torch.argmax(cos_sim, dim=-1)  # (B, P)

        # Tạo one-hot cho k_star để cập nhật vector hóa
        one_hot = F.one_hot(k_star, num_classes=self.num_states).float()  # (B, P, K)

        # Cập nhật tần suất w_k có cổng: w_k <- (1 - eta_w_eff) * w_k + eta_w_eff * 1[k == k*]
        new_w = (1.0 - eta_w_eff) * w_b + eta_w_eff * one_hot
        new_w = new_w / new_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Cập nhật tâm mu_k* và độ phân tán s_k* có cổng
        u_expanded = U_norm.unsqueeze(2).expand(-1, -1, self.num_states, -1)
        # Chỉ cập nhật tại k == k* với bước nhảy điều biến bởi eta_eff
        delta_mu = eta_eff.unsqueeze(-1) * one_hot.unsqueeze(-1) * (u_expanded - mu_b)
        new_mu = F.normalize(mu_b + delta_mu, p=2, dim=-1)

        cos_best = cos_sim.gather(dim=-1, index=k_star.unsqueeze(-1)).squeeze(-1)  # (B, P)
        dist_best = (1.0 - cos_best).clamp(min=0.0)
        dist_expanded = dist_best.unsqueeze(-1)  # (B, P, 1)
        delta_s = eta_eff * one_hot * (dist_expanded - s_b)
        new_s = (s_b + delta_s).clamp(min=1e-4)

        # Cơ chế Reset trạng thái chết / Đón nhận trạng thái nền mới (mặt đường ướt, vệt bóng mới)
        # Chỉ cho phép reset khi vị trí đó không phải là phương tiện rõ rệt (gate >= 0.5)
        low_sim_mask = (cos_best < 0.6) & (gate >= 0.5)  # (B, P)
        min_w, min_k = torch.min(new_w, dim=-1)  # (B, P)
        reinit_mask = low_sim_mask & (min_w < 0.02)
        if reinit_mask.any():
            b_idx, p_idx = torch.where(reinit_mask)
            k_reset = min_k[b_idx, p_idx]
            new_mu[b_idx, p_idx, k_reset] = U_norm[b_idx, p_idx]
            new_w[b_idx, p_idx, k_reset] = 0.05
            new_w[b_idx, p_idx] = new_w[b_idx, p_idx] / new_w[b_idx, p_idx].sum(dim=-1, keepdim=True)
            new_s[b_idx, p_idx, k_reset] = 0.1

        # Ghi ngược lại buffer chính
        self.mu[cids] = new_mu
        self.w[cids] = new_w
        self.s[cids] = new_s
        self.is_initialized[cids] = True

    @torch.no_grad()
    def static_set(self, cids: torch.Tensor) -> torch.Tensor:
        """
        Xác định tập các trạng thái tĩnh S: {k : w_k >= w_min và s_k <= s_max}
        cids: (B,)
        Returns:
            static_mask: (B, P, K) kiểu bool
        """
        w_b = self.w[cids]  # (B, P, K)
        s_b = self.s[cids]  # (B, P, K)
        
        # Ngưỡng s_max = 1.5 * median(s_b) cho từng camera
        s_median = torch.median(s_b.view(s_b.shape[0], -1), dim=-1)[0].view(-1, 1, 1)
        s_max = self.s_max_mult * s_median.clamp(min=0.05)

        static_mask = (w_b >= self.w_min) & (s_b <= s_max)
        return static_mask

    @torch.no_grad()
    def atypicality(self, cids: torch.Tensor, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính điểm khác thường không thời gian a_t(p) cho từng patch.
        cids: (B,)
        U: (B, P, d)
        Returns:
            a: (B, P) Điểm khác thường
            valid: (B, P) bool, True nếu vị trí có ít nhất 1 trạng thái tĩnh
        """
        B, P, d = U.shape
        U_norm = self.normalize_frame_features(U, alpha=self.frame_norm_alpha)
        mu_b = self.mu[cids]  # (B, P, K, d)
        s_b = self.s[cids]    # (B, P, K)
        
        # Tương đồng cosine và khoảng cách cosine: dist = 1 - cos
        cos_sim = (U_norm.unsqueeze(2) * mu_b).sum(dim=-1)  # (B, P, K)
        cos_dist = (1.0 - cos_sim).clamp(min=0.0)

        # Tính chuẩn hóa Mahalanobis-like: dist / (s_k + eps)
        norm_dist = cos_dist / (s_b + self.eps)  # (B, P, K)

        # Lọc theo tập trạng thái tĩnh S
        is_static = self.static_set(cids)  # (B, P, K)
        valid = is_static.any(dim=-1)      # (B, P) bool

        # Điểm khác thường a_t là khoảng cách nhỏ nhất tới các trạng thái tĩnh
        norm_dist_masked = norm_dist.masked_fill(~is_static, float("inf"))
        a = torch.min(norm_dist_masked, dim=-1)[0]  # (B, P)

        # Nếu không có trạng thái tĩnh nào (~valid, vị trí luôn biến động do xe liên tục),
        # gán điểm khác thường cao (2.5) để phản ánh đúng đó là vùng hoạt động tiền cảnh mạnh
        a = torch.where(valid, a, torch.tensor(2.5, device=U.device, dtype=a.dtype))
        return a, valid

    @torch.no_grad()
    def activity_prior(self, cid: int) -> torch.Tensor:
        """
        Tính bản đồ hoạt động dài hạn của camera: A_c(p) = 1 - sum_{k in S} w_k
        Phản ánh tỷ lệ thời gian vị trí p có phương tiện hoặc không tĩnh.
        """
        cid_t = torch.tensor([cid], device=self.mu.device)
        is_static = self.static_set(cid_t)[0]  # (P, K)
        w_p = self.w[cid]                      # (P, K)
        w_static = (w_p * is_static.float()).sum(dim=-1)  # (P,)
        A_c = (1.0 - w_static).clamp(0.0, 1.0)
        return A_c


class GMMCalibrator:
    """
    Ước lượng GMM 2 thành phần trên log(a_t) để hiệu chuẩn điểm khác thường
    thành xác suất tiền cảnh phương tiện pi_t(p) in [0, 1].
    """
    def __init__(self, max_buffer_size: int = 50000):
        self.max_buffer_size = max_buffer_size
        self.buffer = []
        # Tham số GMM (mean, std, weight của 2 thành phần)
        # Thành phần 0: Background (log_a thấp), Thành phần 1: Foreground/Vehicle (log_a cao)
        self.mu_bg = -1.0
        self.mu_fg = 1.0
        self.std_bg = 0.8
        self.std_fg = 0.8
        self.pi_weight = 0.3  # Tỷ lệ tiền cảnh
        self.is_fitted = False

    def push(self, a: torch.Tensor, valid: torch.Tensor):
        """Đưa các điểm khác thường hợp lệ vào buffer."""
        valid_a = a[valid].detach().cpu().numpy().flatten()
        if len(valid_a) > 0:
            log_a = np.log(np.clip(valid_a, 1e-4, 100.0))
            self.buffer.extend(log_a.tolist())
            if len(self.buffer) > self.max_buffer_size:
                self.buffer = self.buffer[-self.max_buffer_size:]

    def refit(self, max_iters: int = 30):
        """Fit GMM 2 thành phần từ buffer với số vòng lặp EM tùy biến."""
        if len(self.buffer) < 200:
            return
        data = np.array(self.buffer)
        
        # EM Algorithm đơn giản 2 cụm 1 chiều
        # Khởi tạo bằng phân vị 30% và 80%
        c1 = np.percentile(data, 30)
        c2 = np.percentile(data, 80)
        s1 = np.std(data) * 0.5 + 1e-3
        s2 = np.std(data) * 0.5 + 1e-3
        w = 0.5

        for _ in range(max(1, max_iters)):
            # E-step
            p1 = np.exp(-0.5 * ((data - c1) / s1) ** 2) / (s1 + 1e-6)
            p2 = np.exp(-0.5 * ((data - c2) / s2) ** 2) / (s2 + 1e-6)
            gamma2 = (w * p2) / (w * p2 + (1.0 - w) * p1 + 1e-8)
            gamma1 = 1.0 - gamma2

            # M-step
            n1 = np.sum(gamma1) + 1e-6
            n2 = np.sum(gamma2) + 1e-6
            new_c1 = np.sum(gamma1 * data) / n1
            new_c2 = np.sum(gamma2 * data) / n2
            new_s1 = np.sqrt(np.sum(gamma1 * (data - new_c1) ** 2) / n1 + 1e-4)
            new_s2 = np.sqrt(np.sum(gamma2 * (data - new_c2) ** 2) / n2 + 1e-4)
            new_w = n2 / float(len(data))

            # Kiểm tra hội tụ sớm
            if abs(new_c1 - c1) < 1e-4 and abs(new_c2 - c2) < 1e-4:
                c1, c2, s1, s2, w = new_c1, new_c2, new_s1, new_s2, new_w
                break
            c1, c2, s1, s2, w = new_c1, new_c2, new_s1, new_s2, new_w

        # Đảm bảo c2 > c1 (c2 là tiền cảnh xe)
        if c2 >= c1:
            self.mu_bg, self.mu_fg = float(c1), float(c2)
            self.std_bg, self.std_fg = float(s1), float(s2)
            self.pi_weight = float(w)
        else:
            self.mu_bg, self.mu_fg = float(c2), float(c1)
            self.std_bg, self.std_fg = float(s2), float(s1)
            self.pi_weight = float(1.0 - w)
        self.is_fitted = True

    def posterior(self, a: torch.Tensor) -> torch.Tensor:
        """
        Tính xác suất hậu nghiệm tiền cảnh pi_t(p) in [0, 1].
        a: (B, P) Tensor điểm khác thường
        Returns:
            pi: (B, P) Tensor xác suất tiền cảnh
        """
        log_a = torch.log(a.clamp(min=1e-4, max=100.0))

        if not self.is_fitted:
            # Fallback sigmoid chuẩn hóa mềm
            return torch.sigmoid(1.5 * (log_a - 0.5))

        # Tính mật độ Gaussian của 2 thành phần
        log_p_bg = -0.5 * ((log_a - self.mu_bg) / self.std_bg) ** 2 - np.log(self.std_bg + 1e-6)
        log_p_fg = -0.5 * ((log_a - self.mu_fg) / self.std_fg) ** 2 - np.log(self.std_fg + 1e-6)

        # Bayes rule
        p_bg = (1.0 - self.pi_weight) * torch.exp(log_p_bg.clamp(min=-20.0, max=20.0))
        p_fg = self.pi_weight * torch.exp(log_p_fg.clamp(min=-20.0, max=20.0))
        pi = p_fg / (p_fg + p_bg + 1e-8)
        return pi.clamp(0.0, 1.0)

    def state_dict(self) -> dict:
        """Xuất trạng thái GMM Calibrator để lưu vào checkpoint."""
        return {
            "mu_bg": float(self.mu_bg),
            "mu_fg": float(self.mu_fg),
            "std_bg": float(self.std_bg),
            "std_fg": float(self.std_fg),
            "pi_weight": float(self.pi_weight),
            "is_fitted": bool(self.is_fitted),
        }

    def load_state_dict(self, state: dict):
        """Khôi phục trạng thái GMM Calibrator từ checkpoint."""
        if not isinstance(state, dict):
            return
        self.mu_bg = float(state.get("mu_bg", -1.0))
        self.mu_fg = float(state.get("mu_fg", 1.0))
        self.std_bg = float(state.get("std_bg", 0.8))
        self.std_fg = float(state.get("std_fg", 0.8))
        self.pi_weight = float(state.get("pi_weight", 0.3))
        self.is_fitted = bool(state.get("is_fitted", False))

