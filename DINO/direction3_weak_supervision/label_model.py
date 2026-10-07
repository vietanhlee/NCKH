"""
=============================================================================
 Hướng B: Weak Supervision — Context-Aware Markov Label Model
 Mô hình gộp nhãn yếu đa nguồn kết hợp Chuỗi Markov thời gian và Ma trận nhầm lẫn theo Ngữ cảnh
 Thuật toán EM (Expectation-Maximization) với Forward-Backward Algorithm chuẩn Q1
=============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class ContextAwareMarkovLabelModel:
    """
    Mô hình học nhãn yếu đa nguồn:
      - Biến ẩn y_t in {0, 1, 2, 3}: Cấp độ dịch vụ ùn tắc thực sự.
      - Ma trận chuyển trạng thái Markov A in R^{4 x 4}: Phản ánh động thái ùn tắc thay đổi chậm theo thời gian.
      - Ma trận quan sát phụ thuộc ngữ cảnh:
            pi_j^(c)(lambda_j | y) = P(lambda_j | y, context c)
        Mỗi LF j có độ tin cậy riêng biệt cho từng ngữ cảnh c (ngày/đêm, giờ cao điểm, v.v.).
      - Hỗ trợ cơ chế Abstain (lambda = -1): LF không chắc thì không đóng góp vào likelihood.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_lfs: int = 5,
        num_contexts: int = 54,
        smooth_eps: float = 1e-6,
    ):
        self.K = num_classes
        self.J = num_lfs
        self.C = num_contexts
        self.eps = smooth_eps

        # Ma trận chuyển trạng thái A[i, j] = P(y_t = j | y_{t-1} = i)
        # Khởi tạo đường chéo 0.85 (tính trơn liên tục)
        self.A = np.full((self.K, self.K), 0.15 / (self.K - 1))
        np.fill_diagonal(self.A, 0.85)

        # Phân bố khởi đầu P(y_1)
        self.pi_init = np.full(self.K, 1.0 / self.K)

        # Ma trận nhầm lẫn theo ngữ cảnh: emission[j, c, y, lambda_val]
        # lambda_val in {0, 1, 2, 3}. Khởi tạo độ chính xác ban đầu ~ 0.70
        self.emission = np.full((self.J, self.C, self.K, self.K), 0.30 / (self.K - 1))
        for j in range(self.J):
            for c in range(self.C):
                for k in range(self.K):
                    self.emission[j, c, k, k] = 0.70

    def compute_log_emission_seq(
        self,
        lf_matrix: np.ndarray,
        context_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Tính toán log P(lambda_t | y_t = k) cho toàn bộ chuỗi thời gian T bước.

        Args:
            lf_matrix: Mảng numpy (T, J) chứa nhãn của J LFs (-1 là Abstain, 0..3 là nhãn).
            context_ids: Mảng numpy (T,) chứa context id của từng bước.

        Returns:
            log_emission: Mảng numpy (T, K).
        """
        T, J = lf_matrix.shape
        log_e = np.zeros((T, self.K), dtype=np.float64)

        for t in range(T):
            c_t = context_ids[t]
            for k in range(self.K):
                prob_prod = 0.0
                for j in range(J):
                    val = lf_matrix[t, j]
                    if val >= 0:  # Không Abstain
                        p_val = max(self.eps, self.emission[j, c_t, k, val])
                        prob_prod += np.log(p_val)
                log_e[t, k] = prob_prod

        return log_e

    def forward_backward(
        self,
        log_emission: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Thuật toán Forward-Backward trong không gian log (log-sum-exp) chống tràn số.

        Returns:
            gamma: Posterior P(y_t = k | lambda_{1:T}) shape (T, K).
            xi: P(y_t = i, y_{t+1} = j | lambda_{1:T}) shape (T - 1, K, K).
            log_likelihood: log P(lambda_{1:T}).
        """
        T, K = log_emission.shape
        log_A = np.log(np.maximum(self.A, self.eps))
        log_pi = np.log(np.maximum(self.pi_init, self.eps))

        # 1. Forward Pass
        log_alpha = np.zeros((T, K), dtype=np.float64)
        log_alpha[0] = log_pi + log_emission[0]

        for t in range(1, T):
            for j in range(K):
                # log sum_i exp(log_alpha[t-1, i] + log_A[i, j])
                prev_terms = log_alpha[t - 1] + log_A[:, j]
                m = np.max(prev_terms)
                log_alpha[t, j] = m + np.log(np.sum(np.exp(prev_terms - m))) + log_emission[t, j]

        # Log-likelihood tổng thể
        m_ll = np.max(log_alpha[-1])
        log_ll = m_ll + np.log(np.sum(np.exp(log_alpha[-1] - m_ll)))

        # 2. Backward Pass
        log_beta = np.zeros((T, K), dtype=np.float64)
        log_beta[-1] = 0.0

        for t in range(T - 2, -1, -1):
            for i in range(K):
                next_terms = log_A[i, :] + log_emission[t + 1, :] + log_beta[t + 1, :]
                m = np.max(next_terms)
                log_beta[t, i] = m + np.log(np.sum(np.exp(next_terms - m)))

        # 3. Tính Gamma = P(y_t = k | observations)
        log_gamma = log_alpha + log_beta
        # Chuẩn hóa log-sum-exp theo từng t
        m_g = np.max(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma - m_g)
        gamma = gamma / np.maximum(1e-12, np.sum(gamma, axis=1, keepdims=True))

        # 4. Tính Xi = P(y_t = i, y_{t+1} = j | observations)
        xi = np.zeros((T - 1, K, K), dtype=np.float64)
        for t in range(T - 1):
            term = (
                log_alpha[t, :, None]
                + log_A
                + log_emission[t + 1, None, :]
                + log_beta[t + 1, None, :]
            )
            m_xi = np.max(term)
            xi_unnorm = np.exp(term - m_xi)
            xi[t] = xi_unnorm / np.maximum(1e-12, np.sum(xi_unnorm))

        return gamma, xi, float(log_ll)

    def fit_em(
        self,
        sequences_lf: List[np.ndarray],
        sequences_ctx: List[np.ndarray],
        max_iters: int = 50,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> List[float]:
        """
        Huấn luyện EM (Expectation-Maximization) trên tập hợp các chuỗi thời gian.
        """
        history_ll = []

        for it in range(max_iters):
            total_ll = 0.0

            # Bộ tích lũy tham số cho M-step
            new_A_num = np.zeros((self.K, self.K), dtype=np.float64)
            new_emission_num = np.zeros((self.J, self.C, self.K, self.K), dtype=np.float64)
            new_emission_den = np.zeros((self.J, self.C, self.K, 1), dtype=np.float64)
            new_pi_num = np.zeros(self.K, dtype=np.float64)

            # E-Step: Tính kỳ vọng posterior trên từng chuỗi
            for lf_mat, ctx_ids in zip(sequences_lf, sequences_ctx):
                if len(lf_mat) < 2:
                    continue
                log_e = self.compute_log_emission_seq(lf_mat, ctx_ids)
                gamma, xi, ll = self.forward_backward(log_e)
                total_ll += ll

                new_pi_num += gamma[0]
                new_A_num += np.sum(xi, axis=0)

                T = len(lf_mat)
                for t in range(T):
                    c_t = ctx_ids[t]
                    for j in range(self.J):
                        val = lf_mat[t, j]
                        if val >= 0:
                            new_emission_num[j, c_t, :, val] += gamma[t]
                            new_emission_den[j, c_t, :, 0] += gamma[t]

            history_ll.append(total_ll)

            # M-Step: Cập nhật tham số
            self.pi_init = new_pi_num / np.maximum(1e-12, np.sum(new_pi_num))

            # Cập nhật ma trận A
            A_row_sums = np.sum(new_A_num, axis=1, keepdims=True)
            self.A = new_A_num / np.maximum(1e-12, A_row_sums)

            # Cập nhật emission
            self.emission = (new_emission_num + self.eps) / np.maximum(1e-12, new_emission_den + self.K * self.eps)

            if verbose and (it + 1) % 5 == 0:
                print(f"   [EM Step {it+1}/{max_iters}] Log-Likelihood: {total_ll:.2f}")

            if it > 0 and abs(history_ll[-1] - history_ll[-2]) < tol:
                if verbose:
                    print(f"   ✅ EM hội tụ tại vòng lặp {it+1}.")
                break

        return history_ll

    def predict_soft_labels(
        self,
        lf_matrix: np.ndarray,
        context_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Sinh phân phối nhãn mềm q(y_t) = [q0, q1, q2, q3] cho từng bước thời gian.
        """
        log_e = self.compute_log_emission_seq(lf_matrix, context_ids)
        gamma, _, _ = self.forward_backward(log_e)
        return gamma
