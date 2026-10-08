"""
=============================================================================
 ST-BMB: Memory Cross-Attention Module
 Khối Tương Tác Chú Ý Đọc Bộ Nhớ Nền Điểm-theo-Điểm (Point-wise Temporal Attention)
=============================================================================
Cải tiến đột phá chuẩn CCTV cố định:
  1. Point-wise Temporal Attention: Camera cố định -> vị trí không gian l của Query
     tương ứng chính xác với vị trí l trong Memory. Tính Attention dọc theo trục T
     tại từng patch không gian L, giảm độ phức tạp từ O(L^2 * T) xuống O(L * T).
  2. Bias log(valid): Hạ triệt để trọng số của các token nằm trên xe (valid ~ 0)
     để Query chỉ chú ý vào nền đường sạch.
  3. Point-wise Background Familiarity Map: Tính Cosine Similarity tại cùng tọa độ
     không gian qua các frame trước khi cộng Temporal PE, chuẩn hóa bằng Sigmoid
     với nhiệt độ và độ dịch học được (fam_temp, fam_shift).
  4. Graceful Fallback & Memory Dropout Support: Xử lý an toàn khi bộ nhớ rỗng
     hoặc bị dropout trong quá trình huấn luyện.
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryCrossAttention(nn.Module):
    """
    Khối Point-wise Temporal Cross-Attention cho Fixed CCTV:
      Query: Visual Tokens của Frame t hiện tại, shape (B, L, D).
      Key, Value: Memory Tokens từ BMB, shape (B, T, L, D).
      Valid Mask: Độ tin cậy nền, shape (B, T, L) in [0, 1].
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) phải chia hết cho num_heads ({num_heads})"

        # Chuẩn hóa tiền xử lý (Pre-LayerNorm)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_mem = nn.LayerNorm(embed_dim)

        # Chiếu tuyến tính Multi-Head Attention
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Chuẩn hóa và khối Feed-Forward Network (FFN/MLP)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Scale căn bậc hai của head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Tham số học được hiệu chỉnh phân bố Background Familiarity Map
        # Khởi tạo nhiệt độ = 8.0, độ dịch trung tâm = 0.50
        self.fam_temp = nn.Parameter(torch.tensor(8.0))
        self.fam_shift = nn.Parameter(torch.tensor(0.50))

    def forward(
        self,
        q_tokens: torch.Tensor,
        mem_tokens: Optional[torch.Tensor] = None,
        raw_mem_tokens: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        is_empty_memory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Thực hiện truy vấn đọc bộ nhớ nền điểm-theo-điểm theo trục thời gian.
        Args:
            q_tokens: Visual tokens của frame hiện tại, shape (B, L, embed_dim).
            mem_tokens: Memory tokens đã cộng PE, shape (B, T, L, embed_dim) hoặc (B, T*L, embed_dim).
            raw_mem_tokens: Memory tokens nguyên bản chưa cộng Temporal PE (cho Familiarity Map).
            valid: Mask độ tin cậy nền, shape (B, T, L) in [0, 1].
            is_empty_memory: Biến cờ báo bộ nhớ rỗng.
        Returns:
            conditioned_tokens: Tensor đặc trưng sau khi hòa trộn với ký ức nền (B, L, embed_dim).
            bg_familiarity: Bản đồ độ tương đồng nền (B, L, 1) in [0, 1].
        """
        B, L, D = q_tokens.shape

        # Trường hợp bộ nhớ rỗng (Frame đầu tiên t=0 hoặc Cold Start): Fallback mượt mà
        if is_empty_memory or mem_tokens is None or mem_tokens.numel() == 0:
            neutral_familiarity = torch.full((B, L, 1), 0.5, device=q_tokens.device, dtype=q_tokens.dtype)
            return q_tokens, neutral_familiarity

        # Xử lý trường hợp mem_tokens truyền vào dạng 3D (B, T*L, D) để tương thích ngược
        if mem_tokens.dim() == 3:
            total_tokens = mem_tokens.shape[1]
            if total_tokens % L == 0:
                T = total_tokens // L
                mem_tokens = mem_tokens.view(B, T, L, D)
                if raw_mem_tokens is not None and raw_mem_tokens.dim() == 3:
                    raw_mem_tokens = raw_mem_tokens.view(B, T, L, D)
                if valid is not None and valid.dim() == 2:
                    valid = valid.view(B, T, L)
            else:
                # Không chia hết: fallback sang global cross-attention an toàn
                return self._fallback_global_attention(q_tokens, mem_tokens)

        B_m, T, L_m, D_m = mem_tokens.shape
        assert L == L_m, f"Kích thước patch spatial không khớp: Q có L={L}, Mem có L={L_m}"

        # Khởi tạo valid mask mặc định nếu chưa có
        if valid is None:
            valid = torch.ones((B, T, L), device=q_tokens.device, dtype=q_tokens.dtype)
        else:
            valid = valid.to(device=q_tokens.device, dtype=q_tokens.dtype)

        # 1. Tiền chuẩn hóa LayerNorm
        q_norm = self.norm_q(q_tokens)        # (B, L, D)
        mem_norm = self.norm_mem(mem_tokens)  # (B, T, L, D)

        # 2. Chiếu đa đầu Q, K, V
        # Q: (B, L, num_heads, head_dim)
        Q = self.q_proj(q_norm).view(B, L, self.num_heads, self.head_dim)
        # K, V: (B, T, L, num_heads, head_dim)
        K = self.k_proj(mem_norm).view(B, T, L, self.num_heads, self.head_dim)
        V = self.v_proj(mem_norm).view(B, T, L, self.num_heads, self.head_dim)

        # 3. Point-wise Temporal Attention theo trục T tại cùng vị trí L:
        # Einsum: Q(B, L, H, d) x K(B, T, L, H, d) -> s(B, H, L, T)
        s = torch.einsum("blhd,btlhd->bhlt", Q, K) * self.scale

        # 4. Gated Valid Bias: log(valid.clamp_min(1e-3))
        # valid: (B, T, L) -> permute(0, 2, 1): (B, L, T) -> unsqueeze(1): (B, 1, L, T)
        # Token có valid ~ 0 (đang có xe) sẽ nhận bias ~ -6.91 -> bị triệt tiêu sau Softmax
        valid_clamped = valid.clamp_min(1e-3)
        valid_bias = torch.log(valid_clamped).permute(0, 2, 1).unsqueeze(1)  # (B, 1, L, T)
        s = s + valid_bias

        # 5. Softmax dọc theo trục thời gian T
        w = F.softmax(s, dim=-1)  # (B, H, L, T)
        w = self.attn_dropout(w)

        # 6. Tập hợp giá trị (Attention Aggregation):
        # Einsum: w(B, H, L, T) x V(B, T, L, H, d) -> out(B, L, H, d) -> reshape (B, L, D)
        out = torch.einsum("bhlt,btlhd->blhd", w, V).contiguous().reshape(B, L, D)
        out = self.proj_dropout(self.out_proj(out))

        # Cổng tin cậy tổng thể của patch (Patch-wise Validity Gate):
        # Khi mọi frame trong bộ nhớ tại patch l đều có xe (valid -> 0), do tính chất bất biến
        # tịnh tiến của Softmax (Softmax(s + c) = Softmax(s)), tổng w vẫn bằng 1.0.
        # Cổng g_l sẽ triệt tiêu out về 0, ngăn chặn việc ép Query phải nhận đặc trưng xe quá khứ!
        patch_gate = valid.max(dim=1)[0].unsqueeze(-1)  # (B, L, 1)
        out = out * patch_gate

        # 7. Residual Connection + MLP Block
        x = q_tokens + out
        x = x + self.mlp(self.norm_mlp(x))

        # 8. Point-wise Background Familiarity Map:
        # Tính Cosine Similarity tại cùng vị trí không gian qua các frame trước khi cộng Temporal PE
        source_mem_for_sim = raw_mem_tokens if raw_mem_tokens is not None else mem_tokens
        if source_mem_for_sim.dim() == 3:
            source_mem_for_sim = source_mem_for_sim.view(B, T, L, D)

        with torch.no_grad():
            q_unit = F.normalize(q_norm, p=2, dim=-1)                              # (B, L, D)
            mem_raw_norm = self.norm_mem(source_mem_for_sim)
            mem_unit = F.normalize(mem_raw_norm, p=2, dim=-1)                      # (B, T, L, D)

            # Cosine similarity tại cùng vị trí l: (B, T, L)
            sim_pointwise = (q_unit.unsqueeze(1) * mem_unit).sum(dim=-1)          # (B, T, L)

            # Phạt sim ở các vùng có xe (valid < 0.5) để không nhận nhầm xe làm nền quen
            # Mask hóa bằng valid: nếu valid thấp, trừ điểm sim
            masked_sim = sim_pointwise + (valid - 1.0) * 1.5                       # (B, T, L)

            # Lấy độ tương đồng tối đa qua các khung hình trong bộ nhớ
            max_sim, _ = torch.max(masked_sim, dim=1)                              # (B, L)

        # 9. Hiệu chỉnh độ lệch phân bố bằng Sigmoid với tham số học được (fam_temp, fam_shift)
        # Sử dụng softplus + 1.0 để đảm bảo nhiệt độ luôn dương, cosine similarity càng cao thì familiarity càng cao
        temp = F.softplus(self.fam_temp) + 1.0
        bg_familiarity = torch.sigmoid(temp * (max_sim - self.fam_shift)).unsqueeze(-1)  # (B, L, 1)

        return x, bg_familiarity

    def _fallback_global_attention(
        self,
        q_tokens: torch.Tensor,
        mem_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dự phòng Attention toàn cục O(L_q * L_mem) khi shape không theo chuẩn (B, T, L, D)."""
        B, L_q, D = q_tokens.shape
        L_mem = mem_tokens.shape[1]

        q_norm = self.norm_q(q_tokens)
        mem_norm = self.norm_mem(mem_tokens)

        Q = self.q_proj(q_norm).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(mem_norm).view(B, L_mem, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(mem_norm).view(B, L_mem, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, L_q, D)
        attn_out = self.proj_dropout(self.out_proj(attn_out))

        x = q_tokens + attn_out
        x = x + self.mlp(self.norm_mlp(x))

        with torch.no_grad():
            q_unit = F.normalize(q_norm, p=2, dim=-1)
            mem_unit = F.normalize(mem_norm, p=2, dim=-1)
            sim_matrix = torch.bmm(q_unit, mem_unit.transpose(1, 2))
            max_sim, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
            temp = F.softplus(self.fam_temp) + 1.0
            bg_familiarity = torch.sigmoid(temp * (max_sim - self.fam_shift))

        return x, bg_familiarity
