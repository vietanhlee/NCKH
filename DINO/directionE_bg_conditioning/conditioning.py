"""
=============================================================================
 Hướng E: Background Conditioning — Conditioning Mechanisms Suite
 Triển khai 3 cơ chế Điều kiện hóa Đa dạng để So sánh Thực nghiệm:
   1. FiLM (Feature-wise Linear Modulation) với Zero-Initialization
   2. Scene Prompt Tokens (Tiêm K tokens ngữ cảnh vào chuỗi ViT)
   3. Cross-Attention (Tương tác Token-to-Token giữa Frame và Background)
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMConditioningLayer(nn.Module):
    """
    Cơ chế 1: FiLM (Feature-wise Linear Modulation).
    Điều biến đặc trưng tuyến tính ở các tầng sâu:
        h_mod = (1 + \\gamma) \\odot \\operatorname{LayerNorm}(h) + \\beta
        
    Kỹ thuật Zero-Init: Khởi tạo trọng số MLP tầng cuối về 0 để ban đầu
    \\gamma = 0, \\beta = 0 -> Mô hình ban đầu tương đương hàm đồng nhất (Identity).
    """

    def __init__(self, descriptor_dim: int, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.ln = nn.LayerNorm(feature_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 2 * feature_dim),
        )
        
        # Zero initialization cho tầng chiếu cuối
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Tensor đặc trưng (B, N, D) hoặc (B, D)
            z: Vector mô tả cảnh (B, descriptor_dim)
        """
        film_params = self.mlp(z)  # (B, 2 * D)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        if h.dim() == 3:
            gamma = gamma.unsqueeze(1)  # (B, 1, D)
            beta = beta.unsqueeze(1)    # (B, 1, D)

        h_norm = self.ln(h)
        h_mod = (1.0 + gamma) * h_norm + beta
        return h_mod


class PromptTokenConditioningLayer(nn.Module):
    """
    Cơ chế 2: Scene Prompt Tokens.
    Chuyển đổi vector mô tả cảnh z thành K visual prompt tokens
    rồi nối trực tiếp vào chuỗi token đầu vào của Transformer.
    """

    def __init__(self, descriptor_dim: int, feature_dim: int, num_prompts: int = 4):
        super().__init__()
        self.num_prompts = num_prompts
        self.feature_dim = feature_dim

        self.prompt_proj = nn.Sequential(
            nn.Linear(descriptor_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_prompts * feature_dim),
        )

    def forward(self, tokens: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D) chuỗi tokens hiện tại
            z: (B, descriptor_dim)
        Returns:
            tokens_with_prompts: (B, N + num_prompts, D)
        """
        B = tokens.shape[0]
        prompts = self.prompt_proj(z).view(B, self.num_prompts, self.feature_dim)
        return torch.cat([prompts, tokens], dim=1)


class CrossAttentionConditioningLayer(nn.Module):
    """
    Cơ chế 3: Cross-Attention Conditioning.
    Token của khung hình hiện tại đóng vai trò Query, tương tác trực tiếp
    với Key và Value là các token patch của ảnh Background.
    """

    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(feature_dim)
        self.ln_kv = nn.LayerNorm(feature_dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # Gating parameter zero-init

    def forward(self, frame_tokens: torch.Tensor, bg_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_tokens: (B, N, D)
            bg_tokens: (B, M, D)
        """
        q = self.ln_q(frame_tokens)
        kv = self.ln_kv(bg_tokens)
        attn_out, _ = self.mha(query=q, key=kv, value=kv)
        return frame_tokens + self.gamma * attn_out
