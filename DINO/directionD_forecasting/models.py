"""
=============================================================================
 Hướng D: Traffic Forecasting — Spatio-Temporal Graph WaveNet Architecture
 Mạng nơ-ron Đồ thị Không-Thời gian (STGNN) với Gated Dilated TCN & Adaptive Graph Convolution
 Tích hợp 3 đầu dự báo đa nhiệm: Hồi quy đa tầm, Phân loại mức độ, Cảnh báo khởi phát
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from directionD_forecasting.graph import AdaptiveAdjacencyLayer, calculate_random_walk_matrix


class DilatedCausalConv1d(nn.Module):
    """
    Tích chập 1D nhân quả mở rộng (Dilated Causal 1D Convolution).
    Causal padding đảm bảo tại thời điểm t chỉ sử dụng thông tin <= t.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N, T)
        """
        # Causal padding chỉ vào phía bên trái chiều thời gian T
        x_pad = F.pad(x, (self.padding, 0, 0, 0))
        return self.conv(x_pad)


class GraphConvolutionLayer(nn.Module):
    """
    Tầng Graph Convolution kết hợp Ma trận khoảng cách vật lý và Ma trận thích ứng.
    """

    def __init__(self, in_channels: int, out_channels: int, num_supports: int = 2):
        super().__init__()
        self.num_supports = num_supports
        self.linear = nn.Linear(in_channels * (num_supports + 1), out_channels)

    def forward(self, x: torch.Tensor, support_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: (B, C, N, T)
            support_matrices: Danh sách các ma trận kề (N, N)
        """
        B, C, N, T = x.shape
        # Chuyển x thành dạng (B, T, N, C) để nhân ma trận với ma trận kề (N, N)
        x_perm = x.permute(0, 3, 2, 1)  # (B, T, N, C)
        out_list = [x_perm]

        for A in support_matrices:
            # (N, N) x (B, T, N, C) -> (B, T, N, C)
            A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            x_gcn = torch.matmul(A_expanded, x_perm)
            out_list.append(x_gcn)

        # Nối tất cả các phép lan truyền đồ thị theo chiều kênh C
        # concatenated: (B, T, N, C * (num_supports + 1))
        concatenated = torch.cat(out_list, dim=-1)
        out = self.linear(concatenated)  # (B, T, N, out_channels)
        return out.permute(0, 3, 2, 1)   # (B, out_channels, N, T)


class STGraphWaveNetBlock(nn.Module):
    """
    Khối Không-Thời gian kết hợp Gated Dilated TCN và Graph Convolution.
    """

    def __init__(
        self,
        channels: int,
        dilation: int,
        num_supports: int = 2,
        kernel_size: int = 2,
    ):
        super().__init__()
        # Gated TCN
        self.filter_conv = DilatedCausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.gate_conv = DilatedCausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        # Graph Convolution
        self.gcn = GraphConvolutionLayer(channels, channels, num_supports=num_supports)
        # Residual & Skip
        self.res_conv = nn.Conv2d(channels, channels, kernel_size=(1, 1))

    def forward(
        self,
        x: torch.Tensor,
        support_matrices: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            res_out: (B, C, N, T) cho khối tiếp theo
            skip_out: (B, C, N, T) cho tầng dự đoán cuối
        """
        # Gated TCN activation: tanh(filter) * sigmoid(gate)
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        tcn_out = filter_out * gate_out

        # GCN
        gcn_out = self.gcn(tcn_out, support_matrices)

        # Residual connection
        res = self.res_conv(x) + gcn_out
        skip = gcn_out
        return res, skip


class CityScaleTrafficForecastingModel(nn.Module):
    """
    Mô hình Dự báo Ùn tắc Quy mô Toàn Thành phố trên Đồ thị Camera (STGNN).
    Hỗ trợ 3 đầu ra:
      1. continuous_pred: (B, N, T_out) - Dự báo mức chiếm dụng liên tục.
      2. ordinal_logits: (B, N, T_out, 4) - Phân loại 4 mức ùn tắc.
      3. onset_prob: (B, N, 3) - Xác suất bùng phát kẹt xe trong 15', 30', 60' tới.
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int = 34,  # e.g. 1 (occupancy) + 32 (dino cls) + 1 (missing mask)
        hidden_channels: int = 32,
        out_steps: int = 12,    # 12 bước = 60 phút
        num_blocks: int = 3,
        num_classes: int = 4,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_steps = out_steps
        self.hidden_channels = hidden_channels

        # Tầng học ma trận kề thích ứng ẩn
        self.adaptive_adj = AdaptiveAdjacencyLayer(num_nodes=num_nodes, embed_dim=16)

        # Tầng mở rộng đặc trưng đầu vào
        self.start_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1))

        # Danh sách các khối ST-GraphWaveNet với hệ số giãn nở lũy thừa 2 (1, 2, 4...)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            self.blocks.append(
                STGraphWaveNetBlock(
                    channels=hidden_channels,
                    dilation=dilation,
                    num_supports=2,  # 1 physical + 1 adaptive
                    kernel_size=2,
                )
            )

        # Tầng tổng hợp Skip Connections
        self.skip_conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
        )

        # Head 1: Hồi quy liên tục đa tầm (Multi-Horizon Continuous Forecasting)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_steps),
        )

        # Head 2: Phân loại mức độ ùn tắc có thứ tự (Ordinal Congestion Classification)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_steps * num_classes),
        )

        # Head 3: Cảnh báo bùng phát kẹt xe (Congestion Onset Warning Head) cho 15', 30', 60'
        # 15' = bước 3, 30' = bước 6, 60' = bước 12
        self.onset_head = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.GELU(),
            nn.Linear(32, 3),  # 3 giá trị xác suất (15', 30', 60')
        )

    def forward(
        self,
        x_input: torch.Tensor,
        physical_adj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_input: (B, C_in, N, T_in)
            physical_adj: (N, N) ma trận khoảng cách chuẩn hóa từ OpenStreetMap / Haversine
            
        Returns:
            Dict chứa:
              - "continuous_pred": (B, N, T_out)
              - "ordinal_logits": (B, N, T_out, 4)
              - "onset_prob": (B, N, 3)
        """
        B, C_in, N, T_in = x_input.shape

        # 1. Tính toán ma trận kề thích ứng
        A_adp = self.adaptive_adj()
        support_matrices = [physical_adj, A_adp]

        # 2. Chiếu đặc trưng ban đầu
        h = self.start_conv(x_input)  # (B, hidden, N, T_in)
        skip_total = 0

        # 3. Qua các khối ST-GraphWaveNet
        for block in self.blocks:
            h, skip = block(h, support_matrices)
            skip_total = skip_total + skip

        # 4. Trích xuất đặc trưng tại bước cuối cùng của chuỗi thời gian (t = T_in)
        # skip_total: (B, hidden, N, T_in)
        feat_skip = self.skip_conv(skip_total)                # (B, hidden, N, T_in)
        feat_final = feat_skip[:, :, :, -1]                   # (B, hidden, N)
        feat_final = feat_final.permute(0, 2, 1)              # (B, N, hidden)

        # 5. Các đầu ra dự đoán
        cont_pred = self.regression_head(feat_final)          # (B, N, T_out)
        
        cls_logits = self.classification_head(feat_final)     # (B, N, T_out * 4)
        cls_logits = cls_logits.view(B, N, self.out_steps, 4) # (B, N, T_out, 4)

        onset_logits = self.onset_head(feat_final)            # (B, N, 3)
        onset_prob = torch.sigmoid(onset_logits)              # (B, N, 3)

        return {
            "continuous_pred": cont_pred,
            "ordinal_logits": cls_logits,
            "onset_prob": onset_prob,
        }
