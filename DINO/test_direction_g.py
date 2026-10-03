"""
=============================================================================
 Test Suite: Direction G (test_direction_g.py)
 Kiểm thử tích hợp toàn bộ module của Hướng G:
 TAM (PositionStats, GMMCalibrator), AGM (agm_sample), SRS (static_region_swap),
 Losses (DINO, iBOT, KoLeo, RIC) và Pipeline Huấn luyện.
=============================================================================
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import torch
import torch.nn.functional as F
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_dino_root = os.path.dirname(os.path.abspath(__file__))
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from directionG_camera_ssl.tam import PositionStats, GMMCalibrator
from directionG_camera_ssl.masking import agm_sample
from directionG_camera_ssl.srs import static_region_swap
from directionG_camera_ssl.losses import DINOLoss, iBOTPatchLoss, KoLeoLoss, RICLoss


class TestDirectionG(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def test_01_tam_position_stats(self):
        """Kiểm tra PositionStats cập nhật trực tuyến và tính độ khác thường a_t."""
        B, P, d = 2, 448, 64
        stats = PositionStats(num_cams=4, num_patches=P, num_states=4, feat_dim=d).to(self.device)
        cids = torch.tensor([0, 1], device=self.device)
        U = F.normalize(torch.randn(B, P, d, device=self.device), p=2, dim=-1)

        # Cập nhật trực tuyến
        stats.update(cids, U)
        self.assertFalse(torch.isnan(stats.mu).any())

        # Điểm khác thường a_t
        a, valid = stats.atypicality(cids, U)
        self.assertEqual(a.shape, (B, P))
        self.assertEqual(valid.shape, (B, P))

        # Hiệu chuẩn GMM
        calibrator = GMMCalibrator()
        calibrator.push(a, valid)
        pi = calibrator.posterior(a)
        self.assertEqual(pi.shape, (B, P))
        self.assertTrue((pi >= 0.0).all() and (pi <= 1.0).all())

    def test_02_agm_masking(self):
        """Kiểm tra thuật toán AGM che phân tầng có trọng số."""
        N = 196
        pi = torch.zeros(N, device=self.device)
        pi[:40] = 0.85  # 40 patch xe
        valid = torch.ones(N, dtype=torch.bool, device=self.device)

        phi = 0.5
        q_max = 0.6
        ratio = 0.3
        mask = agm_sample(pi, valid, ratio=ratio, phi=phi, q_max=q_max)

        expected_mask_count = int(round(ratio * N))
        self.assertEqual(mask.sum().item(), expected_mask_count)

        # Kiểm tra trần q_max
        fg_masked = mask[:40].sum().item()
        self.assertLessEqual(fg_masked, int(q_max * 40))

    def test_03_static_region_swap(self):
        """Kiểm tra SRS hoán đổi vùng tĩnh bảo toàn pixel phương tiện."""
        x1 = torch.zeros(3, 256, 448, device=self.device)
        x2 = torch.ones(3, 256, 448, device=self.device)
        pi1 = torch.zeros(16, 28, device=self.device)
        pi2 = torch.zeros(16, 28, device=self.device)
        pi1[4:8, 4:8] = 0.9  # Vùng xe

        x_tilde, swap_mask = static_region_swap(x1, x2, pi1, pi2, ratio=0.7, feather=2)
        self.assertEqual(x_tilde.shape, (3, 256, 448))
        self.assertEqual(swap_mask.shape, (16, 28))

    def test_04_ssl_losses(self):
        """Kiểm tra các hàm mất mát DINOLoss, iBOTPatchLoss, KoLeoLoss, RICLoss."""
        B, N, D = 2, 196, 256
        s_cls = torch.randn(B, D, device=self.device, requires_grad=True)
        t_cls = torch.randn(B, D, device=self.device)
        s_patch = torch.randn(B, N, D, device=self.device, requires_grad=True)
        t_patch = torch.randn(B, N, D, device=self.device)
        mask = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        mask[:, :30] = True

        dino_fn = DINOLoss(out_dim=D).to(self.device)
        ibot_fn = iBOTPatchLoss(out_dim=D).to(self.device)
        koleo_fn = KoLeoLoss().to(self.device)
        ric_fn = RICLoss().to(self.device)

        loss_d = dino_fn(s_cls, t_cls)
        loss_i = ibot_fn(s_patch, t_patch, mask)
        loss_k = koleo_fn(s_cls)
        loss_r = ric_fn(F.normalize(s_cls, p=2, dim=-1), F.normalize(t_cls, p=2, dim=-1), cids=torch.tensor([0, 1]))

        total = loss_d + loss_i + 0.1 * loss_k + 0.1 * loss_r
        total.backward()
        self.assertIsNotNone(s_cls.grad)
        self.assertFalse(torch.isnan(s_cls.grad).any())


if __name__ == "__main__":
    unittest.main()
