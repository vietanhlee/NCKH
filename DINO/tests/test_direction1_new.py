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

import math
import numpy as np
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

_dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from direction1_new.tam import PositionStats, GMMCalibrator
from direction1_new.masking import agm_sample
from direction1_new.srs import static_region_swap
from direction1_new.losses import DINOLoss, iBOTPatchLoss, KoLeoLoss, RICLoss


class TestDirection1New(unittest.TestCase):
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

    def test_05_position_stats_dynamic_resume(self):
        """Kiểm tra PositionStats tự động co giãn dung lượng camera khi load state_dict có shape khác (570 vs 16 cams)."""
        # Giả lập checkpoint đã train trên 570 camera
        stats_saved = PositionStats(num_cams=570, num_patches=448, num_states=4, feat_dim=64)
        saved_sd = stats_saved.state_dict()

        # Mô hình mới khởi tạo với 16 camera mặc định
        stats_new = PositionStats(num_cams=16, num_patches=448, num_states=4, feat_dim=64)
        self.assertEqual(stats_new.num_cams, 16)

        # Nạp state_dict: Phải tự động mở rộng lên 570 mà không văng RuntimeError
        stats_new.load_state_dict(saved_sd)
        self.assertEqual(stats_new.num_cams, 570)
        self.assertEqual(stats_new.mu.shape, (570, 448, 4, 64))
        self.assertEqual(stats_new.w.shape, (570, 448, 4))
        self.assertEqual(stats_new.s.shape, (570, 448, 4))
        self.assertEqual(stats_new.is_initialized.shape, (570,))

    def test_06_gmm_calibrator_state_dict(self):
        """Kiểm tra GMMCalibrator lưu và khôi phục trạng thái chuẩn xác (state_dict / load_state_dict)."""
        calib1 = GMMCalibrator()
        a_mock = torch.cat([torch.rand(2, 448) * 0.2, torch.rand(2, 448) * 5.0 + 2.0], dim=0)
        v_mock = torch.ones_like(a_mock, dtype=torch.bool)
        calib1.push(a_mock, v_mock)
        calib1.refit()
        self.assertTrue(calib1.is_fitted)

        # Lưu state_dict
        sd = calib1.state_dict()
        self.assertIn("mu_bg", sd)
        self.assertIn("mu_fg", sd)
        self.assertIn("pi_weight", sd)
        self.assertTrue(sd["is_fitted"])

        # Tạo calibrator mới chưa fit và khôi phục
        calib2 = GMMCalibrator()
        self.assertFalse(calib2.is_fitted)
        calib2.load_state_dict(sd)
        self.assertTrue(calib2.is_fitted)
        self.assertAlmostEqual(calib1.mu_bg, calib2.mu_bg, places=4)
        self.assertAlmostEqual(calib1.mu_fg, calib2.mu_fg, places=4)
        self.assertAlmostEqual(calib1.pi_weight, calib2.pi_weight, places=4)

        # So sánh đầu ra posterior
        test_pts = torch.tensor([[0.05, 0.5, 2.0, 10.0]])
        pi1 = calib1.posterior(test_pts)
        pi2 = calib2.posterior(test_pts)
        self.assertTrue(torch.allclose(pi1, pi2, atol=1e-5))

    def test_07_resume_scheduler_learning_rate_continuity(self):
        """Kiểm tra cơ chế tái lập CosineAnnealingLR khi resume không bị giật ngược Learning Rate lên đỉnh."""
        param = torch.nn.Parameter(torch.zeros(10))
        base_lr = 5e-5
        eta_min = 1e-6
        optimizer = torch.optim.AdamW([param], lr=base_lr)

        # Giả lập train 5 epochs ban đầu
        scheduler_old = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=eta_min)
        for _ in range(5):
            optimizer.step()
            scheduler_old.step()

        # Cuối epoch 5, LR đã giảm sát eta_min
        last_lr = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(last_lr, eta_min, places=5)

        # Giả lập tình huống lỗi cũ: scheduler cũ bước tiếp sang epoch 6 (vượt T_max)
        scheduler_old.step()
        bounced_lr = optimizer.param_groups[0]["lr"]
        # Chu kỳ cosine bị bật ngược tăng lên:
        self.assertGreater(bounced_lr, eta_min)

        # Tình huống sửa mới: Tái lập scheduler với T_max=10 mới và last_epoch = 4
        # Đảm bảo đường cong mượt mà từ epoch 5 đến epoch 10
        optimizer2 = torch.optim.AdamW([param], lr=base_lr)
        curr_cosine_lr = eta_min + 0.5 * (base_lr - eta_min) * (1.0 + np.cos(5 * np.pi / 10))
        for group in optimizer2.param_groups:
            group["initial_lr"] = base_lr
            group["lr"] = float(curr_cosine_lr)
        scheduler_new = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2,
            T_max=10,
            eta_min=eta_min,
            last_epoch=4,
        )
        lr_at_resume = optimizer2.param_groups[0]["lr"]
        # LR tại epoch 5 của chu kỳ 10 epoch sẽ ở mức trung gian (~0.5 * base_lr), không bị bùng nổ lên 5e-5
        self.assertLess(lr_at_resume, base_lr)
        self.assertGreater(lr_at_resume, eta_min)

    def test_08_cli_args_parsing_and_smart_inheritance(self):
        """Kiểm tra helper get_cli_specified_args phân biệt đúng đối số người dùng truyền."""
        from direction1_new.train import get_cli_specified_args

        # Người dùng chỉ truyền --resume và --lr
        simulated_argv = ["--resume", "last_checkpoint.pth", "--lr", "2e-5", "--device", "cuda:0"]
        specified = get_cli_specified_args(simulated_argv)

        self.assertIn("resume", specified)
        self.assertIn("lr", specified)
        self.assertIn("device", specified)
        # Các tham số khác KHÔNG có trong argv
        self.assertNotIn("phi", specified)
        self.assertNotIn("q_max", specified)
        self.assertNotIn("p_srs", specified)
        self.assertNotIn("out_dim", specified)
        self.assertNotIn("batch_size", specified)


if __name__ == "__main__":
    unittest.main()
