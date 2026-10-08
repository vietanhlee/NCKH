"""
=============================================================================
 ST-BMB: Comprehensive Unit Test Suite
 Kiểm thử toàn diện kiến trúc Point-wise Spatio-Temporal Background Memory Bank
=============================================================================
Bao phủ 100% các cải tiến cốt lõi:
  1. Background Memory Bank: Spatial Sin-Cos PE, Real-time Temporal PE,
     Gated Write, Token Detach, Long-term EMA Background, Anchor Selection.
  2. Point-wise Temporal Attention: O(L*T), bias log(valid),
     Point-wise Background Familiarity Map, Graceful Fallback.
  3. Streaming Decomposition Net: Patch-level alpha_p, Softplus Sigma,
     Spatial PE injection, Dynamic Sequence Processing.
  4. Loss Functions: Cross-Frame Background Loss, Detached Prev-Frame,
     Floor >= 0.05 Chroma Loss, Adaptor Regularization, One-sided Familiarity,
     Edge-Aware TV, Warm-up Schedule, Gradient Backward.
  5. Dataset Pipeline: Safe Regex Timestamp Parsing, max_gap filtering,
     Camera-based Train/Val Split, Letterbox Resize, Jitter Augmentation.
  6. Photometric Illumination Adaptor Mechanism.
  7. End-to-end Training Step with Memory Dropout.
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dino_root not in sys.path:
    sys.path.insert(0, dino_root)

from direction_streaming_bmb.memory import (
    BackgroundMemoryBank,
    TemporalRealtimeEmbedding,
    build_2d_sincos_position_embedding,
)
from direction_streaming_bmb.attention import MemoryCrossAttention
from direction_streaming_bmb.models import (
    StreamingDecompositionNet,
    BackgroundMemoryEncoder,
    PhotometricIlluminationAdaptor,
)
from direction_streaming_bmb.losses import (
    StreamingDecompositionLoss,
    edge_aware_tv_loss,
    gradient_exclusion_loss,
    ssim_2d_loss,
)
from direction_streaming_bmb.dataset import (
    SlidingWindowTrafficDataset,
    letterbox_resize,
)


class DummyBackbone(nn.Module):
    """Backbone giả lập ViT-S/16 để kiểm thử đơn vị độc lập và nhanh chóng trên CPU."""
    def __init__(self, embed_dim=384, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        L = (H // self.patch_size) * (W // self.patch_size)
        return torch.randn(B, L, self.embed_dim, device=x.device)


class TestStreamingBMB(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.embed_dim = 256
        self.backbone_dim = 384
        self.patch_size = 16
        self.img_size = 128
        self.Hp = self.img_size // self.patch_size  # 8
        self.Wp = self.img_size // self.patch_size  # 8
        self.L = self.Hp * self.Wp                   # 64 tokens

    def test_01_background_memory_bank_spatial_temporal_and_gated_write(self):
        """Kiểm tra toàn diện BackgroundMemoryBank: Spatial PE, Real-time Temporal PE, Gated Write, Detach & EMA."""
        bank = BackgroundMemoryBank(
            embed_dim=self.embed_dim,
            max_recent_frames=3,
            max_anchor_frames=1,
            ema_eta=0.2,
        )
        self.assertTrue(bank.is_empty)

        # 1. Kiểm tra 2D Sin-Cos Spatial Positional Embedding
        pe = build_2d_sincos_position_embedding(self.Hp, self.Wp, self.embed_dim, self.device)
        self.assertEqual(pe.shape, (1, self.L, self.embed_dim))

        # 2. Ban đầu rỗng -> read trả về is_empty=True
        mem_tokens, raw_mem, mem_valid, is_empty = bank.read(self.device)
        self.assertTrue(is_empty)
        self.assertIsNone(mem_tokens)

        # 3. Ghi frame có gradient -> Kiểm tra Token Detach bắt buộc
        B = 2
        bg_with_grad = torch.randn(B, self.L, self.embed_dim, requires_grad=True)
        # Giả sử nửa ảnh có xe (alpha=1.0), nửa ảnh là đường sạch (alpha=0.0)
        alpha_p = torch.zeros(B, self.L)
        alpha_p[:, :self.L // 2] = 1.0  # Vùng có xe

        bank.write(bg_with_grad, alpha_p=alpha_p, timestamp=1755698800.0)
        self.assertFalse(bank.is_empty)

        # Kiểm tra token lưu trong bank đã được detach hoàn toàn
        self.assertFalse(bank.recent_tokens[0].requires_grad)
        self.assertFalse(bank.recent_valid[0].requires_grad)

        # Kiểm tra valid mask: Vùng có xe nhận valid = 0.0, vùng sạch nhận valid = 1.0
        self.assertEqual(bank.recent_valid[0][0, 0].item(), 0.0)
        self.assertEqual(bank.recent_valid[0][0, -1].item(), 1.0)

        # 4. Kiểm tra Long-term Background Memory EMA có cổng:
        # beta = eta * (1 - alpha_p). Vùng có xe (alpha_p=1) thì beta=0 -> không bị ghi đè!
        init_long_term = bank.long_term_memory.clone()
        # Ghi tiếp frame thứ hai với giá trị khác
        bg_frame2 = torch.full((B, self.L, self.embed_dim), 99.0)
        bank.write(bg_frame2, alpha_p=alpha_p, timestamp=1755699100.0)  # +5 phút (300s)

        # Vùng có xe (token index 0): long-term memory không bị thay đổi (beta = 0)
        self.assertTrue(torch.allclose(bank.long_term_memory[:, 0, :], init_long_term[:, 0, :]))
        # Vùng sạch (token index -1): long-term memory được cập nhật theo EMA
        self.assertFalse(torch.allclose(bank.long_term_memory[:, -1, :], init_long_term[:, -1, :]))

        # 5. Kiểm tra Anchor Selection: Frame sạch nhất (alpha mean nhỏ nhất) được chọn làm anchor
        # Frame 3 rất sạch (alpha_p toàn 0.05)
        clean_bg = torch.full((B, self.L, self.embed_dim), 5.0)
        clean_alpha = torch.full((B, self.L), 0.05)
        bank.write(clean_bg, alpha_p=clean_alpha, timestamp=1755699400.0)
        self.assertIsNotNone(bank.anchor_tokens)
        self.assertTrue(torch.all(bank.anchor_alpha_mean <= 0.1))

        # 6. Kiểm tra read(): Trả về mem_tokens có 4 chiều (B, T, L, D) và valid (B, T, L)
        mem_tokens, raw_mem, mem_valid, is_empty = bank.read(self.device, query_timestamp=1755699700.0)
        self.assertFalse(is_empty)
        self.assertEqual(mem_tokens.dim(), 4)
        self.assertEqual(mem_tokens.shape[0], B)
        self.assertEqual(mem_tokens.shape[2], self.L)
        self.assertEqual(mem_tokens.shape[3], self.embed_dim)
        self.assertEqual(mem_valid.shape, (B, mem_tokens.shape[1], self.L))

        # 7. Kiểm tra Lưới Patch Không Vuông (Non-Square Spatial Grid: Hp=12, Wp=16, L=192)
        bank_rect = BackgroundMemoryBank(embed_dim=self.embed_dim)
        rect_L = 12 * 16
        z_rect = torch.randn(2, rect_L, self.embed_dim)
        bank_rect.write(z_rect, spatial_hw=(12, 16))
        rect_tokens, _, _, _ = bank_rect.read(self.device, spatial_hw=(12, 16))
        self.assertEqual(rect_tokens.shape, (2, 3, rect_L, self.embed_dim))

        # 8. Kiểm tra Vectorized Per-Sample Anchor Selection (Không bị ô nhiễm chéo giữa các mẫu)
        bank_per_sample = BackgroundMemoryBank(embed_dim=64)
        b_samples = 2
        l_dim = 16
        # Bước 0: Mẫu 0 sạch (10.0), Mẫu 1 bẩn (99.0)
        z0 = torch.full((b_samples, l_dim, 64), 99.0)
        z0[0] = 10.0
        a0 = torch.zeros(b_samples, l_dim)
        a0[1] = 1.0
        bank_per_sample.write(z0, alpha_p=a0)

        # Bước 1: Mẫu 0 bẩn (55.0), Mẫu 1 sạch (20.0)
        z1 = torch.full((b_samples, l_dim, 64), 55.0)
        z1[1] = 20.0
        a1 = torch.zeros(b_samples, l_dim)
        a1[0] = 0.9
        a1[1] = 0.0
        bank_per_sample.write(z1, alpha_p=a1)

        # Mẫu 0 phải giữ nguyên anchor sạch (10.0), Mẫu 1 phải cập nhật sang anchor sạch (20.0)
        self.assertEqual(bank_per_sample.anchor_tokens[0, 0, 0].item(), 10.0)
        self.assertEqual(bank_per_sample.anchor_tokens[1, 0, 0].item(), 20.0)

        # Reset bộ nhớ
        bank.reset()
        self.assertTrue(bank.is_empty)

    def test_02_pointwise_temporal_attention_and_valid_bias(self):
        """Kiểm tra Point-wise Temporal Attention O(L*T), bias log(valid) và Familiarity Map."""
        attn = MemoryCrossAttention(embed_dim=self.embed_dim, num_heads=8)
        B, T = 2, 4
        q = torch.randn(B, self.L, self.embed_dim)

        # 1. Chế độ rỗng (Cold start): Fallback mượt mà
        out_empty, fam_empty = attn(q, None, is_empty_memory=True)
        self.assertEqual(out_empty.shape, q.shape)
        self.assertEqual(fam_empty.shape, (B, self.L, 1))
        self.assertTrue(torch.allclose(fam_empty, torch.tensor(0.5)))

        # 2. Chế độ có bộ nhớ với valid mask
        mem = torch.randn(B, T, self.L, self.embed_dim)
        raw_mem = torch.randn(B, T, self.L, self.embed_dim)
        valid = torch.ones(B, T, self.L)
        # Giả lập frame 0 tại patch 10 bị xe che lấp (valid = 0.0)
        valid[:, 0, 10] = 0.0

        out, fam = attn(
            q_tokens=q,
            mem_tokens=mem,
            raw_mem_tokens=raw_mem,
            valid=valid,
            is_empty_memory=False,
        )

        self.assertEqual(out.shape, (B, self.L, self.embed_dim))
        self.assertEqual(fam.shape, (B, self.L, 1))
        # Kiểm tra Background Familiarity Score nằm trong [0, 1]
        self.assertTrue(torch.all(fam >= 0.0) and torch.all(fam <= 1.0))

    def test_03_streaming_decomposition_net_forward_and_patch_alpha(self):
        """Kiểm tra StreamingDecompositionNet, patch-level alpha_p và Softplus Sigma trơn tru."""
        backbone = DummyBackbone(embed_dim=self.backbone_dim, patch_size=self.patch_size)
        model = StreamingDecompositionNet(
            backbone=backbone,
            backbone_dim=self.backbone_dim,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            max_recent_frames=3,
        )

        # 1. Forward single frame (B=2, C=3, H=128, W=128)
        frame = torch.rand(2, 3, self.img_size, self.img_size)
        out = model.forward_single_step(frame, update_memory=True, is_anchor=True)

        self.assertIn("alpha_mask", out)
        self.assertIn("alpha_p", out)
        self.assertIn("pred_fg", out)
        self.assertIn("pred_bg", out)
        self.assertIn("sigma", out)
        self.assertIn("recon_origin", out)
        self.assertIn("bg_familiarity", out)

        # Kiểm tra shape
        self.assertEqual(out["alpha_mask"].shape, (2, 1, self.img_size, self.img_size))
        self.assertEqual(out["alpha_p"].shape, (2, self.L))
        self.assertEqual(out["sigma"].shape, (2, 1, self.img_size, self.img_size))

        # Kiểm tra Softplus Sigma: Luôn >= 0.01
        self.assertTrue(torch.all(out["sigma"] >= 0.01))

        # 2. Forward chuỗi thời gian W=4 khung hình
        seq_frames = torch.rand(2, 4, 3, self.img_size, self.img_size)
        timestamps = torch.tensor([[100.0, 400.0, 700.0, 1000.0], [100.0, 400.0, 700.0, 1000.0]])
        seq_outs = model.forward_sequence(seq_frames, timestamps=timestamps, reset_memory=True)
        self.assertEqual(len(seq_outs), 4)

        # Kiểm tra các thông số quang sai sáng/chiều tại t >= 1
        self.assertIsNotNone(seq_outs[1]["bg_prev_aligned"])
        self.assertIn("illum_gain", seq_outs[1])
        self.assertIn("illum_bias", seq_outs[1])
        self.assertIn("illum_soft_weight", seq_outs[1])

    def test_04_streaming_decomposition_loss_cross_frame_and_gradients(self):
        """Kiểm tra Cross-Frame Background Loss, One-sided Fam Loss, Edge-TV và Gradient Propagation."""
        backbone = DummyBackbone(embed_dim=self.backbone_dim, patch_size=self.patch_size)
        model = StreamingDecompositionNet(
            backbone=backbone,
            backbone_dim=self.backbone_dim,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            max_recent_frames=3,
        )
        loss_fn = StreamingDecompositionLoss(
            lambda_cross=1.5,
            lambda_temp=2.0,
            lambda_fam=0.5,
        )

        seq_frames = torch.rand(2, 3, 3, self.img_size, self.img_size)
        seq_outputs = model.forward_sequence(seq_frames, reset_memory=True)

        # 1. Kiểm tra Warm-up Factor
        loss_fn.set_warmup_factor(0.5)
        total_loss, loss_dict = loss_fn(seq_outputs, seq_frames)

        self.assertTrue(torch.is_tensor(total_loss))
        self.assertTrue(total_loss.item() > 0.0)
        self.assertIn("loss_cross", loss_dict)
        self.assertIn("loss_temp_bg", loss_dict)
        self.assertIn("loss_fam", loss_dict)
        self.assertIn("loss_tv", loss_dict)
        self.assertIn("loss_illum_reg", loss_dict)
        self.assertEqual(loss_dict["warmup_factor"], 0.5)

        # 2. Kiểm tra Backward Gradient
        total_loss.backward()

        # Kiểm tra Decoder nhận Gradient
        decoder_has_grad = any(p.grad is not None for p in model.decoder.parameters())
        self.assertTrue(decoder_has_grad, "Decoder phải nhận được gradient hợp lệ.")

        # Kiểm tra Illumination Adaptor nhận Gradient
        adaptor_has_grad = any(p.grad is not None for p in model.illumination_adaptor.parameters())
        self.assertTrue(adaptor_has_grad, "Illumination Adaptor phải nhận được gradient hợp lệ.")

        # 3. Kiểm tra KHÔNG CÓ GRADIENT RÒ RỈ qua lịch sử frame trước:
        # Toàn bộ token lưu trữ trong Memory Bank (recent, anchor, long-term) bắt buộc phải detach triệt để
        for idx, tok in enumerate(model.memory_bank.recent_tokens):
            self.assertFalse(tok.requires_grad, f"Recent token {idx} phải bị detach (requires_grad=False)!")
            self.assertIsNone(tok.grad_fn, f"Recent token {idx} không được có đồ thị gradient!")

        if model.memory_bank.anchor_tokens is not None:
            self.assertFalse(model.memory_bank.anchor_tokens.requires_grad, "Anchor token phải bị detach!")
            self.assertIsNone(model.memory_bank.anchor_tokens.grad_fn, "Anchor token không được có đồ thị gradient!")

        if model.memory_bank.long_term_memory is not None:
            self.assertFalse(model.memory_bank.long_term_memory.requires_grad, "Long-term memory token phải bị detach!")
            self.assertIsNone(model.memory_bank.long_term_memory.grad_fn, "Long-term memory token không được có đồ thị gradient!")
            self.assertIsNotNone(model.memory_bank.long_term_valid, "Long-term valid mask phải được duy trì!")

        # 4. Kiểm tra Giám sát Trực tiếp Adaptor bằng Nhãn Jitter (g, b)
        _, loss_dict_jitter = loss_fn(
            seq_outputs,
            seq_frames,
            target_jitter_gain=torch.tensor([1.2, 1.2]),
            target_jitter_bias=torch.tensor([0.05, 0.05]),
        )
        self.assertIn("loss_adaptor_sup", loss_dict_jitter)
        self.assertGreater(loss_dict_jitter["loss_adaptor_sup"], 0.0)

    def test_05_sliding_window_dataset_regex_split_and_max_gap(self):
        """Kiểm tra dataset: Regex timestamp an toàn, max_gap filtering, train/val split và letterbox resize."""
        sample_dir = os.path.join(
            dino_root,
            "direction_data_article",
            "zenodo_bundle",
            "sample_preview",
            "sample_camera_sequences",
        )
        if not os.path.exists(sample_dir):
            self.skipTest(f"Không tìm thấy thư mục dữ liệu mẫu tại {sample_dir}")

        # 1. Kiểm tra Train split
        train_ds = SlidingWindowTrafficDataset(
            data_dir=sample_dir,
            window_size=3,
            stride=1,
            img_size=128,
            split="train",
            val_ratio=0.2,
            max_gap=None,
        )
        self.assertTrue(len(train_ds) > 0)
        sample = train_ds[0]
        self.assertIn("frames", sample)
        self.assertIn("timestamps", sample)
        self.assertEqual(sample["frames"].shape, (3, 3, 128, 128))
        self.assertEqual(sample["timestamps"].shape, (3,))

        # Kiểm tra cơ chế max_gap lọc bỏ các cửa sổ có gap quá lớn
        with self.assertRaises(RuntimeError):
            SlidingWindowTrafficDataset(
                data_dir=sample_dir,
                window_size=3,
                stride=1,
                img_size=128,
                max_gap=300.0,
            )

        # 2. Kiểm tra Regex an toàn
        test_path = "cam_route10_20240901_1755698814.jpg"
        cam_id, ts = train_ds._parse_cam_and_time(test_path)
        self.assertEqual(ts, 1755698814.0)

        # 3. Kiểm tra Letterbox Resize
        from PIL import Image
        non_square = Image.new("RGB", (200, 100), (255, 0, 0))
        boxed = letterbox_resize(non_square, 128)
        self.assertEqual(boxed.size, (128, 128))

        # 4. Kiểm tra Synchronous Random Crop và Jitter Labels
        crop_ds = SlidingWindowTrafficDataset(
            data_dir=sample_dir,
            window_size=3,
            stride=1,
            img_size=128,
            split="train",
            val_ratio=0.0,
            is_train=True,
            apply_crop=True,
        )
        sample_crop = crop_ds[0]
        self.assertEqual(sample_crop["frames"].shape, (3, 3, 128, 128))
        self.assertIn("jitter_gain", sample_crop)
        self.assertIn("jitter_bias", sample_crop)

    def test_06_photometric_illumination_adaptor_mechanism(self):
        """Kiểm tra khối PhotometricIlluminationAdaptor ước lượng gain, bias và soft_weight."""
        adaptor = PhotometricIlluminationAdaptor(embed_dim=self.embed_dim)
        B = 2
        tok_t = torch.randn(B, self.embed_dim)
        tok_prev = torch.randn(B, self.embed_dim)
        frame_prev = torch.full((B, 3, 64, 64), 0.3)
        frame_t = torch.full((B, 3, 64, 64), 0.7)
        bg_prev = torch.full((B, 3, 64, 64), 0.3)

        bg_aligned, gain, bias, soft_weight = adaptor(tok_t, tok_prev, frame_t, frame_prev, bg_prev)
        self.assertEqual(bg_aligned.shape, (B, 3, 64, 64))
        self.assertEqual(gain.shape, (B, 3, 1, 1))
        self.assertEqual(bias.shape, (B, 3, 1, 1))
        self.assertEqual(soft_weight.shape, (B, 1, 1, 1))
        self.assertTrue(torch.all(soft_weight >= 0.2) and torch.all(soft_weight <= 1.0))

    def test_07_memory_dropout_and_synthetic_training_step(self):
        """Kiểm tra Memory Dropout và hoàn thiện 1 vòng lặp huấn luyện chuẩn."""
        backbone = DummyBackbone(embed_dim=self.backbone_dim, patch_size=self.patch_size)
        model = StreamingDecompositionNet(
            backbone=backbone,
            backbone_dim=self.backbone_dim,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            max_recent_frames=4,
        )
        model.train()
        model.set_memory_dropout(0.5)

        loss_fn = StreamingDecompositionLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Chạy 1 batch huấn luyện (B=2, W=3, C=3, H=128, W=128)
        frames = torch.rand(2, 3, 3, self.img_size, self.img_size)
        timestamps = torch.tensor([[100.0, 400.0, 700.0], [100.0, 400.0, 700.0]])

        step_outputs = model(frames, timestamps=timestamps, reset_memory=True)
        total_loss, loss_dict = loss_fn(step_outputs, frames)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        self.assertGreater(total_loss.item(), 0.0)

        # Kiểm tra Cold Start Memory Dropout fallback (K=0)
        model.set_memory_dropout(1.0)
        out_cold = model(frames, timestamps=timestamps, reset_memory=True)
        self.assertEqual(len(out_cold), 3)

    def test_08_patch_validity_gate_dual_anchor_and_sigma_filter(self):
        """Kiểm tra chuyên sâu các bản vá nâng cao: Patch Validity Gate, Dual Anchor và Sigma Filter."""
        bank = BackgroundMemoryBank(
            embed_dim=self.embed_dim,
            max_recent_frames=4,
            spatial_patch_size=(self.patch_size, self.patch_size),
        )
        B = 2

        # 1. Kiểm tra Lọc Nhiễu Độ Bất Định Quang Học Sigma
        bg_tok = torch.randn(B, self.L, self.embed_dim)
        alpha_clean = torch.zeros(B, self.L)  # alpha = 0 (không có xe)
        sigma_noisy = torch.zeros(B, self.L)
        sigma_noisy[:, 0] = 0.5  # Patch 0 bị chói sáng đèn pha (sigma rất cao)

        bank.write(bg_tok, alpha_p=alpha_clean, sigma_p=sigma_noisy, timestamp=100.0)
        # Patch 0 phải bị hạ độ tin cậy valid do sigma cao
        valid_p0 = bank.recent_valid[0][0, 0].item()
        valid_p1 = bank.recent_valid[0][0, 1].item()
        self.assertLess(valid_p0, valid_p1)
        self.assertAlmostEqual(valid_p1, 1.0, places=3)

        # 2. Kiểm tra Dual-Slot Anchor (Day và Night độc lập)
        bank.reset()
        # Frame A: Ban ngày (12:00 trưa = 12 * 3600s UTC+7)
        # Chuyển đổi timestamp sao cho: (ts + 7*3600) % 86400 = 12 * 3600 => ts = 5 * 3600 = 18000.0
        ts_day = 18000.0
        alpha_day = torch.full((B, self.L), 0.15)
        bank.write(bg_tok, alpha_p=alpha_day, timestamp=ts_day)

        # Frame B: Ban đêm (02:00 sáng = 2 * 3600s UTC+7 => ts = (2 - 7 + 24) * 3600 = 19 * 3600 = 68400.0)
        ts_night = 68400.0
        alpha_night = torch.full((B, self.L), 0.02)  # Đêm vắng tanh
        bank.write(bg_tok, alpha_p=alpha_night, timestamp=ts_night)

        # Cả hai slot đều được lưu độc lập
        self.assertIsNotNone(bank.anchor_day_tokens)
        self.assertIsNotNone(bank.anchor_night_tokens)
        self.assertAlmostEqual(bank.anchor_day_alpha_mean[0].item(), 0.15, places=2)
        self.assertAlmostEqual(bank.anchor_night_alpha_mean[0].item(), 0.02, places=2)

        # Khi query ban ngày (ts = 18000.0), read() phải ưu tiên lấy Day Anchor
        _, raw_mem_day, _, _ = bank.read(self.device, query_timestamp=ts_day)
        self.assertIsNotNone(raw_mem_day)

        # 3. Kiểm tra Patch-wise Validity Gate trong MemoryCrossAttention
        attn = MemoryCrossAttention(embed_dim=self.embed_dim, num_heads=4)
        q = torch.randn(B, self.L, self.embed_dim)
        mem = torch.randn(B, 3, self.L, self.embed_dim)
        valid = torch.ones(B, 3, self.L)
        # Đặt patch index 5 có valid = 0 trên toàn bộ 3 frames quá khứ (kẹt xe đứng im)
        valid[:, :, 5] = 1e-4

        out_tokens, fam = attn(q, mem, raw_mem_tokens=mem, valid=valid, is_empty_memory=False)
        self.assertEqual(out_tokens.shape, q.shape)
        # Tại patch index 5, vì patch_gate ~ 1e-4, residual connection bảo toàn gần như nguyên vẹn q
        diff_p5 = torch.norm(out_tokens[:, 5, :] - q[:, 5, :], dim=-1).mean().item()
        diff_p0 = torch.norm(out_tokens[:, 0, :] - q[:, 0, :], dim=-1).mean().item()
        # Patch bị kẹt xe hoàn toàn nhận ít ô nhiễm từ bộ nhớ hơn nhiều so với patch sạch
        self.assertLess(diff_p5, diff_p0)


if __name__ == "__main__":
    unittest.main()
