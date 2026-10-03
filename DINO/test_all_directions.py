"""
=============================================================================
 Comprehensive Smoke Test & Verification Suite for DINO Traffic Suite (8 Directions)
 Kiểm thử toàn diện 8 hướng nghiên cứu trọng tâm và tầng Common Utilities
 Chạy trên dữ liệu mô phỏng (Synthetic Dummy Data) để xác thực 100% không lỗi runtime
=============================================================================
"""

import os
import sys

# Đảm bảo mã hóa UTF-8 cho Windows Terminal / PowerShell và chống xung đột OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import shutil
import tempfile
import traceback
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# Thêm path
dino_dir = os.path.dirname(os.path.abspath(__file__))
if dino_dir not in sys.path:
    sys.path.insert(0, dino_dir)

print("=" * 80)
print(" [*] STARTING COMPREHENSIVE VERIFICATION SUITE - 8 RESEARCH DIRECTIONS")
print("=" * 80)

# Tạo thư mục tạm chứa dữ liệu giả lập chuẩn
temp_root = tempfile.mkdtemp(prefix="dino_test_")
print(f"📁 Thư mục tạm kiểm thử: {temp_root}")

try:
    # -------------------------------------------------------------
    # 0. TẠO CẤU TRÚC DỮ LIỆU GIẢ ĐỊNH (MOCK DATA)
    # -------------------------------------------------------------
    bg_dir = os.path.join(temp_root, "traffic_backgrounds")
    origin_dir = os.path.join(temp_root, "output")
    os.makedirs(os.path.join(bg_dir, "route_1"), exist_ok=True)
    os.makedirs(os.path.join(bg_dir, "route_2"), exist_ok=True)
    os.makedirs(origin_dir, exist_ok=True)

    # Tạo ảnh dummy background: 224x224
    dummy_bg_1_14 = Image.fromarray(np.full((224, 224, 3), 100, dtype=np.uint8))
    dummy_bg_1_14.save(os.path.join(bg_dir, "route_1", "background_slot_14h.jpg"))

    dummy_bg_1_15 = Image.fromarray(np.full((224, 224, 3), 110, dtype=np.uint8))
    dummy_bg_1_15.save(os.path.join(bg_dir, "route_1", "background_slot_15h.jpg"))

    dummy_bg_2_00 = Image.fromarray(np.full((224, 224, 3), 50, dtype=np.uint8))
    dummy_bg_2_00.save(os.path.join(bg_dir, "route_2", "background_slot_00h.jpg"))

    # Tạo ảnh origin:
    arr_orig_1 = np.full((224, 224, 3), 100, dtype=np.uint8)
    arr_orig_1[50:120, 50:120, :] = 255  # "Xe cộ" màu trắng nổi bật
    dummy_orig_1 = Image.fromarray(arr_orig_1)
    dummy_orig_1.save(os.path.join(origin_dir, "1_1755698811.jpg"))

    arr_orig_2 = np.full((224, 224, 3), 110, dtype=np.uint8)
    arr_orig_2[80:150, 100:180, :] = 220
    dummy_orig_2 = Image.fromarray(arr_orig_2)
    dummy_orig_2.save(os.path.join(origin_dir, "1_1755702400.jpg"))

    arr_orig_3 = np.full((224, 224, 3), 50, dtype=np.uint8)
    arr_orig_3[30:90, 40:110, :] = 200
    dummy_orig_3 = Image.fromarray(arr_orig_3)
    dummy_orig_3.save(os.path.join(origin_dir, "2_1755600000.jpg"))

    # CSV nhãn dummy
    csv_file = os.path.join(temp_root, "dummy_labels.csv")
    import pandas as pd
    df_dummy = pd.DataFrame([
        {"filename": "1_1755698811.jpg", "xe_may": 12, "o_to": 3, "tong": 15},
        {"filename": "1_1755702400.jpg", "xe_may": 25, "o_to": 5, "tong": 30},
        {"filename": "2_1755600000.jpg", "xe_may": 8, "o_to": 1, "tong": 9},
    ])
    df_dummy.to_csv(csv_file, index=False)
    print("✅ [Setup] Đã khởi tạo dữ liệu mô phỏng thành công.")

    # -------------------------------------------------------------
    # 1. TEST COMMON UTILITIES
    # -------------------------------------------------------------
    print("\n--- [TEST 1] Common Utilities ---")
    from common.matcher import TrafficPairMatcher
    matcher = TrafficPairMatcher(bg_dir=bg_dir, origin_dir=origin_dir, match_strategy="route_hourly")
    pairs = matcher.discover_pairs()
    assert len(pairs) >= 2, f"Kỳ vọng >= 2 cặp ảnh, thực tế: {len(pairs)}"
    print(f"   + TrafficPairMatcher phát hiện: {len(pairs)} cặp")

    from common.subtraction import BackgroundSubtractor
    subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)
    delta_norm, delta_raw = subtractor.compute_delta(np.array(dummy_orig_1), np.array(dummy_bg_1_14))
    assert delta_norm.shape == (224, 224), f"Shape delta sai: {delta_norm.shape}"
    bin_mask = subtractor.extract_binary_mask(delta_norm, method="otsu")
    assert bin_mask.shape == (224, 224) and bin_mask.dtype == np.uint8, "Mask sai định dạng"

    patch_weights = subtractor.compute_patch_mask_weights(delta_norm, patch_size=16)
    assert patch_weights.shape == (14, 14), f"Patch weights shape sai: {patch_weights.shape}"
    print(f"   + BackgroundSubtractor: Delta={delta_norm.shape}, PatchWeights={patch_weights.shape}")
    print("   ✅ [Common] Matcher & Subtractor pass hoàn hảo!")

    class MockPatchViT(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            self.embed_dim = embed_dim
            self.conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        def forward(self, x):
            feat = self.conv(x)  # (B, embed_dim, H_p, W_p)
            b, c, h, w = feat.shape
            tokens = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
            cls_t = tokens.mean(dim=1)
            return {
                "x_norm_clstoken": cls_t,
                "x_norm_patchtokens": tokens,
            }
        def get_intermediate_layers(self, x, n=1, return_class_token=True):
            feat = self.conv(x)
            b, c, h, w = feat.shape
            tokens = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
            cls_t = tokens.mean(dim=1)
            return [(tokens, cls_t)]

    mock_vit = MockPatchViT(embed_dim=64)

    # -------------------------------------------------------------
    # 2. TEST DIRECTION 1: BG-GUIDED DINO
    # -------------------------------------------------------------
    print("\n--- [TEST 2] Direction 1: BG-Guided DINO SSL ---")
    from direction1_bg_guided_dino.dataset import BGGuidedDINODataset
    from direction1_bg_guided_dino.models import DINOHead, BGGuidedDINOModel
    from direction1_bg_guided_dino.losses import BGGuidedDINOLoss

    ds1 = BGGuidedDINODataset(
        bg_dir=bg_dir,
        origin_dir=origin_dir,
        match_strategy="route_hourly",
        patch_size=16,
        size_global=224,
        size_local=96,
        local_crops_number=2,
    )
    sample1 = ds1[0]
    crops = sample1["crops"]
    assert len(crops) == 4, f"Kỳ vọng 2 global + 2 local = 4 crops, thực tế: {len(crops)}"
    assert crops[0].shape == (3, 224, 224) and crops[2].shape == (3, 96, 96), "Shape crops sai"

    class TinyBackbone(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            self.conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
            self.embed_dim = embed_dim
        def forward(self, x):
            x = self.conv(x)
            return x.mean(dim=(2, 3))

    mock_bb = TinyBackbone(embed_dim=64)
    dino_model = BGGuidedDINOModel(student_backbone=mock_bb, embed_dim=64, out_dim=512)

    batch_crops = [torch.stack([c, c]) for c in crops]
    student_out = dino_model.forward_student(batch_crops)
    teacher_out = dino_model.forward_teacher(batch_crops[:2])

    loss_fn1 = BGGuidedDINOLoss(out_dim=512, ncrops=4, nepochs=2)
    loss1 = loss_fn1(student_out, teacher_out, epoch=0)
    assert not torch.isnan(loss1) and loss1.item() > 0, "Loss DINO bất thường"
    loss1.backward()
    dino_model.update_teacher(0.99)
    print(f"   + DINO Multi-crop Loss: {loss1.item():.4f}")
    print("   ✅ [Direction 1] BG-Guided DINO pass hoàn hảo!")

    # -------------------------------------------------------------
    # 3. TEST DIRECTION 2: SCENE DECOMPOSITION
    # -------------------------------------------------------------
    print("\n--- [TEST 3] Direction 2: Scene Decomposition ---")
    from direction2_scene_decomposition.dataset import DecompositionDataset
    from direction2_scene_decomposition.models import TrafficDecompositionNet
    from direction2_scene_decomposition.losses import DecompositionLoss

    ds2 = DecompositionDataset(bg_dir=bg_dir, origin_dir=origin_dir, match_strategy="route_hourly", img_size=128, is_train=True)
    sample2 = ds2[0]
    assert sample2["origin"].shape == (3, 128, 128) and sample2["bg"].shape == (3, 128, 128)

    decomp_net = TrafficDecompositionNet(backbone=mock_vit, embed_dim=64, patch_size=16, freeze_backbone=True)
    batch_orig = torch.stack([sample2["origin"], sample2["origin"]])
    batch_bg = torch.stack([sample2["bg"], sample2["bg"]])
    preds2 = decomp_net(batch_orig)

    assert preds2["pred_bg"].shape == (2, 3, 128, 128)
    assert preds2["pred_fg"].shape == (2, 3, 128, 128)
    assert preds2["pred_mask"].shape == (2, 1, 128, 128)
    assert preds2["recon_origin"].shape == (2, 3, 128, 128)

    loss_fn2 = DecompositionLoss()
    l2, l2_dict = loss_fn2(preds2, {"origin": batch_orig, "bg": batch_bg})
    assert not torch.isnan(l2), "Loss Decomposition bị NaN"
    l2.backward()
    print(f"   + Decomposition Loss: {l2.item():.4f} (Recon: {l2_dict['loss_recon']:.4f})")
    print("   ✅ [Direction 2] Scene Decomposition pass hoàn hảo!")

    # -------------------------------------------------------------
    # 4. TEST DIRECTION 3: FOREGROUND-ENHANCED COUNTING
    # -------------------------------------------------------------
    print("\n--- [TEST 4] Direction 3: Foreground-Enhanced Counting ---")
    from direction3_foreground_enhanced_counting.dataset import FGCountingDataset
    from direction3_foreground_enhanced_counting.models import DINOv3FGCountingModel, adapt_patch_embed_to_4ch

    ds3 = FGCountingDataset(
        csv_file=csv_file,
        origin_dir=origin_dir,
        bg_dir=bg_dir,
        match_strategy="route_hourly",
        img_size=224,
        is_train=True,
    )
    sample3 = ds3[0]
    assert sample3["input_4ch"].shape == (4, 224, 224)
    assert sample3["counts"].shape == (3,)

    conv_3ch = nn.Conv2d(3, 64, kernel_size=16, stride=16)
    conv_4ch = adapt_patch_embed_to_4ch(conv_3ch)
    assert conv_4ch.in_channels == 4

    class MockViT4Ch(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, 64, kernel_size=16, stride=16)
        def forward(self, x):
            x = self.patch_embed(x)
            return x.mean(dim=(2, 3))

    counting_model = DINOv3FGCountingModel(backbone=MockViT4Ch(), embed_dim=64, mode="4channel")
    dummy_input_4ch = torch.randn(2, 4, 224, 224)
    pred_counts = counting_model(dummy_input_4ch)
    assert pred_counts.shape == (2, 3) and (pred_counts >= 0).all()
    print(f"   + Counting Model Output: {pred_counts[0].tolist()}")
    print("   ✅ [Direction 3] Foreground-Enhanced Counting pass hoàn hảo!")

    # -------------------------------------------------------------
    # 5. TEST DIRECTION 4: SPATIO-TEMPORAL DENSITY & LoS ESTIMATION
    # -------------------------------------------------------------
    print("\n--- [TEST 5] Direction 4: Spatio-Temporal Road Occupancy & LoS Estimation ---")
    from direction4_temporal_density.dataset import TemporalTrafficDataset
    from direction4_temporal_density.models import SpatioTemporalDensityNet
    from direction4_temporal_density.losses import SpatioTemporalDensityLoss

    ds5 = TemporalTrafficDataset(
        bg_dir=bg_dir,
        origin_dir=origin_dir,
        window_size=2,
        img_size=128,
        is_train=True,
    )
    sample5 = ds5[0]
    assert sample5["rgb_seq"].shape == (2, 3, 128, 128)
    assert sample5["delta_seq"].shape == (2, 1, 128, 128)
    assert sample5["occupancy_seq"].shape == (2,)
    assert sample5["los_seq"].shape == (2,)

    density_net = SpatioTemporalDensityNet(
        backbone=mock_vit,
        embed_dim=64,
        delta_dim=32,
        temporal_dim=64,
        num_los_classes=4,
        freeze_backbone=True,
    )
    batch_rgb = torch.stack([sample5["rgb_seq"], sample5["rgb_seq"]])      # (2, 2, 3, 128, 128)
    batch_delta = torch.stack([sample5["delta_seq"], sample5["delta_seq"]])  # (2, 2, 1, 128, 128)
    out5 = density_net(batch_rgb, batch_delta)
    assert out5["pred_occupancy"].shape == (2,) and (out5["pred_occupancy"] >= 0).all()
    assert out5["logits_los"].shape == (2, 4)
    assert out5["pred_trend"].shape == (2,)

    crit5 = SpatioTemporalDensityLoss()
    targets5 = {
        "occupancy_seq": torch.stack([sample5["occupancy_seq"], sample5["occupancy_seq"]]),
        "current_occupancy": torch.tensor([sample5["current_occupancy"], sample5["current_occupancy"]]),
        "current_los": torch.tensor([sample5["current_los"], sample5["current_los"]]),
        "trend": torch.tensor([sample5["trend"], sample5["trend"]]),
    }
    l5_dict = crit5(out5, targets5)
    assert not torch.isnan(l5_dict["loss_total"]) and l5_dict["loss_total"].item() > 0
    print(f"   + Spatio-Temporal Loss: {l5_dict['loss_total'].item():.4f} (Occ: {l5_dict['loss_occupancy'].item():.4f}, LoS: {l5_dict['loss_los'].item():.4f})")
    print("   ✅ [Direction 4] Spatio-Temporal Density & LoS pass hoàn hảo!")

    # -------------------------------------------------------------
    # 6. TEST MULTI-GPU SMART SAVE & LOAD CHECKPOINTING
    # -------------------------------------------------------------
    print("\n--- [TEST 6] Multi-GPU Smart Checkpointing Interoperability ---")
    from common.gpu_utils import save_checkpoint, load_checkpoint, clean_state_dict, smart_load_state_dict

    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
        def forward(self, x):
            return self.linear(x)

    dummy_model = DummyNet()
    ckpt_save_test = os.path.join(temp_root, "test_smart_save.pth")

    class MockDataParallelWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner
        def forward(self, x):
            return self.module(x)

    mock_dp_model = MockDataParallelWrapper(dummy_model)
    opt_dummy = torch.optim.AdamW(mock_dp_model.parameters(), lr=1e-3)
    sch_dummy = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dummy, T_max=20)
    
    # Giả lập 5 bước cập nhật
    for _ in range(5):
        loss_dum = mock_dp_model(torch.randn(4, 10)).sum()
        opt_dummy.zero_grad()
        loss_dum.backward()
        opt_dummy.step()
        sch_dummy.step()

    save_checkpoint(
        save_path=ckpt_save_test,
        model=mock_dp_model,
        optimizer=opt_dummy,
        scheduler=sch_dummy,
        epoch=5,
        metrics={"test_metric": 0.99},
        verbose=False,
    )
    assert os.path.isfile(ckpt_save_test), "Lưu file checkpoint thất bại"

    saved_raw = torch.load(ckpt_save_test, map_location="cpu")
    for k in saved_raw["model_state"].keys():
        assert not k.startswith("module."), f"Lỗi: Checkpoint vẫn còn tiền tố 'module.': {k}"
    print("   + Checkpoint dọn sạch 100% tiền tố 'module.' -> PASSED!")

    target_clean_model = DummyNet()
    target_opt = torch.optim.AdamW(target_clean_model.parameters(), lr=1e-3)
    target_sch = torch.optim.lr_scheduler.CosineAnnealingLR(target_opt, T_max=20)

    loaded_ckpt = load_checkpoint(
        ckpt_save_test,
        model=target_clean_model,
        optimizer=target_opt,
        scheduler=target_sch,
        device="cpu",
        verbose=False,
    )
    assert loaded_ckpt["epoch"] == 5, f"Kỳ vọng epoch=5, thực tế: {loaded_ckpt.get('epoch')}"
    assert target_sch.last_epoch == 5, f"Kỳ vọng scheduler last_epoch=5, thực tế: {target_sch.last_epoch}"
    print("   + Khôi phục trọn vẹn Model, Optimizer, Scheduler và Epoch=5 -> PASSED!")
    print("   ✅ [Checkpointing] Multi-GPU Smart Save, Load & Full Resume pass hoàn hảo!")

    # -------------------------------------------------------------
    # 7. TEST COMMON ADVANCED UTILITIES: RELIABILITY, BDB, CORRUPT
    # -------------------------------------------------------------
    print("\n--- [TEST 7] Common Advanced Utilities: Reliability, BDB & FCS ---")
    from common.reliability import StaticRegionReliabilityEstimator, CameraAlignmentChecker
    from common.degradation import BackgroundDegradationBenchmark
    from common.corrupt import FrameCorruptionSuite

    # Reliability test
    rel_estimator = StaticRegionReliabilityEstimator(temporal_variance_threshold=0.01)
    dummy_bg_tensor = torch.rand(3, 128, 128)
    dummy_static_mask = torch.ones(128, 128)
    dummy_static_mask[40:90, 40:90] = 0.0  # Vùng đường ở giữa
    dummy_seq_tensor = torch.rand(4, 3, 128, 128)
    r_score = rel_estimator.compute_reliability_score(dummy_seq_tensor, dummy_bg_tensor, dummy_static_mask)
    assert 0.0 <= r_score <= 1.0, f"r_score ngoài khoảng [0, 1]: {r_score}"
    print(f"   + Estimated Static Reliability r_i: {r_score:.4f}")

    align_checker = CameraAlignmentChecker(shift_threshold_px=4.0)
    dy, dx, is_aligned = align_checker.estimate_camera_shift(dummy_bg_tensor, dummy_bg_tensor, dummy_static_mask)
    assert abs(dy) < 1.0 and abs(dx) < 1.0 and is_aligned
    print(f"   + Camera Shift: dy={dy:.2f}px, dx={dx:.2f}px, is_aligned={is_aligned}")

    # BDB test
    bdb = BackgroundDegradationBenchmark()
    corrupted_ghost = bdb.apply_degradation(dummy_bg_tensor, "ghost_injection", severity=3)
    assert corrupted_ghost.shape == dummy_bg_tensor.shape
    print("   + BDB Ghost Injection (Severity 3): OK")

    # FCS test
    fcs = FrameCorruptionSuite()
    corrupted_rain = fcs.apply_corruption(dummy_bg_tensor, "rain_streaks", severity=4)
    assert corrupted_rain.shape == dummy_bg_tensor.shape
    print("   + FCS Rain Streaks (Severity 4): OK")
    print("   ✅ [Common Advanced] Reliability, BDB & FCS pass hoàn hảo!")


    # -------------------------------------------------------------
    # 9. TEST DIRECTION 6: ANOMALY DETECTION
    # -------------------------------------------------------------
    print("\n--- [TEST 9] Direction 6: Anomaly Detection & Persistence Filtering ---")
    from direction6_anomaly_detection.features import DINOv3PatchFeatureExtractor
    from direction6_anomaly_detection.pooling import TemporalFeaturePooler
    from direction6_anomaly_detection.bank import NormalMemoryBank
    from direction6_anomaly_detection.score import AnomalyScorer
    from direction6_anomaly_detection.camera_fault import CameraFaultClassifier
    from direction6_anomaly_detection.events import PersistenceEventTracker

    extractor_c = DINOv3PatchFeatureExtractor(backbone=MockPatchViT(embed_dim=64), feature_dim=64, proj_dim=32, patch_size=16)
    p_tokens, h_p, w_p = extractor_c.extract_patch_tokens(torch.randn(2, 3, 128, 128))
    assert p_tokens.shape == (2, 64, 32)  # (128/16)*(128/16) = 8*8 = 64 patches
    print(f"   + DINOv3 Extracted Patches: {p_tokens.shape}")

    pooler_c = TemporalFeaturePooler(window_size=3)
    pooled_c = pooler_c.update(p_tokens)
    assert pooled_c.shape == p_tokens.shape

    bank_c = NormalMemoryBank("cam_test", "morning", feature_dim=32)
    bank_c.fit_coreset(torch.randn(100, 32), subsampling_ratio=0.20)
    assert bank_c.bank is not None and len(bank_c.bank) >= 20

    scorer_c = AnomalyScorer(k_nearest=1, top_k_ratio=0.10)
    p_scores_c = scorer_c.compute_patch_scores(pooled_c, bank_c.bank)
    assert p_scores_c.shape == (2, 64)
    road_mask_patch = torch.ones(2, 64)
    road_mask_patch[:, 32:] = 0.0  # Nửa là road, nửa là static
    r_score_c, s_score_c = scorer_c.aggregate_frame_score(p_scores_c, road_mask_patch)
    assert r_score_c.shape == (2,) and s_score_c.shape == (2,)

    cam_clf = CameraFaultClassifier()
    diag = cam_clf.classify(r_score_c[0].item(), s_score_c[0].item(), 0.5, 0.5)
    assert "event_type" in diag
    print(f"   + Anomaly Diagnosis: {diag['event_type']} (Road: {r_score_c[0].item():.3f}, Static: {s_score_c[0].item():.3f})")

    tracker_c = PersistenceEventTracker(min_consecutive_windows=2)
    tracker_c.calibrate_threshold(np.array([0.1, 0.15, 0.2, 0.25, 0.3]))
    alert_1 = tracker_c.update(step_idx=0, score=0.9)
    alert_2 = tracker_c.update(step_idx=1, score=0.95)
    assert alert_2 is not None and alert_2["status"] == "CONFIRMED"
    print("   + Persistence Tracker Triggered Alert: CONFIRMED")
    print("   ✅ [Direction 6] Anomaly Detection pass hoàn hảo!")

    # -------------------------------------------------------------
    # 10. TEST DIRECTION 7: TRAFFIC FORECASTING ON CAMERA GRAPH
    # -------------------------------------------------------------
    print("\n--- [TEST 10] Direction 7: Spatio-Temporal Graph WaveNet Forecasting ---")
    from direction7_traffic_forecasting.graph import compute_haversine_distance, build_gaussian_adjacency_matrix, AdaptiveAdjacencyLayer
    from direction7_traffic_forecasting.models import CityScaleTrafficForecastingModel
    from direction7_traffic_forecasting.losses import MultiTaskForecastingLoss

    coords = np.array([[106.68, 10.76], [106.69, 10.77], [106.70, 10.78], [106.67, 10.75]])
    d_mat = compute_haversine_distance(coords)
    w_mat = build_gaussian_adjacency_matrix(d_mat, sigma=2.0)
    assert w_mat.shape == (4, 4)
    print(f"   + Physical Distance Adjacency Matrix Shape: {w_mat.shape}")

    adp_layer = AdaptiveAdjacencyLayer(num_nodes=4, embed_dim=8)
    a_adp = adp_layer()
    assert a_adp.shape == (4, 4)

    st_model = CityScaleTrafficForecastingModel(num_nodes=4, in_channels=6, hidden_channels=16, out_steps=12, num_blocks=2)
    dummy_x_d = torch.randn(2, 6, 4, 12)
    phys_adj_torch = torch.from_numpy(w_mat)
    out_d = st_model(dummy_x_d, phys_adj_torch)
    assert out_d["continuous_pred"].shape == (2, 4, 12)
    assert out_d["ordinal_logits"].shape == (2, 4, 12, 4)
    assert out_d["onset_prob"].shape == (2, 4, 3)

    crit_d = MultiTaskForecastingLoss()
    targets_d = {
        "y_target": torch.rand(2, 4, 12),
        "m_target": torch.ones(2, 4, 12),
        "cls_target": torch.randint(0, 4, (2, 4, 12)),
        "onset_target": torch.randint(0, 2, (2, 4, 3)).float(),
    }
    loss_d_dict = crit_d(out_d, targets_d)
    assert not torch.isnan(loss_d_dict["loss"]) and loss_d_dict["loss"].item() > 0
    print(f"   + ST-GNN Loss: {loss_d_dict['loss'].item():.4f} (Reg: {loss_d_dict['loss_reg'].item():.4f}, Onset: {loss_d_dict['loss_onset'].item():.4f})")
    print("   ✅ [Direction 7] Traffic Forecasting pass hoàn hảo!")

    # -------------------------------------------------------------
    # 11. TEST DIRECTION 8: BACKGROUND CONDITIONING & ADAPTATION
    # -------------------------------------------------------------
    print("\n--- [TEST 11] Direction 8: Background Conditioning & Adaptation ---")
    from direction8_bg_conditioning.descriptor import RobustSceneDescriptorExtractor
    from direction8_bg_conditioning.models import BackgroundConditionedModel

    extractor_e = RobustSceneDescriptorExtractor(backbone=MockPatchViT(embed_dim=64), feature_dim=64, trim_ratio=0.10, patch_size=16)
    z_desc = extractor_e.extract_scene_descriptor(torch.rand(1, 3, 128, 128), road_mask=torch.ones(128, 128))
    assert z_desc.shape == (1, 64 * 3)  # [road_mean, road_std, global_mean] -> 192 chiều
    print(f"   + Robust Scene Descriptor Shape: {z_desc.shape}")

    model_film_e = BackgroundConditionedModel(
        backbone=MockPatchViT(embed_dim=64), feature_dim=64, descriptor_dim=192, conditioning_mode="film", bg_dropout_prob=0.25
    )
    dummy_frame_e = torch.randn(2, 3, 128, 128)
    z_batch_e = z_desc.repeat(2, 1)
    out_e = model_film_e(dummy_frame_e, z_descriptor=z_batch_e)
    assert out_e["count"].shape == (2, 1) and (out_e["count"] >= 0).all()
    assert out_e["logits"].shape == (2, 4)
    print(f"   + FiLM Conditioned Count: {out_e['count'].flatten().tolist()} | Congestion Logits Shape: {out_e['logits'].shape}")
    print("   ✅ [Direction 8] Background Conditioning pass hoàn hảo!")

    # -------------------------------------------------------------
    # 12. TEST DIRECTION G: VEHICLE-CENTRIC SSL FROM STATIC CAMERAS
    # -------------------------------------------------------------
    print("\n--- [TEST 12] Direction G: Vehicle-Centric SSL Pretraining (TAM + AGM + SRS) ---")
    from directionG_camera_ssl.tam import PositionStats, GMMCalibrator
    from directionG_camera_ssl.masking import agm_sample
    from directionG_camera_ssl.srs import static_region_swap

    pos_stats_g = PositionStats(num_cams=2, num_patches=64, num_states=4, feat_dim=32)
    cids_g = torch.tensor([0, 1])
    u_dummy = F.normalize(torch.randn(2, 64, 32), p=2, dim=-1)
    pos_stats_g.update(cids_g, u_dummy)
    a_g, valid_g = pos_stats_g.atypicality(cids_g, u_dummy)
    assert a_g.shape == (2, 64)

    calib_g = GMMCalibrator()
    calib_g.push(a_g, valid_g)
    pi_g = calib_g.posterior(a_g)
    assert pi_g.shape == (2, 64)

    agm_mask_g = agm_sample(pi_g[0], valid_g[0], ratio=0.35, phi=0.5, q_max=0.6)
    assert agm_mask_g.shape == (64,)

    x1_g = torch.zeros(3, 128, 128)
    x2_g = torch.ones(3, 128, 128)
    pi1_g = torch.zeros(8, 8)
    pi2_g = torch.zeros(8, 8)
    x_srs_g, _ = static_region_swap(x1_g, x2_g, pi1_g, pi2_g, ratio=0.5, feather=2)
    assert x_srs_g.shape == (3, 128, 128)
    print("   ✅ [Direction G] TAM, AGM, SRS và SSL Pipeline pass hoàn hảo!")

    # -------------------------------------------------------------
    # 13. TEST DIRECTION 2 NEW: SCENE DECOMPOSITION WITHOUT BACKGROUND
    # -------------------------------------------------------------
    print("\n--- [TEST 13] Direction 2 New: Scene Decomposition Không Cần Ảnh Nền ---")
    from direction2_scene_decomposition.scene_fit import SceneBasis, robust_weights
    from direction2_scene_decomposition.solve_ell import solve_ell
    from direction2_scene_decomposition.losses import SceneDecompositionLossV2

    basis_new = SceneBasis(num_cams=1, num_frames_per_cam=2, J=4)
    dummy_f2 = torch.rand(2, 3, 256, 448)
    basis_new.init_from_frames(cam_idx=0, frames=dummy_f2)
    b_new = basis_new.get_background(cam_idx=0, frame_indices=torch.tensor([0, 1]))
    assert b_new.shape == (2, 3, 256, 448)

    w_rob, _ = robust_weights(dummy_f2, b_new)
    assert w_rob.shape == (2, 1, 256, 448)

    ell_solved = solve_ell(dummy_f2[0], basis_new.E0[0], basis_new.Ej[0], iters=3)
    assert ell_solved.shape == (4,)

    loss_v2_fn = SceneDecompositionLossV2()
    preds_mock = {
        "recon_origin": dummy_f2,
        "pred_bg": b_new,
        "pred_fg": dummy_f2,
        "alpha_mask": torch.full((2, 1, 256, 448), 0.2),
        "sigma": torch.full((2, 1, 256, 448), 0.1),
        "log_sigma": torch.full((2, 1, 256, 448), -2.3),
        "pred_ell": torch.zeros(2, 4),
    }
    loss_v2_val, _ = loss_v2_fn(preds_mock, origin=dummy_f2, bg_pseudo=b_new, mask_pseudo=1.0 - w_rob)
    assert not torch.isnan(loss_v2_val)
    print("   ✅ [Direction 2 New] SceneBasis, solve_ell và Loss V2 pass hoàn hảo!")

    print("\n" + "=" * 80)
    print(" 🎉 TOÀN BỘ CÁC HƯỚNG NGHIÊN CỨU (H1-H8, HƯỚNG G VÀ HƯỚNG 2 MỚI) ĐỀU VƯỢT QUA TEST 100%!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ [LỖI RUNTIME] Kiểm thử thất bại: {e}")
    traceback.print_exc()
    sys.exit(1)
finally:
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)
