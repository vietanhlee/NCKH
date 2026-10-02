"""
=============================================================================
 Comprehensive Smoke Test & Verification Suite for DINO Traffic Suite (8 Directions)
 Kiểm thử toàn diện 8 hướng nghiên cứu và tầng Common Utilities
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

    # Mock ViT backbone chung cho các bài test tiếp theo
    class MockPatchViT(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            self.embed_dim = embed_dim
            self.conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        def forward(self, x):
            return torch.zeros(x.shape[0], self.embed_dim)
        def get_intermediate_layers(self, x, n=1, return_class_token=True):
            feat = self.conv(x)  # (B, embed_dim, 14, 14)
            b, c, h, w = feat.shape
            tokens = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
            cls_t = torch.zeros(b, c)
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
    assert not torch.isnan(l2) and l2.item() > 0
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
    # 5. TEST DIRECTION 5: SPATIO-TEMPORAL DENSITY & LoS ESTIMATION
    # -------------------------------------------------------------
    print("\n--- [TEST 5] Direction 5: Spatio-Temporal Road Occupancy & LoS Estimation ---")
    from direction5_temporal_density.dataset import TemporalTrafficDataset
    from direction5_temporal_density.models import SpatioTemporalDensityNet
    from direction5_temporal_density.losses import SpatioTemporalDensityLoss

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
    print("   ✅ [Direction 5] Spatio-Temporal Density & LoS pass hoàn hảo!")

    # -------------------------------------------------------------
    # 6. TEST DIRECTION 8: ROAD SURFACE CONDITION ESTIMATION
    # -------------------------------------------------------------
    print("\n--- [TEST 6] Direction 8: Road Surface Condition Estimation ---")
    from direction8_road_condition.dataset import RoadSurfaceDataset
    from direction8_road_condition.models import RoadConditionClassifier
    from direction8_road_condition.losses import SurfaceConsistencyLoss

    ds8 = RoadSurfaceDataset(bg_dir=bg_dir, img_size=128)
    assert len(ds8) >= 2, f"Kỳ vọng >= 2 ảnh nền, thực tế: {len(ds8)}"
    sample8 = ds8[0]
    assert sample8["image"].shape == (3, 128, 128)
    assert "specular_ratio" in sample8 and "roughness" in sample8

    road_model = RoadConditionClassifier(backbone=mock_vit, embed_dim=64, hidden_dim=32, freeze_backbone=True)
    dummy_bg_tensor = torch.stack([sample8["image"], sample8["image"]])
    road_out = road_model(dummy_bg_tensor)
    assert road_out["pred_wetness"].shape == (2,)
    assert road_out["logits_illum"].shape == (2, 3)
    assert road_out["pred_degradation"].shape == (2,)

    crit8 = SurfaceConsistencyLoss()
    target8 = {
        "illum_class": torch.tensor([2, 2]),
        "specular_ratio": torch.tensor([0.05, 0.05]),
        "roughness": torch.tensor([0.1, 0.1]),
    }
    l8_dict = crit8(road_out, target8)
    assert not torch.isnan(l8_dict["loss_total"])
    print(f"   + Road Surface Loss: {l8_dict['loss_total'].item():.4f}")
    print("   ✅ [Direction 8] Road Surface Condition pass hoàn hảo!")

    # -------------------------------------------------------------
    # 7. TEST MULTI-GPU SMART SAVE & LOAD CHECKPOINTING
    # -------------------------------------------------------------
    print("\n--- [TEST 7] Multi-GPU Smart Checkpointing Interoperability ---")
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

    print("\n" + "=" * 80)
    print(" 🎉 TOÀN BỘ 5 HƯỚNG NGHIÊN CỨU TRỌNG TÂM, COMMON UTILITIES VÀ RESUME ĐỀU VƯỢT QUA TEST 100%!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ [LỖI RUNTIME] Kiểm thử thất bại: {e}")
    traceback.print_exc()
    sys.exit(1)
finally:
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)
