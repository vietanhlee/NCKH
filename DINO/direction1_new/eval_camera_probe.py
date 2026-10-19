"""
=============================================================================
 Hướng G: eval_camera_probe.py
 Đánh giá định lượng Shortcut Camera ID (Camera-ID Linear Probe)
 Đo lường mức độ mô hình còn mã hóa đặc trưng nhận diện góc máy / camera ID
 trên CLS token và Mean Patch Token để chứng minh triệt tiêu shortcut.
=============================================================================
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_dino_root = os.path.dirname(_script_dir)
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from common.backbone_loader import get_dino_backbone, extract_tokens, imagenet_normalize
from common.gpu_utils import clean_state_dict, smart_load_state_dict


class ProbeDataset(Dataset):
    """Dataset trích xuất ảnh giao thông phân chia theo Camera ID."""
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (256, 448), max_per_cam: int = 50):
        super().__init__()
        self.H, self.W = img_size
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        all_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    all_files.append(os.path.join(root, f))
        all_files.sort()

        if not all_files:
            raise ValueError(f"Không tìm thấy ảnh hợp lệ trong '{data_dir}'!")

        cam_to_files = {}
        for path in all_files:
            rel = os.path.relpath(path, data_dir)
            parts = rel.split(os.sep)
            cam_key = parts[0] if len(parts) > 1 else path.split("_")[0]
            if cam_key not in cam_to_files:
                cam_to_files[cam_key] = []
            cam_to_files[cam_key].append(path)

        self.cam_keys = sorted(list(cam_to_files.keys()))
        self.cam_to_id = {k: i for i, k in enumerate(self.cam_keys)}
        self.num_cams = len(self.cam_keys)

        self.samples = []
        for k in self.cam_keys:
            cid = self.cam_to_id[k]
            files = cam_to_files[k][:max_per_cam]
            for f in files:
                self.samples.append((f, cid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, cid = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (self.W, self.H):
                img = img.resize((self.W, self.H), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor_img = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor_img, cid


class LinearProbeClassifier(nn.Module):
    """Linear Probe chuẩn hóa Cosine để phân loại Camera ID."""
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        return self.fc(x_norm)


@torch.no_grad()
def extract_all_features(backbone, loader, device, patch_size=16):
    """Trích xuất representation CLS và Mean Patch token cho toàn bộ dataset."""
    backbone.eval()
    cls_list, patch_mean_list, labels_list = [], [], []

    for imgs, cids in loader:
        imgs = imgs.to(device)
        imgs_norm = imagenet_normalize(imgs)
        cls_tok, patch_spatial = extract_tokens(backbone, imgs_norm, patch_size=patch_size)
        # patch_spatial: (B, Hp, Wp, D)
        B, Hp, Wp, D = patch_spatial.shape
        patch_mean = patch_spatial.reshape(B, Hp * Wp, D).mean(dim=1)

        cls_list.append(cls_tok.cpu())
        patch_mean_list.append(patch_mean.cpu())
        labels_list.append(cids)

    cls_all = torch.cat(cls_list, dim=0)
    patch_all = torch.cat(patch_mean_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    return cls_all, patch_all, labels_all


def train_eval_probe(features: torch.Tensor, labels: torch.Tensor, num_classes: int, epochs: int = 50, lr: float = 1e-2):
    """Huấn luyện Linear Probe và đo độ chính xác Top-1 / Top-5."""
    N, D = features.shape
    # Chia train / val 80% / 20%
    perm = torch.randperm(N)
    n_train = int(0.8 * N)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]

    probe = LinearProbeClassifier(feat_dim=D, num_classes=num_classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch_sz = min(64, len(train_idx))
    for ep in range(epochs):
        probe.train()
        step_perm = torch.randperm(len(train_idx))
        for i in range(0, len(train_idx), batch_sz):
            b_idx = step_perm[i : i + batch_sz]
            bx, by = X_train[b_idx], y_train[b_idx]
            logits = probe(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Đánh giá trên tập val
    probe.eval()
    with torch.no_grad():
        val_logits = probe(X_val)
        top1_preds = val_logits.argmax(dim=-1)
        top1_acc = (top1_preds == y_val).float().mean().item() * 100.0

        k = min(5, num_classes)
        top5_preds = torch.topk(val_logits, k=k, dim=-1)[1]
        top5_acc = (top5_preds == y_val.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0

    return top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(description="Camera-ID Linear Probe: Đo lường mức độ rò rỉ Camera ID")
    parser.add_argument("--data_dir", type=str, default="/workspace/output", help="Thư mục dữ liệu ảnh thực tế")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn checkpoint (.pth) cần đánh giá")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16", help="Tên backbone DINOv3")
    parser.add_argument("--weights", type=str, default=None, help="Trọng số backbone")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace Token")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size trích xuất feature")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="probe_results.json", help="File lưu kết quả")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print(" 📡 [CAMERA-ID LINEAR PROBE] ĐÁNH GIÁ MỨC ĐỘ RÒ RỈ CAMERA ID TRONG BIỂU DIỄN")
    print("=" * 80)

    # 1. Nạp Dataset
    if os.path.isdir(args.data_dir):
        dataset = ProbeDataset(data_dir=args.data_dir, max_per_cam=40)
        print(f"✅ Nạp {len(dataset)} ảnh phân bổ trên {dataset.num_cams} camera từ: '{args.data_dir}'")
    else:
        print(f"⚠️ Thư mục '{args.data_dir}' không tồn tại. Tạo dữ liệu giả lập 4 camera để test...")
        from direction1_new.train import SyntheticTrafficDataset
        mock_ds = SyntheticTrafficDataset(num_samples=80, num_cams=4)
        dataset = mock_ds
        dataset.num_cams = 4

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Nạp Mô hình Backbone
    backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.model_name,
        pretrained=True,
        weights_path=args.weights,
        device=device,
        hf_token=args.hf_token,
    )

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"🔄 Nạp trọng số Student Backbone từ Checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        bb_sd = ckpt.get("student_backbone", ckpt.get("model_state", {}))
        smart_load_state_dict(backbone, bb_sd, strict=False, verbose=True)

    backbone = backbone.to(device)

    # 3. Trích xuất biểu diễn
    print("⏳ Đang trích xuất đặc trưng CLS token và Mean Patch token...")
    start_t = time.time()
    cls_all, patch_all, labels_all = extract_all_features(backbone, loader, device, patch_size=patch_size)
    print(f"   Trích xuất xong {cls_all.shape[0]} mẫu trong {time.time() - start_t:.2f}s.")

    # 4. Huấn luyện Linear Probe
    num_classes = dataset.num_cams
    print(f"🎯 Huấn luyện Linear Probe trên CLS Token (D={cls_all.shape[-1]}, classes={num_classes})...")
    cls_top1, cls_top5 = train_eval_probe(cls_all, labels_all, num_classes=num_classes)

    print(f"🎯 Huấn luyện Linear Probe trên Mean Patch Token...")
    patch_top1, patch_top5 = train_eval_probe(patch_all, labels_all, num_classes=num_classes)

    chance_acc = 100.0 / max(1, num_classes)
    results = {
        "checkpoint": args.checkpoint if args.checkpoint else "Pretrained Backbone",
        "num_cameras": num_classes,
        "chance_level_acc": float(f"{chance_acc:.2f}"),
        "cls_probe": {
            "top1_acc": float(f"{cls_top1:.2f}"),
            "top5_acc": float(f"{cls_top5:.2f}"),
        },
        "patch_probe": {
            "top1_acc": float(f"{patch_top1:.2f}"),
            "top5_acc": float(f"{patch_top5:.2f}"),
        },
    }

    print("\n" + "=" * 80)
    print(" 📊 KẾT QUẢ LINEAR PROBE CAMERA ID:")
    print("=" * 80)
    print(f"   - Tỷ lệ đoán mò ngẫu nhiên (Chance Level) : {chance_acc:.2f}%")
    print(f"   - CLS Token Probe Top-1 Accuracy           : {cls_top1:.2f}% (Top-5: {cls_top5:.2f}%)")
    print(f"   - Patch Token Probe Top-1 Accuracy         : {patch_top1:.2f}% (Top-5: {patch_top5:.2f}%)")
    if cls_top1 < 50.0:
        print("   ✅ [KẾT LUẬN] Độ chính xác nhận diện Camera ID thấp -> Triệt tiêu Shortcut thành công!")
    else:
        print("   ⚠️ [LƯU Ý] Mô hình vẫn còn giữ một phần thông tin Camera ID.")
    print("=" * 80)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"💾 Đã lưu kết quả tại: {args.output_json}")


if __name__ == "__main__":
    main()
