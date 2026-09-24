"""
=============================================================================
 DINOv2 Self-Supervised Domain Adaptation for Urban Traffic Surveillance
 Continual SSL Pre-training & Feature Caching on City-Scale CCTV Feeds
 
 Architecture:
   - Backbone: Meta's DINOv2 (ViT-S/14, ViT-B/14, ViT-L/14) pre-trained on LVD-142M
   - Distillation: EMA Momentum Teacher with Centering & Sharpening
   - Optimization: Layer-wise Decoupled LR (Backbone LR << Projection Head LR)
   - Resolution: Patch-14 aligned (Global: 224x224, Local: 98x98)

 References:
   - Caron et al. "Emerging Properties in Self-Supervised Vision Transformers" (ICCV 2021)
   - Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision" (TMLR 2024)
=============================================================================
"""

import argparse
import copy
import glob
import math
import os
import random
import sys
import time
import warnings
from typing import List, Tuple, Dict, Any

# Note: If xformers is not installed, PyTorch uses native FlashAttention / SDPA.
# To enable xformers acceleration, run: pip install xformers

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# =====================================================================
# 1. DATA AUGMENTATION: MULTI-CROP PIPELINE FOR DINOv2 (PATCH-14 ALIGNED)
# =====================================================================

class GaussianBlur:
    """Gaussian blur augmentation with randomized radius."""
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class Solarization:
    """Solarization augmentation (inverting pixels above threshold)."""
    def __init__(self, p: float = 0.2):
        self.prob = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            return ImageOps.solarize(img)
        return img


class DataAugmentationDINOv2:
    """
    Multi-crop data augmentation strategy for DINOv2 ViT.
    NOTE on Resolutions:
      DINOv2 utilizes a patch size of 14x14.
      Both Global and Local crop dimensions MUST be multiples of 14:
        - Global Views: 224 x 224 (16 x 16 patches) -> Fed to both Teacher & Student
        - Local Views :  98 x  98 ( 7 x  7 patches) -> Fed to Student ONLY
    """
    def __init__(
        self,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 4,
        size_global: int = 224,
        size_local: int = 98,
    ):
        assert size_global % 14 == 0, f"size_global ({size_global}) must be a multiple of 14 for DINOv2!"
        assert size_local % 14 == 0, f"size_local ({size_local}) must be a multiple of 14 for DINOv2!"

        self.local_crops_number = local_crops_number

        color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
        )
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # 1. Global View 1 (Gaussian Blur, No Solarize)
        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                size_global, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            transforms.ToTensor(),
            normalize,
        ])

        # 2. Global View 2 (Light Blur + Solarization)
        self.global_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                size_global, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        # 3. Local Views (Focal Crops for fine-grained vehicle features)
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size_local, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        crops = []
        crops.append(self.global_transform_1(image))
        crops.append(self.global_transform_2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


class TrafficImageDataset(Dataset):
    """Loads raw unlabelled CCTV camera frames from arbitrary folder hierarchies."""
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                crops = self.transform(img)
                return crops
            return img


# =====================================================================
# 2. MODEL ARCHITECTURE: DINOv2 BACKBONE & DINO PROJECTION HEAD
# =====================================================================

class DINOHead(nn.Module):
    """
    3-layer MLP projection head with L2-normalized bottleneck and weight-normalized output.
    Maps high-dimensional CLS token embeddings to representation space (e.g., 4096 or 65536).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        linear = nn.Linear(bottleneck_dim, out_dim, bias=False)
        try:
            # Official PyTorch 2.x standard: torch.nn.utils.parametrizations.weight_norm
            from torch.nn.utils.parametrizations import weight_norm
            self.last_layer = weight_norm(linear)
            self.last_layer.parametrizations.weight.original0.data.fill_(1)
            if norm_last_layer:
                self.last_layer.parametrizations.weight.original0.requires_grad = False
        except (ImportError, AttributeError):
            # Backward compatibility fallback for legacy PyTorch versions
            self.last_layer = nn.utils.weight_norm(linear)
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


def build_backbone(
    model_name: str = "dinov2_vits14",
    pretrained: bool = True,
    weights_path: str = None
) -> Tuple[nn.Module, int]:
    """
    Initializes Vision Backbone with first-class support for Meta's DINOv2.
    Supports:
      - 'dinov2_vits14' (ViT-Small/14, 384-dim, 21M params - Recommended for edge & speed)
      - 'dinov2_vitb14' (ViT-Base/14, 768-dim, 86M params - SOTA representation)
      - 'dinov2_vitl14' (ViT-Large/14, 1024-dim, 300M params)
      - 'dinov2_vits14_reg' (ViT-Small with 4 register tokens)
      - timm ViTs / ConvNeXt / ResNet baselines
    """
    # 1. Native Meta AI DINOv2 via PyTorch Hub (Official LVD-142M Foundation Weights)
    if "dinov2" in model_name.lower():
        hub_name = model_name.lower().strip()
        # Aliases
        if hub_name in ["dinov2", "dinov2_small", "dinov2_s"]:
            hub_name = "dinov2_vits14"
        elif hub_name in ["dinov2_base", "dinov2_b"]:
            hub_name = "dinov2_vitb14"
        elif hub_name in ["dinov2_large", "dinov2_l"]:
            hub_name = "dinov2_vitl14"

        try:
            print(f"   [Model Loader] Loading official Meta DINOv2 '{hub_name}' via torch.hub (pretrained={pretrained})...")
            model = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=pretrained)
            embed_dim = getattr(model, "embed_dim", None)
            if embed_dim is None:
                dim_map = {
                    "dinov2_vits14": 384,
                    "dinov2_vits14_reg": 384,
                    "dinov2_vitb14": 768,
                    "dinov2_vitb14_reg": 768,
                    "dinov2_vitl14": 1024,
                    "dinov2_vitg14": 1536,
                }
                embed_dim = dim_map.get(hub_name, 384)

            # Load local offline checkpoint if specified
            if weights_path and os.path.exists(weights_path):
                print(f"   [Model Loader] Applying custom offline weights from: {weights_path}")
                state = torch.load(weights_path, map_location="cpu")
                if "student" in state:
                    state = {k.replace("0.", ""): v for k, v in state["student"].items() if k.startswith("0.")}
                model.load_state_dict(state, strict=False)

            print(f"   [Model Loader] Loaded DINOv2 '{hub_name}' successfully! Embedding Dim: {embed_dim}")
            return model, embed_dim
        except Exception as e:
            print(f"   [Warning] Could not load '{hub_name}' via torch.hub: {e}. Falling back to timm / torchvision...")

    # 2. Timm Models fallback (timm DINOv2 or ConvNeXt)
    if HAS_TIMM:
        try:
            timm_name = model_name
            if "dinov2" in model_name:
                timm_name = "vit_small_patch14_dinov2.lvd142m" if "vits" in model_name else "vit_base_patch14_dinov2.lvd142m"
            print(f"   [Model Loader] Loading '{timm_name}' via timm...")
            model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
            embed_dim = model.num_features
            return model, embed_dim
        except Exception as e:
            print(f"   [Warning] Failed loading '{model_name}' via timm: {e}. Falling back...")

    # 3. Torchvision ConvNeXt & ResNet fallbacks
    from torchvision import models
    if "convnext" in model_name:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = models.convnext_tiny(weights=weights)
        embed_dim = m.classifier[2].in_features
        m.classifier = nn.Identity()
        return m, embed_dim
    elif "resnet50" in model_name:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
        embed_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, embed_dim
    else:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=weights)
        embed_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, embed_dim


# =====================================================================
# 3. DINO LOSS: CENTERING & SHARPENING
# =====================================================================

class DINOLoss(nn.Module):
    """
    Self-Distillation Cross-Entropy Loss with Teacher Centering and Sharpening.
    Guarantees stable representation learning without collapse.
    """
    def __init__(
        self,
        out_dim: int = 4096,
        ncrops: int = 6,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 10,
        nepochs: int = 50,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Teacher temperature cosine warmup schedule
        warmup_teacher_temp_epochs = min(warmup_teacher_temp_epochs, nepochs)
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(0, nepochs - warmup_teacher_temp_epochs)) * teacher_temp
        ))

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Computes cross-entropy between Student predictions and Teacher target distribution.
          student_output: (ncrops * B, out_dim)
          teacher_output: (2 * B, out_dim)
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0.0
        n_loss_terms = 0

        # Match 2 Teacher views against all (2 global + N local) Student views
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip matching identical crops
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True) / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# =====================================================================
# 4. TRAINING & FINE-TUNING PIPELINE
# =====================================================================

def get_cosine_schedule(base_value: float, final_value: float, epochs: int, niter_per_ep: int, warmup_epochs: int = 5) -> np.ndarray:
    """Generates cosine learning rate or momentum schedules."""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(0, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(1, len(iters))))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule


def train_ssl_dinov2(args):
    # Set deterministic seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(" 🚀 STARTING DINOv2 SELF-SUPERVISED CONTINUAL PRE-TRAINING")
    print("=" * 70)
    print(f" Pretrained Backbone : {args.backbone} (Meta LVD-142M Foundation Model)")
    print(f" Compute Device      : {device}")
    print(f" Batch Size          : {args.batch_size}")
    print(f" Epochs              : {args.epochs}")
    print(f" Peak LR (Head)      : {args.lr}")
    print(f" Backbone LR Scale   : {args.backbone_lr_scale} (Peak Backbone LR: {args.lr * args.backbone_lr_scale})")
    print(f" Global Crop Size    : {args.size_global}x{args.size_global} (Patch 14 Divisible)")
    print(f" Local Crop Size     : {args.size_local}x{args.size_local} (Patch 14 Divisible)")
    print(f" Local Crops Count   : {args.local_crops}")
    print(f" Output Projection   : {args.out_dim}-dim")
    print("=" * 70)

    # 1. Discover Images
    image_paths = []
    if os.path.isdir(args.data_dir):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            image_paths.extend(glob.glob(os.path.join(args.data_dir, "**", ext), recursive=True))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No valid images found in directory: {args.data_dir}")

    print(f"   Discovered {len(image_paths)} unlabelled traffic camera frames.")

    # 2. Data Loader with Patch-14 Multi-crop Augmentation
    transform = DataAugmentationDINOv2(
        local_crops_number=args.local_crops,
        size_global=args.size_global,
        size_local=args.size_local,
    )
    dataset = TrafficImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 3. Build Student & Teacher Models initialized from Meta's DINOv2
    student_backbone, embed_dim = build_backbone(args.backbone, pretrained=args.pretrained_init, weights_path=args.pretrained_weights)
    student_head = DINOHead(embed_dim, out_dim=args.out_dim)
    student = nn.Sequential(student_backbone, student_head).to(device)

    teacher_backbone, _ = build_backbone(args.backbone, pretrained=args.pretrained_init, weights_path=args.pretrained_weights)
    teacher_head = DINOHead(embed_dim, out_dim=args.out_dim)
    teacher = nn.Sequential(teacher_backbone, teacher_head).to(device)

    # Synchronize teacher weights with student at step 0
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # 4. Decoupled Parameter Groups: Lower LR for Foundation Backbone, Higher LR for Head
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        is_decay = ("bias" not in name and len(param.shape) > 1)
        if name.startswith("0."):  # Backbone parameters
            if is_decay:
                backbone_decay.append(param)
            else:
                backbone_no_decay.append(param)
        else:  # Head parameters
            if is_decay:
                head_decay.append(param)
            else:
                head_no_decay.append(param)

    backbone_lr_init = args.lr * args.backbone_lr_scale
    params_groups = [
        {"params": backbone_decay, "lr": backbone_lr_init, "weight_decay": 0.04, "is_backbone": True},
        {"params": backbone_no_decay, "lr": backbone_lr_init, "weight_decay": 0.0, "is_backbone": True},
        {"params": head_decay, "lr": args.lr, "weight_decay": 0.04, "is_backbone": False},
        {"params": head_no_decay, "lr": args.lr, "weight_decay": 0.0, "is_backbone": False},
    ]
    optimizer = torch.optim.AdamW(params_groups)

    n_iter_per_epoch = len(dataloader)
    lr_schedule = get_cosine_schedule(args.lr, 1e-6, args.epochs, n_iter_per_epoch, warmup_epochs=min(5, args.epochs // 5))
    momentum_schedule = get_cosine_schedule(0.996, 1.0, args.epochs, n_iter_per_epoch, warmup_epochs=0)

    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        ncrops=2 + args.local_crops,
        nepochs=args.epochs,
        student_temp=0.1,
    ).to(device)

    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    # 5. Training Loop
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        student.train()
        total_epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:03d}/{args.epochs}", leave=False)
        for it, crops in enumerate(pbar):
            global_it = len(dataloader) * epoch + it

            # Update Learning Rate & Teacher Momentum
            curr_base_lr = lr_schedule[global_it]
            for pg in optimizer.param_groups:
                if pg.get("is_backbone", False):
                    pg["lr"] = curr_base_lr * args.backbone_lr_scale
                else:
                    pg["lr"] = curr_base_lr
            m = momentum_schedule[global_it]

            # Send crops to device
            crops = [im.to(device, non_blocking=True) for im in crops]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                # 1. Teacher forward pass (Only 2 global crops of 224x224)
                with torch.no_grad():
                    teacher_output = teacher(torch.cat(crops[:2]))

                # 2. Student forward pass (Handled separately to support different global/local resolutions)
                # Global crops (224x224)
                student_global_out = student(torch.cat(crops[:2]))
                # Local crops (98x98)
                if len(crops) > 2:
                    student_local_out = student(torch.cat(crops[2:]))
                    student_output = torch.cat([student_global_out, student_local_out], dim=0)
                else:
                    student_output = student_global_out

                # 3. Compute DINO Loss
                loss = dino_loss(student_output, teacher_output, epoch)

            # Backward & Gradient Update
            if device.type == 'cuda':
                scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
                optimizer.step()

            # 4. EMA Update for Teacher weights
            with torch.no_grad():
                for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                    param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

            total_epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{curr_base_lr:.6f}")

        avg_loss = total_epoch_loss / len(dataloader)
        print(f"Ep {epoch+1:03d}/{args.epochs:03d} | DINO Loss: {avg_loss:.4f} | Base LR: {curr_base_lr:.6f} | Temp: {dino_loss.teacher_temp_schedule[epoch]:.4f}")

        # Save Checkpoint
        if avg_loss < best_loss or (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            best_loss = min(best_loss, avg_loss)

            # Save full student/teacher training state
            full_ckpt_path = os.path.join(args.save_dir, "dinov2_traffic_ssl_latest.pth")
            torch.save({
                "epoch": epoch + 1,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
                "backbone_name": args.backbone,
                "embed_dim": embed_dim,
            }, full_ckpt_path)

            # Save Clean Domain-Adapted DINOv2 Backbone (Ready for downstream tasks / graph caching!)
            backbone_path = os.path.join(args.save_dir, "dinov2_traffic_backbone.pth")
            torch.save(student[0].state_dict(), backbone_path)
            print(f"   -> Checkpoint saved to: {backbone_path}")

    elapsed = time.time() - start_time
    print(f"\n🏁 DINOv2 Continual Pre-training Completed in {elapsed/60:.2f} minutes!")
    print(f"💾 Adapted DINOv2 Backbone weights ready: {os.path.join(args.save_dir, 'dinov2_traffic_backbone.pth')}\n")


# =====================================================================
# 5. OFFLINE FEATURE CACHING UTILITY (FOR GRAPH & COUNTING TASKS)
# =====================================================================

@torch.no_grad()
def extract_feature_cache(
    backbone_weights: str,
    image_dir: str,
    output_file: str,
    backbone_name: str = "dinov2_vits14",
    device: str = "cuda"
):
    """
    Extracts dense CLS token embeddings across all traffic camera images
    using the domain-adapted DINOv2 ViT, exporting a dictionary tensor
    ready for Spatio-Temporal Graph forecasting (TA-STGCN) and downstream counting!
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"\n📦 Initializing Feature Extraction with '{backbone_name}' on {dev}...")

    # Load Backbone
    backbone, embed_dim = build_backbone(backbone_name, pretrained=(not os.path.exists(backbone_weights)))
    if os.path.exists(backbone_weights):
        state = torch.load(backbone_weights, map_location="cpu")
        if "student" in state:
            state = {k.replace("0.", ""): v for k, v in state["student"].items() if k.startswith("0.")}
        backbone.load_state_dict(state, strict=False)
        print(f"   Loaded adapted DINOv2 weights from: {backbone_weights}")
    else:
        print("   Notice: Using official Meta DINOv2 pre-trained weights (Zero-shot / Off-the-shelf).")

    backbone = backbone.to(dev).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    files = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
    if len(files) == 0:
        files = sorted(glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True))

    print(f"   Extracting offline embeddings for {len(files)} camera images...")

    cache: Dict[str, torch.Tensor] = {}
    for f in tqdm(files, desc="Extracting DINOv2 Features"):
        with Image.open(f) as img:
            tensor = transform(img.convert("RGB")).unsqueeze(0).to(dev)
            emb = backbone(tensor).squeeze(0).cpu()  # Shape: (embed_dim,)
            cache[os.path.basename(f)] = emb

    torch.save(cache, output_file)
    print(f"✅ Successfully cached {len(cache)} embeddings to: {output_file} (Embedding Dim: {embed_dim})")


# =====================================================================
# 6. CLI ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="DINOv2 Self-Supervised Domain Adaptation for Traffic Cameras")
    parser.add_argument("--data_dir", type=str, default="data/camera_images", help="Path to traffic camera images directory")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14", help="Backbone model (e.g. dinov2_vits14, dinov2_vitb14, convnext_tiny)")
    parser.add_argument("--pretrained_init", action="store_true", default=True, help="Initialize with Meta LVD-142M weights before SSL fine-tuning")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Optional path to local .pth checkpoint for offline loading")
    parser.add_argument("--epochs", type=int, default=30, help="Number of SSL fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.0005, help="Peak learning rate for DINO projection head")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1, help="LR multiplier for pre-trained backbone (prevents catastrophic forgetting)")
    parser.add_argument("--size_global", type=int, default=224, help="Global crop dimension (must be divisible by 14)")
    parser.add_argument("--size_local", type=int, default=98, help="Local crop dimension (must be divisible by 14, e.g. 98 = 14*7)")
    parser.add_argument("--out_dim", type=int, default=4096, help="Dimensionality of DINO projection head output")
    parser.add_argument("--local_crops", type=int, default=4, help="Number of local crops in multi-crop augmentation")
    parser.add_argument("--clip_grad", type=float, default=3.0, help="Gradient clipping norm")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints/dinov2_traffic_ssl", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="Target compute device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Feature caching flag
    parser.add_argument("--cache_features", action="store_true", help="Extract feature embeddings for downstream tasks")
    parser.add_argument("--cache_output", type=str, default="traffic_dinov2_embeddings.pt", help="Output path for cached embeddings")

    args = parser.parse_args()

    if args.cache_features:
        weights = os.path.join(args.save_dir, "dinov2_traffic_backbone.pth")
        extract_feature_cache(
            backbone_weights=weights,
            image_dir=args.data_dir,
            output_file=args.cache_output,
            backbone_name=args.backbone,
            device=args.device,
        )
    else:
        train_ssl_dinov2(args)


if __name__ == "__main__":
    main()
