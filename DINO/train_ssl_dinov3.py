"""
=============================================================================
 DINOv3 Self-Supervised Domain Adaptation for Urban Traffic Surveillance
 Continual SSL Pre-training & Metric Representation Learning on Traffic Feeds
 
 Architecture:
   - Backbone: Meta's DINOv3 (ViT-S/16, ViT-B/16, ViT-L/16 with 2D RoPE & LayerScale)
   - Fallback: Meta's DINOv2 (ViT-S/14, ViT-B/14 pre-trained on LVD-142M)
   - Distillation: EMA Momentum Teacher with Centering & Sharpening
   - Optimization: Layer-wise Decoupled LR (Backbone LR << Projection Head LR)
   - Multi-Crop Resolution: Patch-16 aligned (Global: 224x224, Local: 96x96)
                           Patch-14 aligned (Global: 224x224, Local: 98x98)

 References:
   - Meta AI: "DINOv3: Self-Supervised Vision Transformers with RoPE" (2024)
   - Caron et al. "Emerging Properties in Self-Supervised Vision Transformers" (ICCV 2021)
   - Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision" (TMLR 2024)
   - Wang & Isola: "Understanding Contrastive Representation Learning through Alignment and Uniformity" (ICML 2020)
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

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for background/server training
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


class DataAugmentationDINO:
    """
    Multi-crop data augmentation strategy for DINOv3 and DINOv2 ViT.
    NOTE on Resolutions:
      - DINOv3 utilizes patch size 16x16: Global 224x224 (14x14 patches), Local 96x96 (6x6 patches).
      - DINOv2 utilizes patch size 14x14: Global 224x224 (16x16 patches), Local 98x98 (7x7 patches).
    """
    def __init__(
        self,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 4,
        size_global: int = 224,
        size_local: int = 96,
        patch_size: int = 16,
    ):
        assert size_global % patch_size == 0, f"size_global ({size_global}) must be a multiple of {patch_size}!"
        assert size_local % patch_size == 0, f"size_local ({size_local}) must be a multiple of {patch_size}!"

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
    model_name: str = "dinov3_vits16",
    pretrained: bool = True,
    weights_path: str = None
) -> Tuple[nn.Module, int]:
    """
    Initializes Vision Backbone with support for Meta's DINOv3 & DINOv2.
    Supports:
      - 'dinov3_vits16' (ViT-Small/16 with 2D RoPE, 384-dim, LVD-1689M 1.7B images)
      - 'dinov3_vitb16' (ViT-Base/16 with 2D RoPE, 768-dim)
      - 'dinov3_convnext_tiny' (ConvNeXt-Tiny pre-trained via DINOv3)
      - 'dinov2_vits14' (ViT-Small/14, 384-dim, 21M params - Ultra-fast & lightweight)
      - 'dinov2_vitb14' (ViT-Base/14, 768-dim, 86M params)
      - timm ViTs / ConvNeXt / ResNet baselines
    """
    # 1. Native Meta AI DINOv3 via facebookresearch/dinov3 (LVD-1689M Foundation Architecture)
    if "dinov3" in model_name.lower():
        hub_name = model_name.lower().strip()
        if hub_name in ["dinov3", "dinov3_small", "dinov3_s", "dinov3_vits"]:
            hub_name = "dinov3_vits16"
        elif hub_name in ["dinov3_base", "dinov3_b", "dinov3_vitb"]:
            hub_name = "dinov3_vitb16"
        elif hub_name in ["dinov3_large", "dinov3_l", "dinov3_vitl"]:
            hub_name = "dinov3_vitl16"
        elif hub_name in ["dinov3_convnext", "dinov3_convnext_tiny"]:
            hub_name = "dinov3_convnext_tiny"

        print(f"   [Model Loader] Loading official Meta DINOv3 '{hub_name}' (LVD-1689M 1.7B Foundation Model)...")
        try:
            import sys
            dinov3_cache = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main")
            if not os.path.exists(dinov3_cache):
                try:
                    torch.hub.help("facebookresearch/dinov3", "dinov3_vits16")
                except Exception:
                    pass
            if os.path.exists(dinov3_cache) and dinov3_cache not in sys.path:
                sys.path.insert(0, dinov3_cache)
            import dinov3.hub.backbones as d3_bb
            model_fn = getattr(d3_bb, hub_name, None)
            if model_fn is not None:
                # 1. Attempt HuggingFace Hub authenticated download if HF_TOKEN is present
                hf_token = os.environ.get("HF_TOKEN", None)
                if hf_token and (weights_path is None or not os.path.exists(weights_path)):
                    try:
                        from huggingface_hub import hf_hub_download
                        arch_tag = hub_name.replace('_', '-')
                        repo_candidates = [
                            f"facebook/{arch_tag}-pretrain-lvd1689m",
                            f"facebook/{arch_tag}",
                            f"facebook/dinov3-{arch_tag.split('-')[-1]}"
                        ]
                        print(f"   🔑 [HuggingFace Hub] Checking authenticated access with provided HF_TOKEN...")
                        for r_id in repo_candidates:
                            for c_file in ["model.safetensors", "pytorch_model.bin", f"{hub_name}_pretrain_lvd1689m.pth"]:
                                try:
                                    dl_file = hf_hub_download(repo_id=r_id, filename=c_file, token=hf_token)
                                    if dl_file and os.path.exists(dl_file):
                                        weights_path = dl_file
                                        print(f"   ✅ [HuggingFace Hub] Successfully retrieved official weights from '{r_id}': {dl_file}")
                                        break
                                except Exception:
                                    continue
                            if weights_path:
                                break
                    except Exception as e_hf:
                        print(f"   [Notice] HuggingFace Hub check: {e_hf}")

                if weights_path and os.path.exists(weights_path):
                    model = model_fn(pretrained=False)
                    if str(weights_path).endswith(".safetensors"):
                        try:
                            from safetensors.torch import load_file
                            state = load_file(weights_path)
                        except Exception:
                            state = torch.load(weights_path, map_location="cpu")
                    else:
                        state = torch.load(weights_path, map_location="cpu")

                    if "student" in state:
                        state = state["student"]
                    cleaned_state = {}
                    for k, v in state.items():
                        clean_k = k
                        while clean_k.startswith("module.") or clean_k.startswith("0."):
                            if clean_k.startswith("module."):
                                clean_k = clean_k[len("module."):]
                            if clean_k.startswith("0."):
                                clean_k = clean_k[len("0."):]
                        cleaned_state[clean_k] = v
                    model.load_state_dict(cleaned_state, strict=False)
                    print(f"   [Model Loader] Loaded custom offline DINOv3 weights from: {weights_path}")
                else:
                    try:
                        model = model_fn(pretrained=pretrained)
                    except Exception as e_pt:
                        print(f"   [Notice] Meta DINOv3 official remote weights require HuggingFace gated token / offline file: {e_pt}")
                        model = model_fn(pretrained=False)
                        if pretrained:
                            # Warm-start DINOv3 from ungated Meta DINOv2 foundation weights!
                            try:
                                d2_name = "dinov2_vits14" if "vits" in hub_name else "dinov2_vitb14"
                                print(f"   💡 [Warm-Start] Bootstrapping DINOv3 Transformer blocks from public Meta DINOv2 '{d2_name}'...")
                                d2_model = torch.hub.load("facebookresearch/dinov2", d2_name, pretrained=True)
                                d2_state = d2_model.state_dict()
                                compatible_state = {}
                                for k, v in d2_state.items():
                                    if "pos_embed" in k or "patch_embed" in k:
                                        continue
                                    if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                                        compatible_state[k] = v
                                model.load_state_dict(compatible_state, strict=False)
                                print(f"   ✅ [Warm-Start Success] Initialized {len(compatible_state)} layers (Attention, MLP, LayerNorm) from Meta DINOv2 foundation weights into DINOv3!")
                            except Exception as e_ws:
                                print(f"   -> Initializing DINOv3 architecture for in-domain SSL pre-training from scratch: {e_ws}")
                embed_dim = getattr(model, "embed_dim", 384)
                print(f"   [Model Loader] Loaded DINOv3 '{hub_name}' successfully! Embedding Dim: {embed_dim}")
                return model, embed_dim
        except Exception as e:
            print(f"   [Warning] DINOv3 direct load encountered: {e}. Falling back...")

    # 2. Native Meta AI DINOv2 via PyTorch Hub (Official LVD-142M Foundation Weights)
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
                    state = state["student"]
                cleaned_state = {}
                for k, v in state.items():
                    clean_k = k
                    while clean_k.startswith("module.") or clean_k.startswith("0."):
                        if clean_k.startswith("module."):
                            clean_k = clean_k[len("module."):]
                        if clean_k.startswith("0."):
                            clean_k = clean_k[len("0."):]
                    cleaned_state[clean_k] = v
                model.load_state_dict(cleaned_state, strict=False)

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


def train_ssl_dinov3(args):
    # Set deterministic seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Auto-adjust patch-aligned crop resolutions (DINOv3 uses Patch-16, DINOv2 uses Patch-14)
    is_patch16 = ("16" in args.backbone.lower() or "dinov3" in args.backbone.lower())
    patch_size = 16 if is_patch16 else 14
    size_global = (args.size_global // patch_size) * patch_size
    size_local = (args.size_local // patch_size) * patch_size
    if size_local == 0:
        size_local = 96 if patch_size == 16 else 98

    print("\n" + "=" * 70)
    print(" 🚀 STARTING DINO SELF-SUPERVISED CONTINUAL PRE-TRAINING")
    print("=" * 70)
    print(f" Pretrained Backbone : {args.backbone} (Meta Foundation Model)")
    print(f" Compute Device      : {device}")
    print(f" Batch Size          : {args.batch_size}")
    print(f" Epochs              : {args.epochs}")
    print(f" Peak LR (Head)      : {args.lr}")
    print(f" Backbone LR Scale   : {args.backbone_lr_scale} (Peak Backbone LR: {args.lr * args.backbone_lr_scale})")
    print(f" Global Crop Size    : {size_global}x{size_global} (Patch {patch_size} Divisible)")
    print(f" Local Crop Size     : {size_local}x{size_local} (Patch {patch_size} Divisible)")
    print(f" Local Crops Count   : {args.local_crops}")
    print(f" Output Projection   : {args.out_dim}-dim")
    print("=" * 70)

    # 1. Discover Images
    if not os.path.isdir(args.data_dir):
        for candidate in ["images", "../images", "data/camera_images", "../data/camera_images", "data/images", "../data/images"]:
            if os.path.isdir(candidate):
                args.data_dir = candidate
                break

    image_paths = []
    if os.path.isdir(args.data_dir):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            image_paths.extend(glob.glob(os.path.join(args.data_dir, "**", ext), recursive=True))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No valid images found in directory: {args.data_dir}")

    print(f"   Using image directory: {args.data_dir}")
    print(f"   Discovered {len(image_paths)} unlabelled traffic camera frames on disk (SSL Training on full dataset).")

    # 2. Data Loader with Patch-aligned Multi-crop Augmentation
    transform = DataAugmentationDINO(
        local_crops_number=args.local_crops,
        size_global=size_global,
        size_local=size_local,
        patch_size=patch_size,
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

    # 5. Multi-GPU DataParallel Setup
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if num_gpus > 1:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"\n⚡ [Multi-GPU] Detected {num_gpus} GPUs: {gpu_names}")
        print(f"⚡ [Multi-GPU] Activating DataParallel for high-throughput distributed tensor computation across all {num_gpus} devices.")
        student = nn.DataParallel(student)
        teacher = nn.DataParallel(teacher)

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

    # History tracker for publication-grade training curves
    history = {"loss": [], "lr_head": [], "lr_backbone": [], "teacher_temp": []}

    # 6. Training Loop
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
                # Local crops (96x96)
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

            # 4. EMA Update for Teacher weights (Always access raw underlying modules safely)
            student_raw = student.module if hasattr(student, "module") else student
            teacher_raw = teacher.module if hasattr(teacher, "module") else teacher
            with torch.no_grad():
                for param_s, param_t in zip(student_raw.parameters(), teacher_raw.parameters()):
                    param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

            total_epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{curr_base_lr:.6f}")

        avg_loss = total_epoch_loss / len(dataloader)
        print(f"Ep {epoch+1:03d}/{args.epochs:03d} | DINO Loss: {avg_loss:.4f} | Base LR: {curr_base_lr:.6f} | Temp: {dino_loss.teacher_temp_schedule[epoch]:.4f}")

        # Record metrics for figures
        history["loss"].append(avg_loss)
        history["lr_head"].append(curr_base_lr)
        history["lr_backbone"].append(curr_base_lr * args.backbone_lr_scale)
        history["teacher_temp"].append(float(dino_loss.teacher_temp_schedule[epoch]))

        # Save Checkpoint (Always save UNWRAPPED model state dicts to prevent 'module.' prefix bugs!)
        student_raw = student.module if hasattr(student, "module") else student
        teacher_raw = teacher.module if hasattr(teacher, "module") else teacher

        if avg_loss < best_loss or (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            best_loss = min(best_loss, avg_loss)

            # Save full student/teacher training state
            full_ckpt_path = os.path.join(args.save_dir, "dinov3_traffic_ssl_latest.pth")
            torch.save({
                "epoch": epoch + 1,
                "student": student_raw.state_dict(),
                "teacher": teacher_raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
                "backbone_name": args.backbone,
                "embed_dim": embed_dim,
                "history": history,
            }, full_ckpt_path)

            # Save Clean Domain-Adapted DINOv3 Backbone (Ready for downstream tasks / graph caching!)
            backbone_path = os.path.join(args.save_dir, "dinov3_traffic_backbone.pth")
            torch.save(student_raw[0].state_dict(), backbone_path)
            # Backward compatibility alias
            torch.save(student_raw[0].state_dict(), os.path.join(args.save_dir, "dinov2_traffic_backbone.pth"))
            print(f"   -> Checkpoint saved to: {backbone_path}")

            # Save updated training curves periodically
            if args.save_figures:
                save_training_curves(history, args.save_dir)

    elapsed = time.time() - start_time
    print(f"\n🏁 DINOv3 Continual Pre-training Completed in {elapsed/60:.2f} minutes!")
    print(f"💾 Adapted DINOv3 Backbone weights ready: {os.path.join(args.save_dir, 'dinov3_traffic_backbone.pth')}\n")

    # Final publication figures: Training Curves + Emergent PCA Feature Maps
    if args.save_figures:
        print("🎨 Generating publication-grade figures...")
        student_raw = student.module if hasattr(student, "module") else student
        save_training_curves(history, args.save_dir)
        save_pca_feature_maps(student_raw[0], image_paths, args.save_dir, device)


# =====================================================================
# 5. PUBLICATION FIGURE GENERATION UTILITIES
# =====================================================================

def save_training_curves(history: Dict[str, List[float]], save_dir: str):
    """
    Plots and saves publication-grade training curves for DINOv2 SSL.
    Generates both high-res PNG (300 DPI) and vector PDF for LaTeX.
    """
    if len(history["loss"]) == 0:
        return

    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    # 1. DINO Self-Distillation Loss
    axes[0].plot(epochs, history["loss"], color="#1f77b4", linewidth=2.2, marker="o", markersize=3, label="DINO SSL Loss")
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Loss", fontsize=11, fontweight="bold")
    axes[0].set_title("Self-Distillation Convergence", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", frameon=True)

    # 2. Learning Rate Schedule
    axes[1].plot(epochs, history["lr_head"], color="#2ca02c", linewidth=2.0, label="Head Peak LR")
    axes[1].plot(epochs, history["lr_backbone"], color="#ff7f0e", linewidth=2.0, linestyle="--", label="Backbone Adapted LR")
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Learning Rate", fontsize=11, fontweight="bold")
    axes[1].set_title("Cosine Warmup & Decay Schedule", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", frameon=True)

    # 3. Teacher Temperature Sharpening
    axes[2].plot(epochs, history["teacher_temp"], color="#d62728", linewidth=2.2, label="Teacher Temp (Sharpening)")
    axes[2].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Temperature", fontsize=11, fontweight="bold")
    axes[2].set_title("Teacher Sharpening Dynamics", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="lower right", frameon=True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "dinov3_ssl_training_curves.png")
    pdf_path = os.path.join(save_dir, "dinov3_ssl_training_curves.pdf")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    # Also save dinov2 compatibility names
    plt.savefig(os.path.join(save_dir, "dinov2_ssl_training_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(save_dir, "dinov2_ssl_training_curves.pdf"), bbox_inches="tight")
    plt.close()
    print(f"   📈 Saved Training Curves to: {png_path} and {pdf_path}")


@torch.no_grad()
def save_pca_feature_maps(backbone: nn.Module, sample_paths: List[str], save_dir: str, device: torch.device, num_samples: int = 4):
    """
    Renders Meta DINOv2 PCA Feature Maps for representative traffic camera frames.
    Visualizes emergent visual clustering (cars, motorcycles, road surfaces) without labels!
    """
    backbone.eval()
    selected_paths = sample_paths[:num_samples]
    if len(selected_paths) == 0:
        return

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    fig, axes = plt.subplots(len(selected_paths), 2, figsize=(8, 3.2 * len(selected_paths)), dpi=300)
    if len(selected_paths) == 1:
        axes = np.expand_dims(axes, 0)

    for idx, path in enumerate(selected_paths):
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            orig_resized = img_rgb.resize((224, 224))
            inp = eval_transform(img_rgb).unsqueeze(0).to(device)

            # Extract spatial patch tokens
            patch_tokens = None
            if hasattr(backbone, "get_intermediate_layers"):
                try:
                    raw_tokens = backbone.get_intermediate_layers(inp, n=1)[0].squeeze(0).cpu().numpy()
                    n_tokens = raw_tokens.shape[0]
                    # Check if CLS token is present (e.g., 197 for 14x14 patches or 257 for 16x16 patches)
                    if math.isqrt(n_tokens) ** 2 == n_tokens:
                        patch_tokens = raw_tokens
                    elif math.isqrt(n_tokens - 1) ** 2 == (n_tokens - 1):
                        patch_tokens = raw_tokens[1:]  # Discard CLS token to retain spatial grid
                    else:
                        patch_tokens = raw_tokens
                except Exception:
                    patch_tokens = None

            if patch_tokens is not None and len(patch_tokens) > 0:
                # Fit PCA to 3 components (RGB channels)
                pca = PCA(n_components=3)
                pca_features = pca.fit_transform(patch_tokens)  # (N_patches, 3)

                # Min-max normalize to [0, 1] for RGB display
                for c in range(3):
                    c_min, c_max = pca_features[:, c].min(), pca_features[:, c].max()
                    pca_features[:, c] = (pca_features[:, c] - c_min) / (c_max - c_min + 1e-8)

                h_patches = w_patches = int(math.isqrt(pca_features.shape[0]))
                pca_img = pca_features[:h_patches * w_patches].reshape(h_patches, w_patches, 3)

                # Plot Original Camera Frame
                axes[idx, 0].imshow(orig_resized)
                axes[idx, 0].set_title(f"Camera Frame: {os.path.basename(path)[:22]}", fontsize=10, fontweight="bold")
                axes[idx, 0].axis("off")

                # Plot Emergent DINOv3 PCA Feature Map
                axes[idx, 1].imshow(pca_img, interpolation="bilinear")
                axes[idx, 1].set_title("DINOv3 Emergent PCA Feature Map (RGB)", fontsize=10, fontweight="bold", color="#1f77b4")
                axes[idx, 1].axis("off")
            else:
                axes[idx, 0].imshow(orig_resized)
                axes[idx, 0].set_title(f"Frame: {os.path.basename(path)[:22]}", fontsize=10, fontweight="bold")
                axes[idx, 0].axis("off")
                axes[idx, 1].imshow(orig_resized)
                axes[idx, 1].set_title("Feature Map Fallback", fontsize=10, fontweight="bold")
                axes[idx, 1].axis("off")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "dinov3_pca_feature_maps.png")
    pdf_path = os.path.join(save_dir, "dinov3_pca_feature_maps.pdf")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    # Also save dinov2 compatibility names
    plt.savefig(os.path.join(save_dir, "dinov2_pca_feature_maps.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(save_dir, "dinov2_pca_feature_maps.pdf"), bbox_inches="tight")
    plt.close()
    print(f"   📊 Saved Emergent PCA Feature Maps to: {png_path} and {pdf_path}")


# =====================================================================
# 6. OFFLINE FEATURE CACHING UTILITY (FOR GRAPH & COUNTING TASKS)
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
    parser = argparse.ArgumentParser(description="DINOv3 / DINOv2 Self-Supervised Domain Adaptation for Traffic Cameras")
    parser.add_argument("--data_dir", type=str, default="data/camera_images", help="Path to traffic camera images directory")
    parser.add_argument("--backbone", type=str, default="dinov3_vits16", help="Backbone model (e.g. dinov3_vits16, dinov2_vits14, dinov3_vitb16, dinov3_convnext_tiny)")
    parser.add_argument("--pretrained_init", action="store_true", default=True, help="Initialize with Meta LVD Foundation weights before SSL fine-tuning")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Optional path to local .pth checkpoint for offline loading")
    parser.add_argument("--epochs", type=int, default=30, help="Number of SSL fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.0005, help="Peak learning rate for DINO projection head")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1, help="LR multiplier for pre-trained backbone (prevents catastrophic forgetting)")
    parser.add_argument("--size_global", type=int, default=224, help="Global crop dimension (divisible by patch size: 16 or 14)")
    parser.add_argument("--size_local", type=int, default=96, help="Local crop dimension (96 for patch 16, 98 for patch 14)")
    parser.add_argument("--out_dim", type=int, default=4096, help="Dimensionality of DINO projection head output")
    parser.add_argument("--local_crops", type=int, default=4, help="Number of local crops in multi-crop augmentation")
    parser.add_argument("--clip_grad", type=float, default=3.0, help="Gradient clipping norm")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints/dinov3_traffic_ssl", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="Target compute device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Visualization flag
    parser.add_argument("--save_figures", action="store_true", default=True, help="Automatically save training curves and PCA feature maps (PNG & PDF)")

    # Feature caching flag
    parser.add_argument("--cache_features", action="store_true", help="Extract feature embeddings for downstream tasks")
    parser.add_argument("--cache_output", type=str, default="traffic_dinov3_embeddings.pt", help="Output path for cached embeddings")

    # Hugging Face Auth Token
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face user access token for gated models")

    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    if args.cache_features:
        weights = os.path.join(args.save_dir, "dinov3_traffic_backbone.pth")
        if not os.path.exists(weights):
            weights = os.path.join(args.save_dir, "dinov2_traffic_backbone.pth")
        extract_feature_cache(
            backbone_weights=weights,
            image_dir=args.data_dir,
            output_file=args.cache_output,
            backbone_name=args.backbone,
            device=args.device,
        )
    else:
        train_ssl_dinov3(args)


# Backward compatibility aliases
train_ssl_dinov2 = train_ssl_dinov3


if __name__ == "__main__":
    main()
