"""
ViT Inference with Custom CUDA Softmax Kernel
Architecture matches vit-pytorch exactly for weight compatibility
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from custom_softmax import custom_softmax, CustomSoftmax, CUDA_AVAILABLE

SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    """Matches vit-pytorch FeedForward exactly"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Matches vit-pytorch Attention exactly, but uses custom CUDA softmax"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        with nvtx.range("attention_matmul_qk"):
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        with nvtx.range("custom_cuda_softmax"):
            attn = custom_softmax(dots, dim=-1)
        attn = self.dropout(attn)

        with nvtx.range("attention_matmul_v"):
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Matches vit-pytorch Transformer exactly"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class CustomViT(nn.Module):
    """
    Vision Transformer with Custom CUDA Softmax
    Architecture matches vit-pytorch ViT exactly for weight compatibility
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        # Matches vit-pytorch exactly
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        batch = img.shape[0]

        with nvtx.range("patch_embedding"):
            x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b=batch)
        x = torch.cat((cls_tokens, x), dim=1)

        seq = x.shape[1]

        with nvtx.range("position_embedding"):
            x = x + self.pos_embedding[:seq]
            x = self.dropout(x)

        with nvtx.range("transformer"):
            x = self.transformer(x)

        with nvtx.range("classification"):
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)

        return self.mlp_head(x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA kernel available: {CUDA_AVAILABLE}")

    # Set seed before model initialization for reproducibility
    set_seed(SEED)
    print(f"Random seed: {SEED}")

    print("\n" + "=" * 50)
    print("Creating Custom ViT with CUDA Softmax")
    print("=" * 50)

    model = CustomViT(
        image_size=224,
        patch_size=16,
        num_classes=10,       # CIFAR-10
        dim=768,              # embedding dimension
        depth=12,             # transformer blocks
        heads=12,             # attention heads
        mlp_dim=3072,         # feedforward hidden dim
        dropout=0.0,
        emb_dropout=0.0
    ).to(device)

    # Load trained weights if available
    weight_path = Path(__file__).parent.parent / 'base' / 'vit_cifar10.pth'
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from: {weight_path}")
    else:
        print("Warning: No trained weights found, using random initialization")

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\nLoading CIFAR-10 dataset...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Warming up...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # ========== Inference ==========
    print("Starting inference...")
    correct = 0
    total = 0
    total_time = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            torch.cuda.synchronize()
            start_time = time.time()

            with nvtx.range(f"batch_{batch_idx}"):
                outputs = model(images)

            torch.cuda.synchronize()
            end_time = time.time()

            total_time += (end_time - start_time)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Batch [{batch_idx + 1}/{len(test_loader)}] - "
                      f"Accuracy: {100 * correct / total:.2f}%")

    accuracy = 100 * correct / total
    avg_time_per_batch = total_time / len(test_loader) * 1000
    avg_time_per_image = total_time / total * 1000
    throughput = total / total_time

    print("\n" + "=" * 50)
    print("Custom ViT + CUDA Softmax Results")
    print("=" * 50)
    print(f"CUDA Kernel: {'Enabled' if CUDA_AVAILABLE else 'Disabled (fallback to PyTorch)'}")
    print(f"Total images: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total inference time: {total_time:.2f} s")
    print(f"Average time per batch: {avg_time_per_batch:.2f} ms")
    print(f"Average time per image: {avg_time_per_image:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")
    print("=" * 50)


if __name__ == '__main__':
    main()
