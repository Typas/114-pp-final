# inference_vit_pytorch_nvtx.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'vit-pytorch'))

import torch
import torch.cuda.nvtx as nvtx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

from vit_pytorch import ViT


def main():
    device = torch.device('cuda')

    nvtx.range_push("Create Model")
    print("Creating ViT model...")
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.0,
        emb_dropout=0.0
    ).to(device)
    model.eval()
    nvtx.range_pop()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    nvtx.range_push("Load Dataset")
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
    nvtx.range_pop()

    # ========== Warmup ==========
    nvtx.range_push("Warmup")
    print("Warming up...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # ========== Inference ==========
    print("Starting inference (profiling 10 batches)...")
    total_time = 0
    max_batches = 10

    nvtx.range_push("Inference Loop")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            nvtx.range_push(f"Batch {batch_idx}: Data Transfer")
            images = images.to(device)
            nvtx.range_pop()

            torch.cuda.synchronize()
            start_time = time.time()

            nvtx.range_push(f"Batch {batch_idx}: Forward")
            outputs = model(images)
            nvtx.range_pop()

            torch.cuda.synchronize()
            total_time += (time.time() - start_time)

    nvtx.range_pop()

    avg_time = total_time / max_batches * 1000
    print(f"\nAverage time per batch: {avg_time:.2f} ms")
    print(f"Throughput: {32 * max_batches / total_time:.2f} images/sec")


if __name__ == '__main__':
    main()
