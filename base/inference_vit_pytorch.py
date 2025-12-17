from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'vit-pytorch'))

import torch
import torch.cuda.nvtx as nvtx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from vit_pytorch import ViT

SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed before model initialization for reproducibility
    set_seed(SEED)
    print(f"Random seed: {SEED}")

    print("Creating ViT model (vit-pytorch version)...")

    model = ViT(
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
    weight_path = Path(__file__).parent / 'vit_cifar10.pth'
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from: {weight_path}")
    else:
        print("Warning: No trained weights found, using random initialization")

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("Loading CIFAR-10 dataset...")

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

    # ========== Warmup ==========
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
    print("vit-pytorch Results (ViT-Base CIFAR-10)")
    print("=" * 50)
    print(f"Total images: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total inference time: {total_time:.2f} s")
    print(f"Average time per batch: {avg_time_per_batch:.2f} ms")
    print(f"Average time per image: {avg_time_per_image:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")
    print("=" * 50)


if __name__ == '__main__':
    main()
