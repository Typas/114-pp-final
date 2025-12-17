import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import datasets
from torch.utils.data import DataLoader
import time

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_name = 'nateraw/vit-base-patch16-224-cifar10'
    print(f"Loading pretrained model: {model_name}")

    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("Loading CIFAR-10 dataset...")

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=None
    )

    def collate_fn(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        inputs = processor(images=images, return_tensors="pt")
        return inputs['pixel_values'], torch.tensor(labels)

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print("Warming up...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    print("Starting inference...")
    correct = 0
    total = 0
    total_time = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            outputs = model(images).logits

            if device.type == 'cuda':
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
    print("Baseline Results (ViT-Base CIFAR-10)")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Total images: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total inference time: {total_time:.2f} s")
    print(f"Average time per batch: {avg_time_per_batch:.2f} ms")
    print(f"Average time per image: {avg_time_per_image:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")
    print("=" * 50)


if __name__ == '__main__':
    main()
