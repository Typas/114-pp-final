from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'vit-pytorch'))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vit_pytorch import ViT
import time


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========== Hyperparameters ==========
    batch_size = 128
    epochs = 50
    learning_rate = 3e-4
    weight_decay = 0.1
    warmup_epochs = 5

    # ========== Model ==========
    print("Creating ViT model...")
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=10,       # CIFAR-10
        dim=768,              # embedding dimension
        depth=12,             # transformer blocks
        heads=12,             # attention heads
        mlp_dim=3072,         # feedforward hidden dim
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ========== Data ==========
    print("Loading CIFAR-10 dataset...")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ========== Loss & Optimizer ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (epochs - warmup_epochs) * 3.14159)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ========== Training ==========
    best_acc = 0.0
    save_path = Path(__file__).parent / 'vit_cifar10.pth'

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {100*correct/total:.2f}%")

        scheduler.step()

        epoch_time = time.time() - start_time
        train_acc = 100 * correct / total
        avg_loss = train_loss / len(train_loader)

        # ========== Validation ==========
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100 * test_correct / test_total

        print(f"\nEpoch [{epoch+1}/{epochs}] - Time: {epoch_time:.1f}s")
        print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with accuracy: {best_acc:.2f}%")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    print("=" * 50)


if __name__ == '__main__':
    main()
