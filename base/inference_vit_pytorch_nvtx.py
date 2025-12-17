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
    max_batches = 10

    # 只在 capture 邊界同步一次，避免把 pipeline 砍斷
    torch.cuda.synchronize()
    nvtx.range_push("NSYS_CAPTURE")

    nvtx.range_push("Inference Loop")

    # events for timing (GPU)
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    # streams
    compute_stream = torch.cuda.current_stream()
    copy_stream    = torch.cuda.Stream()

    # helper: async prefetch next batch to GPU on copy_stream
    def prefetch_to_gpu(batch):
        images, labels = batch
        with torch.cuda.stream(copy_stream):
            nvtx.range_push("H2D Prefetch")
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            nvtx.range_pop()
        ev = torch.cuda.Event()
        ev.record(copy_stream)   # record when copy is done
        return images, labels, ev

    it = iter(test_loader)

    with torch.inference_mode():
        # prefetch first batch
        first = next(it)
        images_cur, labels_cur, ev_cur = prefetch_to_gpu(first)

        starter.record()

        for batch_idx in range(max_batches):
            # prefetch next batch ASAP (overlap with current compute)
            if batch_idx + 1 < max_batches:
                nxt = next(it)
                images_next, labels_next, ev_next = prefetch_to_gpu(nxt)
            else:
                images_next = labels_next = ev_next = None

            # make sure current batch H2D finished before compute uses it
            compute_stream.wait_event(ev_cur)

            nvtx.range_push(f"Batch {batch_idx}: Forward")
            outputs = model(images_cur)
            nvtx.range_pop()

            # swap buffers
            images_cur, labels_cur, ev_cur = images_next, labels_next, ev_next

        ender.record()

    # only one sync at the end (for correct timing)
    torch.cuda.synchronize()

    nvtx.range_pop()  # Inference Loop
    nvtx.range_pop()  # NSYS_CAPTURE

    total_ms = starter.elapsed_time(ender)
    avg_time_per_batch = total_ms / max_batches
    print(f"\nAverage time per batch: {avg_time_per_batch:.2f} ms")
    print(f"Throughput: {32 * max_batches / (total_ms / 1000.0):.2f} images/sec")

    
    # # ========== Inference ==========
    # print("Starting inference (profiling 10 batches)...")
    # max_batches = 10

    # # 只在 capture 邊界同步一次，避免把 pipeline 砍斷
    # torch.cuda.synchronize()
    # nvtx.range_push("NSYS_CAPTURE")

    # nvtx.range_push("Inference Loop")

    # # 用 CUDA events 量測，不在 batch 內 synchronize
    # starter = torch.cuda.Event(enable_timing=True)
    # ender   = torch.cuda.Event(enable_timing=True)

    # with torch.inference_mode():  # inference_mode 比 no_grad 更適合推論
    #     starter.record()

    #     for batch_idx, (images, labels) in enumerate(test_loader):
    #         if batch_idx >= max_batches:
    #             break

    #         nvtx.range_push(f"Batch {batch_idx}: Data Transfer")
    #         images = images.to(device, non_blocking=True)
    #         nvtx.range_pop()

    #         nvtx.range_push(f"Batch {batch_idx}: Forward")
    #         outputs = model(images)
    #         nvtx.range_pop()

    #     ender.record()

    # nvtx.range_pop()  # Inference Loop

    # # 這裡同步一次就夠：確保 10 batches 的 GPU 工作都完成，時間也才會正確
    # torch.cuda.synchronize()
    # nvtx.range_pop()  # NSYS_CAPTURE

    # # 10 batches 的總 GPU 時間（ms），再算平均
    # total_ms = starter.elapsed_time(ender)
    # avg_time_per_batch = total_ms / max_batches
    # print(f"\nAverage time per batch: {avg_time_per_batch:.2f} ms")
    # print(f"Throughput: {32 * max_batches / (total_ms / 1000.0):.2f} images/sec")

if __name__ == '__main__':
    main()
