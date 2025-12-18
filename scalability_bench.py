#!/usr/bin/env python3
"""
Scalability benchmark for ViT Self-Attention kernels.
Tests different image sizes (token counts) across all kernel versions.
"""
import sys
import argparse
from pathlib import Path

# Add paths for imports
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / 'vit-pytorch'))
sys.path.insert(0, str(ROOT_DIR / 'custom' / 'v0'))
sys.path.insert(0, str(ROOT_DIR / 'custom' / 'v1'))
sys.path.insert(0, str(ROOT_DIR / 'custom' / 'v2'))
sys.path.insert(0, str(ROOT_DIR / 'custom' / 'v3'))
sys.path.insert(0, str(ROOT_DIR / 'custom' / 'v4'))
sys.path.insert(0, str(ROOT_DIR / 'custom' / 'v5'))
sys.path.insert(0, str(ROOT_DIR / 'custom' / 'v6'))

import torch
import torch.cuda.nvtx as nvtx

from vit_pytorch import ViT

SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_patch_function(version):
    """Dynamically import and return the patch function for a given version."""
    if version == 'baseline':
        return None
    elif version == 'v0':
        from custom.v0.patch_sa_v0 import patch_vit_pytorch_attention_sa_v0
        return patch_vit_pytorch_attention_sa_v0
    elif version == 'v1':
        from custom.v1.patch_sa_v1 import patch_vit_pytorch_attention_sa_v1
        return patch_vit_pytorch_attention_sa_v1
    elif version == 'v2':
        from custom.v2.patch_sa_v2 import patch_vit_pytorch_attention_sa_v2
        return patch_vit_pytorch_attention_sa_v2
    elif version == 'v3':
        from custom.v3.patch_sa_v3 import patch_vit_pytorch_attention_sa_v3
        return patch_vit_pytorch_attention_sa_v3
    elif version == 'v4':
        from custom.v4.patch_sa_v4 import patch_vit_pytorch_attention_sa_v4
        return patch_vit_pytorch_attention_sa_v4
    elif version == 'v5':
        from custom.v5.patch_sa_v5 import patch_vit_pytorch_attention_sa_v5
        return patch_vit_pytorch_attention_sa_v5
    elif version == 'v6':
        from custom.v6.patch_sa_v6 import patch_vit_pytorch_attention_sa_v6
        return patch_vit_pytorch_attention_sa_v6
    else:
        raise ValueError(f"Unknown version: {version}")


def run_benchmark(image_size, patch_size, version, batch_size=32, num_batches=10, warmup_iters=10):
    """Run benchmark for a specific configuration."""
    device = torch.device('cuda')

    # Calculate token count
    num_patches = (image_size // patch_size) ** 2
    num_tokens = num_patches + 1  # +1 for CLS token

    print(f"\n{'='*60}")
    print(f"Config: image_size={image_size}, patch_size={patch_size}")
    print(f"        tokens={num_tokens}, version={version}")
    print(f"{'='*60}")

    # Set seed for reproducibility
    set_seed(SEED)

    # Create model
    nvtx.range_push("Create Model")
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.0,
        emb_dropout=0.0
    ).to(device)

    # Apply custom kernel patch if not baseline
    patch_fn = get_patch_function(version)
    if patch_fn is not None:
        model = patch_fn(model)

    model.eval()
    nvtx.range_pop()

    print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Warmup
    nvtx.range_push("Warmup")
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    nvtx.range_pop()
    print(f"Warmup done ({warmup_iters} iterations)")

    # Benchmark
    torch.cuda.synchronize()
    nvtx.range_push("NSYS_CAPTURE")
    nvtx.range_push("Inference Loop")

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    # Pre-generate all input batches
    inputs = [torch.randn(batch_size, 3, image_size, image_size, device=device)
              for _ in range(num_batches)]

    with torch.inference_mode():
        starter.record()

        for batch_idx in range(num_batches):
            nvtx.range_push(f"Batch {batch_idx}: Forward")
            _ = model(inputs[batch_idx])
            nvtx.range_pop()

        ender.record()

    nvtx.range_pop()  # Inference Loop
    torch.cuda.synchronize()
    nvtx.range_pop()  # NSYS_CAPTURE

    total_ms = starter.elapsed_time(ender)
    avg_ms = total_ms / num_batches
    throughput = batch_size * num_batches / (total_ms / 1000.0)

    print(f"Results: avg={avg_ms:.2f} ms/batch, throughput={throughput:.2f} img/s")

    # Clean up
    del model, inputs, dummy_input
    torch.cuda.empty_cache()

    return {
        'image_size': image_size,
        'patch_size': patch_size,
        'num_tokens': num_tokens,
        'version': version,
        'batch_size': batch_size,
        'num_batches': num_batches,
        'total_ms': total_ms,
        'avg_ms': avg_ms,
        'throughput': throughput,
    }


def main():
    parser = argparse.ArgumentParser(description='Scalability benchmark for ViT attention kernels')
    parser.add_argument('--image-size', type=int, required=True,
                        help='Image size (e.g., 224, 384, 512)')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size (default: 16)')
    parser.add_argument('--version', type=str, required=True,
                        choices=['baseline', 'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
                        help='Kernel version to test')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num-batches', type=int, default=10,
                        help='Number of batches to run (default: 10)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations (default: 10)')

    args = parser.parse_args()

    result = run_benchmark(
        image_size=args.image_size,
        patch_size=args.patch_size,
        version=args.version,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        warmup_iters=args.warmup,
    )

    # Print summary for easy parsing
    print(f"\n--- RESULT ---")
    print(f"IMAGE_SIZE={result['image_size']}")
    print(f"TOKENS={result['num_tokens']}")
    print(f"VERSION={result['version']}")
    print(f"AVG_MS={result['avg_ms']:.4f}")
    print(f"THROUGHPUT={result['throughput']:.2f}")


if __name__ == '__main__':
    main()
