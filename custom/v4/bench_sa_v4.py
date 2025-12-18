"""
Benchmark script for Self-Attention v4 kernel
Tests correctness against PyTorch baseline and measures performance
"""

import torch
import torch.nn.functional as F
import argparse
import time

def pytorch_attention_fp16(Q, K, V, scale):
    """PyTorch reference implementation in fp16 (manual matmul + softmax)"""
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out

def pytorch_attention_fp32(Q, K, V, scale):
    """PyTorch reference implementation in fp32 (manual matmul + softmax)"""
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    scores = torch.matmul(Qf, Kf.transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, Vf)
    return out.half()

def pytorch_sdpa(Q, K, V, scale):
    """PyTorch scaled_dot_product_attention (uses Flash Attention)"""
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(Q, K, V, scale=scale)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=32, help="Batch size")
    parser.add_argument("--H", type=int, default=12, help="Number of heads")
    parser.add_argument("--N", type=int, default=197, help="Sequence length")
    parser.add_argument("--D", type=int, default=64, help="Head dimension")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(42)

    B, H, N, D = args.B, args.H, args.N, args.D
    scale = D ** -0.5

    print(f"{'='*60}")
    print(f"Self-Attention v4 Benchmark")
    print(f"B={B}, H={H}, N={N}, D={D}")
    print(f"{'='*60}")

    # Create inputs
    Q = torch.randn(B, H, N, D, device=device, dtype=torch.float16).contiguous()
    K = torch.randn(B, H, N, D, device=device, dtype=torch.float16).contiguous()
    V = torch.randn(B, H, N, D, device=device, dtype=torch.float16).contiguous()

    # Import v4 kernel
    try:
        from sa_v4 import sa_forward_v4
        print("[OK] sa_v4_ext loaded successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import sa_v4_ext: {e}")
        print("Please run: python setup.py install")
        return

    # Correctness test
    print("\n--- Correctness Test ---")
    ref_out = pytorch_attention_fp32(Q, K, V, scale)
    v4_out = sa_forward_v4(Q, K, V, scale)

    # Compare
    abs_diff = (ref_out.float() - v4_out.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Relative error
    rel_diff = abs_diff / (ref_out.float().abs() + 1e-6)
    max_rel = rel_diff.max().item()

    print(f"Max absolute diff: {max_diff:.6f}")
    print(f"Mean absolute diff: {mean_diff:.6f}")
    print(f"Max relative diff: {max_rel:.6f}")

    if max_diff < 0.1 and mean_diff < 0.01:
        print("[PASS] Correctness test passed!")
    else:
        print("[WARN] Large numerical difference detected")

    # Warmup
    print(f"\n--- Performance Test ({args.iters} iters, {args.warmup} warmup) ---")

    for _ in range(args.warmup):
        _ = sa_forward_v4(Q, K, V, scale)
    torch.cuda.synchronize()

    # Benchmark v4
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(args.iters):
        _ = sa_forward_v4(Q, K, V, scale)
    end_event.record()
    torch.cuda.synchronize()

    v4_time = start_event.elapsed_time(end_event) / args.iters
    print(f"v4 kernel: {v4_time:.4f} ms/iter")

    # Benchmark PyTorch manual matmul+softmax (fp16)
    for _ in range(args.warmup):
        _ = pytorch_attention_fp16(Q, K, V, scale)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(args.iters):
        _ = pytorch_attention_fp16(Q, K, V, scale)
    end_event.record()
    torch.cuda.synchronize()

    pytorch_manual_time = start_event.elapsed_time(end_event) / args.iters
    print(f"PyTorch manual fp16 (matmul+softmax): {pytorch_manual_time:.4f} ms/iter")

    # Benchmark PyTorch SDPA (Flash Attention)
    for _ in range(args.warmup):
        _ = pytorch_sdpa(Q, K, V, scale)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(args.iters):
        _ = pytorch_sdpa(Q, K, V, scale)
    end_event.record()
    torch.cuda.synchronize()

    pytorch_sdpa_time = start_event.elapsed_time(end_event) / args.iters
    print(f"PyTorch SDPA (Flash Attention): {pytorch_sdpa_time:.4f} ms/iter")

    # Speedup comparisons
    speedup_manual = pytorch_manual_time / v4_time
    speedup_sdpa = pytorch_sdpa_time / v4_time
    print(f"\nSpeedup vs manual: {speedup_manual:.2f}x {'(v4 faster)' if speedup_manual > 1 else '(manual faster)'}")
    print(f"Speedup vs SDPA:   {speedup_sdpa:.2f}x {'(v4 faster)' if speedup_sdpa > 1 else '(SDPA faster)'}")

    # FLOPS calculation
    # Self-attention: 2*B*H*N*N*D (QK^T) + 2*B*H*N*N*D (softmax*V) = 4*B*H*N^2*D
    flops = 4 * B * H * N * N * D
    v4_tflops = (flops / (v4_time / 1000)) / 1e12
    manual_tflops = (flops / (pytorch_manual_time / 1000)) / 1e12
    sdpa_tflops = (flops / (pytorch_sdpa_time / 1000)) / 1e12

    print(f"\nv4 TFLOPS: {v4_tflops:.2f}")
    print(f"PyTorch manual TFLOPS: {manual_tflops:.2f}")
    print(f"PyTorch SDPA TFLOPS: {sdpa_tflops:.2f}")

if __name__ == "__main__":
    main()
