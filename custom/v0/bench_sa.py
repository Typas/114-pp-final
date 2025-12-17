import argparse
import torch
import torch.cuda.nvtx as nvtx

from sa import sa_forward


def explicit_attention_ref(Q, K, V, scale):
    """
    Reference: scores = QK^T * scale; attn = softmax(scores); O = attn V
    Compute in fp32 for stability, return fp16 to compare with kernel output.
    Shapes: [B,H,N,D]
    """
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    scores = torch.matmul(Qf, Kf.transpose(-1, -2)) * scale  # [B,H,N,N]
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, Vf)  # [B,H,N,D]
    return out.half()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--H", type=int, default=12)
    parser.add_argument("--N", type=int, default=197)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--profile", action="store_true", help="wrap with NSYS_CAPTURE NVTX range")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    device = "cuda"

    torch.manual_seed(0)
    B, H, N, D = args.B, args.H, args.N, args.D
    scale = D ** -0.5

    # Inputs: fp16, contiguous
    Q = torch.randn(B, H, N, D, device=device, dtype=torch.float16).contiguous()
    K = torch.randn(B, H, N, D, device=device, dtype=torch.float16).contiguous()
    V = torch.randn(B, H, N, D, device=device, dtype=torch.float16).contiguous()

    # Warmup
    for _ in range(args.warmup):
        _ = sa_forward(Q, K, V, scale)
    torch.cuda.synchronize()

    # Optional correctness check (one run)
    if args.check:
        O_kernel = sa_forward(Q, K, V, scale)
        O_ref = explicit_attention_ref(Q, K, V, scale)
        max_diff = (O_kernel - O_ref).abs().max().item()
        mean_diff = (O_kernel - O_ref).abs().mean().item()
        print(f"[CHECK] max abs diff:  {max_diff:.6f}")
        print(f"[CHECK] mean abs diff: {mean_diff:.6f}")

    # Timing with CUDA events (no sync inside iters)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    if args.profile:
        torch.cuda.synchronize()
        nvtx.range_push("NSYS_CAPTURE")

    starter.record()
    for _ in range(args.iters):
        _ = sa_forward(Q, K, V, scale)
    ender.record()

    if args.profile:
        torch.cuda.synchronize()
        nvtx.range_pop()
    else:
        torch.cuda.synchronize()

    total_ms = starter.elapsed_time(ender)
    avg_ms = total_ms / args.iters
    print(f"[BENCH] B={B} H={H} N={N} D={D} iters={args.iters}")
    print(f"[BENCH] total: {total_ms:.3f} ms  |  avg: {avg_ms:.3f} ms/call")


if __name__ == "__main__":
    main()
