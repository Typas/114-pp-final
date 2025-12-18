"""
Analyze PyTorch baseline attention performance
"""
import torch
import torch.nn.functional as F

def main():
    device = torch.device("cuda")
    torch.manual_seed(42)

    B, H, N, D = 32, 12, 197, 64
    scale = D ** -0.5

    Q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

    # Check if scaled_dot_product_attention is available (PyTorch 2.0+)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    # Test different attention implementations
    print("\n--- Testing Different Attention Implementations ---")

    # 1. Manual matmul + softmax
    def manual_attention(q, k, v, scale):
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    # Warmup
    for _ in range(10):
        _ = manual_attention(Q, K, V, scale)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        _ = manual_attention(Q, K, V, scale)
    end.record()
    torch.cuda.synchronize()
    print(f"Manual matmul+softmax: {start.elapsed_time(end)/100:.4f} ms")

    # 2. F.scaled_dot_product_attention (if available)
    if hasattr(F, 'scaled_dot_product_attention'):
        print("\nF.scaled_dot_product_attention available!")

        # Check backends
        print(f"  Flash Attention: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"  Mem Efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"  Math: {torch.backends.cuda.math_sdp_enabled()}")

        for _ in range(10):
            _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
        torch.cuda.synchronize()

        start.record()
        for _ in range(100):
            _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
        end.record()
        torch.cuda.synchronize()
        print(f"F.scaled_dot_product_attention: {start.elapsed_time(end)/100:.4f} ms")

        # Try forcing different backends
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            for _ in range(10):
                _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
            torch.cuda.synchronize()
            start.record()
            for _ in range(100):
                _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
            end.record()
            torch.cuda.synchronize()
            print(f"  (math only): {start.elapsed_time(end)/100:.4f} ms")

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            try:
                for _ in range(10):
                    _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
                torch.cuda.synchronize()
                start.record()
                for _ in range(100):
                    _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
                end.record()
                torch.cuda.synchronize()
                print(f"  (flash only): {start.elapsed_time(end)/100:.4f} ms")
            except Exception as e:
                print(f"  (flash only): Not available - {e}")

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            try:
                for _ in range(10):
                    _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
                torch.cuda.synchronize()
                start.record()
                for _ in range(100):
                    _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
                end.record()
                torch.cuda.synchronize()
                print(f"  (mem_efficient only): {start.elapsed_time(end)/100:.4f} ms")
            except Exception as e:
                print(f"  (mem_efficient only): Not available - {e}")

if __name__ == "__main__":
    main()
