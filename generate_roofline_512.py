#!/usr/bin/env python3
"""
Roofline Model Analysis for ViT Self-Attention Kernels (512x512 only)

RTX 5090 Specs:
- FP16 Tensor Core Peak: 419 TFLOPS
- Memory Bandwidth: 1792 GB/s
- Ridge Point: 419 / 1.792 = 234 FLOPs/Byte
"""
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Hardware Specs (RTX 5090)
# =============================================================================
PEAK_TFLOPS = 419.0        # FP16 Tensor Core peak (TFLOPS)
BANDWIDTH_GBS = 1792.0     # Memory bandwidth (GB/s)
RIDGE_POINT = PEAK_TFLOPS * 1000 / BANDWIDTH_GBS  # ~234 FLOPs/Byte

# =============================================================================
# Configuration (512x512 image, patch_size=16)
# =============================================================================
IMAGE_SIZE = 512
PATCH_SIZE = 16
N = (IMAGE_SIZE // PATCH_SIZE) ** 2 + 1  # 1025 tokens (including CLS)
D = 64   # head dimension (768 / 12 heads)
H = 12   # number of heads
B = 32   # batch size (from benchmark)

dtype_bytes = 2  # FP16

# =============================================================================
# Measured Kernel Times (from scalability_report.txt, 512px)
# =============================================================================
# Attention kernel average time per call (in seconds)
KERNEL_TIMES = {
    'baseline': 2.12e-3,    # 2.12 ms (cuDNN Flash Attention)
    'v1': 129.13e-3,        # 129.13 ms
    'v2': 51.95e-3,         # 51.95 ms
    'v3': 36.64e-3,         # 36.64 ms
    'v4': 20.92e-3,         # 20.92 ms
    'v5': 12.16e-3,         # 12.16 ms
    'v6': 11.22e-3,         # 11.22 ms
}

# =============================================================================
# FLOPs Calculation (Self-Attention)
# =============================================================================
# Q @ K^T: (B*H) x (N x D) @ (D x N) = 2*B*H*N*N*D FLOPs
# softmax: ~5*B*H*N*N FLOPs (exp, sum, div, etc.)
# attn @ V: (B*H) x (N x N) @ (N x D) = 2*B*H*N*N*D FLOPs
# Total: 4*B*H*N^2*D + 5*B*H*N^2 ≈ 4*B*H*N^2*D (softmax is small)

flops_total = 4 * B * H * N * N * D + 5 * B * H * N * N

print("=" * 70)
print("Roofline Model Analysis for ViT Self-Attention (512x512)")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Image: {IMAGE_SIZE}x{IMAGE_SIZE}, Patch: {PATCH_SIZE}x{PATCH_SIZE}")
print(f"  Tokens (N): {N}")
print(f"  Head dim (D): {D}, Heads (H): {H}, Batch (B): {B}")
print(f"\nHardware (RTX 5090):")
print(f"  Peak FP16: {PEAK_TFLOPS} TFLOPS")
print(f"  Bandwidth: {BANDWIDTH_GBS} GB/s")
print(f"  Ridge Point: {RIDGE_POINT:.1f} FLOPs/Byte")
print(f"\nFLOPs per attention call: {flops_total:,} ({flops_total/1e9:.2f} GFLOPs)")

# =============================================================================
# Memory Access Models (Based on actual kernel implementations)
# =============================================================================
def calc_memory_bytes(version):
    """
    Calculate memory bytes accessed for each kernel version.

    Based on actual kernel implementations:
    - Each block processes Q_ROWS queries
    - K/V are loaded in tiles of TK=64
    - Number of Q tiles = ceil(N/Q_ROWS)
    - Each Q tile block loads all of K and V

    Global memory access per batch*head:
    - Q: read once = N*D
    - K: loaded num_q_tiles times (once per Q tile block)
    - V: loaded num_q_tiles times (once per Q tile block)
    - O: write once = N*D
    """
    if version == 'baseline':
        # Flash Attention: highly optimized, minimal memory access
        # Single pass over Q, K, V, O (never materializes attention matrix)
        return B * H * (4 * N * D) * dtype_bytes

    elif version == 'v1':
        # v1: Q_ROWS=16, no proper tiling optimization
        q_rows = 16
        num_q_tiles = (N + q_rows - 1) // q_rows  # ceil(1025/16) = 65
        q_bytes = B * H * N * D * dtype_bytes
        k_bytes = B * H * N * D * dtype_bytes * num_q_tiles
        v_bytes = B * H * N * D * dtype_bytes * num_q_tiles
        o_bytes = B * H * N * D * dtype_bytes
        return q_bytes + k_bytes + v_bytes + o_bytes

    elif version == 'v2':
        # v2: Q_ROWS=16, some coalescing improvements
        q_rows = 16
        num_q_tiles = (N + q_rows - 1) // q_rows
        q_bytes = B * H * N * D * dtype_bytes
        k_bytes = B * H * N * D * dtype_bytes * num_q_tiles
        v_bytes = B * H * N * D * dtype_bytes * num_q_tiles
        o_bytes = B * H * N * D * dtype_bytes
        return q_bytes + k_bytes + v_bytes + o_bytes

    elif version == 'v3':
        # v3: Q_ROWS=16, WMMA optimization
        q_rows = 16
        num_q_tiles = (N + q_rows - 1) // q_rows
        q_bytes = B * H * N * D * dtype_bytes
        k_bytes = B * H * N * D * dtype_bytes * num_q_tiles
        v_bytes = B * H * N * D * dtype_bytes * num_q_tiles
        o_bytes = B * H * N * D * dtype_bytes
        return q_bytes + k_bytes + v_bytes + o_bytes

    elif version == 'v4':
        # v4: Q_ROWS=64, 4 warps, better parallelism
        q_rows = 64
        num_q_tiles = (N + q_rows - 1) // q_rows  # ceil(1025/64) = 17
        q_bytes = B * H * N * D * dtype_bytes
        k_bytes = B * H * N * D * dtype_bytes * num_q_tiles
        v_bytes = B * H * N * D * dtype_bytes * num_q_tiles
        o_bytes = B * H * N * D * dtype_bytes
        return q_bytes + k_bytes + v_bytes + o_bytes

    elif version in ['v5', 'v6']:
        # v5/v6: Q_ROWS=128, 8 warps, best memory efficiency
        q_rows = 128
        num_q_tiles = (N + q_rows - 1) // q_rows  # ceil(1025/128) = 9
        q_bytes = B * H * N * D * dtype_bytes
        k_bytes = B * H * N * D * dtype_bytes * num_q_tiles
        v_bytes = B * H * N * D * dtype_bytes * num_q_tiles
        o_bytes = B * H * N * D * dtype_bytes
        return q_bytes + k_bytes + v_bytes + o_bytes

# =============================================================================
# Calculate Arithmetic Intensity and Achieved Performance
# =============================================================================
print("\n" + "=" * 70)
print("Per-Version Analysis")
print("=" * 70)

results = {}
for version in ['baseline', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']:
    time_s = KERNEL_TIMES[version]
    mem_bytes = calc_memory_bytes(version)

    # Arithmetic Intensity = FLOPs / Bytes
    ai = flops_total / mem_bytes

    # Achieved Performance = FLOPs / Time
    achieved_tflops = (flops_total / time_s) / 1e12

    # Roofline bound
    memory_bound_perf = ai * BANDWIDTH_GBS / 1000  # TFLOPS
    roofline_bound = min(PEAK_TFLOPS, memory_bound_perf)

    # Efficiency
    efficiency = achieved_tflops / roofline_bound * 100

    results[version] = {
        'ai': ai,
        'achieved_tflops': achieved_tflops,
        'mem_bytes': mem_bytes,
        'roofline_bound': roofline_bound,
        'efficiency': efficiency,
        'is_memory_bound': ai < RIDGE_POINT
    }

    bound_type = "Memory" if ai < RIDGE_POINT else "Compute"
    print(f"\n{version}:")
    print(f"  Time: {time_s*1000:.2f} ms")
    print(f"  Memory: {mem_bytes/1e9:.2f} GB")
    print(f"  Arithmetic Intensity: {ai:.2f} FLOPs/Byte")
    print(f"  Achieved: {achieved_tflops:.2f} TFLOPS")
    print(f"  Roofline Bound: {roofline_bound:.2f} TFLOPS ({bound_type}-bound)")
    print(f"  Efficiency: {efficiency:.1f}%")

# =============================================================================
# Generate Roofline Plot
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Plot limits
x_min, x_max = 5, 800
y_min, y_max = 0.5, 600

# Fill background regions for memory-bound and compute-bound areas
# Memory-bound region: left of ridge point (light blue)
ax.axvspan(x_min, RIDGE_POINT, alpha=0.15, color='blue', zorder=0)
# Compute-bound region: right of ridge point (light red)
ax.axvspan(RIDGE_POINT, x_max, alpha=0.15, color='red', zorder=0)

# Add region labels
ax.text(30, 300, 'Memory\nBound', fontsize=14, fontweight='bold',
        color='blue', alpha=0.5, ha='center', va='center')
ax.text(450, 300, 'Compute\nBound', fontsize=14, fontweight='bold',
        color='red', alpha=0.5, ha='center', va='center')

# Roofline lines
ai_range = np.logspace(0, 3, 500)  # 1 to 1000
memory_bound = ai_range * BANDWIDTH_GBS / 1000  # Convert to TFLOPS
roofline = np.minimum(memory_bound, PEAK_TFLOPS)

# Plot roofline
ax.loglog(ai_range, memory_bound, 'b--', linewidth=2, alpha=0.7,
          label=f'Memory Bound ({BANDWIDTH_GBS:.0f} GB/s)')
ax.axhline(y=PEAK_TFLOPS, color='r', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Compute Bound ({PEAK_TFLOPS:.0f} TFLOPS)')
ax.loglog(ai_range, roofline, 'k-', linewidth=3, alpha=0.9, label='Roofline')

# Ridge point annotation
ax.axvline(x=RIDGE_POINT, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.annotate(f'Ridge Point\n({RIDGE_POINT:.0f} FLOPs/B)',
            xy=(RIDGE_POINT, PEAK_TFLOPS * 0.6), fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# Colors and markers
colors = {
    'baseline': '#2ecc71',  # green
    'v1': '#e74c3c',        # red
    'v2': '#3498db',        # blue
    'v3': '#9b59b6',        # purple
    'v4': '#f39c12',        # orange
    'v5': '#00bcd4',        # cyan
    'v6': '#ff5722',        # deep orange
}

markers = {
    'baseline': 'o',
    'v1': 's',
    'v2': '^',
    'v3': 'D',
    'v4': 'p',
    'v5': 'h',
    'v6': '*',
}

marker_sizes = {
    'baseline': 250,
    'v1': 180,
    'v2': 180,
    'v3': 180,
    'v4': 200,
    'v5': 220,
    'v6': 280,
}

# Plot each version
for version, data in results.items():
    ax.scatter(data['ai'], data['achieved_tflops'],
               c=colors[version], marker=markers[version],
               s=marker_sizes[version], edgecolors='black', linewidth=1.5, zorder=5,
               label=f"{version}: {data['achieved_tflops']:.2f} TFLOPS (AI={data['ai']:.1f})")

# Formatting
ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=14)
ax.set_ylabel('Performance (TFLOPS)', fontsize=14)
ax.set_title('Roofline Model: ViT Self-Attention Kernels\n(512×512 image, RTX 5090)', fontsize=16)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(True, which='both', linestyle='--', alpha=0.3)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
output_path = 'results/scalability/roofline_512.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n{'=' * 70}")
print(f"Plot saved to: {output_path}")
print("=" * 70)

plt.show()
