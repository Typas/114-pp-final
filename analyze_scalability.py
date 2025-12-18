#!/usr/bin/env python3
"""
Analyze scalability benchmark results and generate a comprehensive report.
"""

import subprocess
import csv
import io
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results" / "scalability"
OUTPUT_FILE = RESULTS_DIR / "scalability_report.txt"
PLOT_FILE = RESULTS_DIR / "scalability_plot.png"

IMAGE_SIZES = [224, 384, 512]
PATCH_SIZE = 16
VERSIONS = ['baseline', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']  # v0 excluded due to slow performance

# Plot styling
VERSION_COLORS = {
    'baseline': '#2ecc71',  # green
    'v0': '#e74c3c',        # red
    'v1': '#3498db',        # blue
    'v2': '#9b59b6',        # purple
    'v3': '#f39c12',        # orange
    'v4': '#e91e63',        # pink
    'v5': '#00bcd4',        # cyan
    'v6': '#ff5722',        # deep orange
}
VERSION_MARKERS = {
    'baseline': 'o',
    'v0': 's',
    'v1': '^',
    'v2': 'D',
    'v3': 'v',
    'v4': 'p',              # pentagon
    'v5': 'h',              # hexagon
    'v6': '*',              # star
}


def run_nsys_stats(nsys_rep_path):
    """Run nsys stats and return kernel summary."""
    cmd = [
        "/usr/local/cuda-12.8/bin/nsys", "stats",
        "--report", "cuda_gpu_kern_sum",
        "--format", "csv",
        "--force-export", "true",
        str(nsys_rep_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


def parse_kernel_csv(csv_output):
    """Parse CUDA kernel summary CSV output."""
    lines = csv_output.strip().split('\n')
    kernels = []

    data_lines = []
    header_found = False
    for line in lines:
        if 'Time (%)' in line:
            header_found = True
            data_lines.append(line)
            continue
        if header_found and line.strip():
            data_lines.append(line)

    if not data_lines:
        return kernels

    reader = csv.reader(io.StringIO('\n'.join(data_lines)))
    header = next(reader, None)

    if not header:
        return kernels

    for row in reader:
        if len(row) >= 9:
            try:
                time_pct = float(row[0])
                total_time_ns = int(row[1])
                instances = int(row[2])
                avg_ns = float(row[3])
                name = row[8]
                kernels.append({
                    'time_pct': time_pct,
                    'total_time_ns': total_time_ns,
                    'instances': instances,
                    'avg_ns': avg_ns,
                    'name': name
                })
            except (ValueError, IndexError):
                continue
    return kernels


def get_attention_kernel_time(kernels, version):
    """Get the self-attention kernel time for a specific version."""
    for k in kernels:
        name = k['name']
        name_lower = name.lower()
        if version == 'baseline':
            if 'softmax' in name_lower:
                return k['total_time_ns'], k['avg_ns'], k['instances']
        else:
            # Check for custom kernel names (case-sensitive for template names)
            # e.g., sa_forward_v1_kernel<...>, sa_forward_v2_kernel<...>
            if f'sa_forward_{version}_kernel' in name or f'sa_{version}_kernel' in name:
                return k['total_time_ns'], k['avg_ns'], k['instances']
            # Also check lowercase for other formats
            if f'sa_forward_{version}' in name_lower:
                return k['total_time_ns'], k['avg_ns'], k['instances']
    return None, None, None


def format_time(ns):
    """Format nanoseconds to readable format."""
    if ns is None:
        return "N/A"
    if ns >= 1e9:
        return f"{ns/1e9:.2f} s"
    elif ns >= 1e6:
        return f"{ns/1e6:.2f} ms"
    elif ns >= 1e3:
        return f"{ns/1e3:.2f} us"
    else:
        return f"{ns:.0f} ns"


def plot_scalability_single(results, versions_to_plot, suffix=""):
    """Generate scalability plot for a specific set of versions."""
    token_counts = [(img_size // PATCH_SIZE) ** 2 + 1 for img_size in IMAGE_SIZES]

    # ===== Linear scale plots =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Attention Kernel Average Time
    ax1 = axes[0]
    for version in versions_to_plot:
        times_ms = []
        valid_tokens = []
        for img_size, tokens in zip(IMAGE_SIZES, token_counts):
            if img_size in results and version in results[img_size]:
                avg_time = results[img_size][version].get('attn_avg_time')
                if avg_time is not None:
                    times_ms.append(avg_time / 1e6)  # ns to ms
                    valid_tokens.append(tokens)

        if times_ms:
            ax1.plot(valid_tokens, times_ms,
                    color=VERSION_COLORS.get(version, 'gray'),
                    marker=VERSION_MARKERS.get(version, 'o'),
                    markersize=10, linewidth=2, label=version)

    ax1.set_xlabel('Token Count (N)', fontsize=12)
    ax1.set_ylabel('Avg Kernel Time (ms)', fontsize=12)
    ax1.set_title('Self-Attention Kernel Time vs Token Count', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(token_counts)
    ax1.set_xticklabels([f'{t}\n({s}px)' for t, s in zip(token_counts, IMAGE_SIZES)])

    # Plot 2: Total GPU Time
    ax2 = axes[1]
    for version in versions_to_plot:
        times_ms = []
        valid_tokens = []
        for img_size, tokens in zip(IMAGE_SIZES, token_counts):
            if img_size in results and version in results[img_size]:
                total_time = results[img_size][version].get('total_gpu_time')
                if total_time is not None:
                    times_ms.append(total_time / 1e6)  # ns to ms
                    valid_tokens.append(tokens)

        if times_ms:
            ax2.plot(valid_tokens, times_ms,
                    color=VERSION_COLORS.get(version, 'gray'),
                    marker=VERSION_MARKERS.get(version, 'o'),
                    markersize=10, linewidth=2, label=version)

    ax2.set_xlabel('Token Count (N)', fontsize=12)
    ax2.set_ylabel('Total GPU Time (ms)', fontsize=12)
    ax2.set_title('Total GPU Time vs Token Count', fontsize=14)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(token_counts)
    ax2.set_xticklabels([f'{t}\n({s}px)' for t, s in zip(token_counts, IMAGE_SIZES)])

    plt.tight_layout()
    plot_file = RESULTS_DIR / f"scalability_plot{suffix}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {plot_file}")

    # ===== Log scale plots =====
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Attention Kernel Time (log scale)
    ax3 = axes2[0]
    for version in versions_to_plot:
        times_ms = []
        valid_tokens = []
        for img_size, tokens in zip(IMAGE_SIZES, token_counts):
            if img_size in results and version in results[img_size]:
                avg_time = results[img_size][version].get('attn_avg_time')
                if avg_time is not None:
                    times_ms.append(avg_time / 1e6)
                    valid_tokens.append(tokens)

        if times_ms:
            ax3.plot(valid_tokens, times_ms,
                    color=VERSION_COLORS.get(version, 'gray'),
                    marker=VERSION_MARKERS.get(version, 'o'),
                    markersize=10, linewidth=2, label=version)

    ax3.set_xlabel('Token Count (N)', fontsize=12)
    ax3.set_ylabel('Avg Kernel Time (ms) - Log Scale', fontsize=12)
    ax3.set_title('Self-Attention Kernel Time (Log Scale)', fontsize=14)
    ax3.set_yscale('log')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xticks(token_counts)
    ax3.set_xticklabels([f'{t}\n({s}px)' for t, s in zip(token_counts, IMAGE_SIZES)])

    # Plot 2: Speedup relative to baseline
    ax4 = axes2[1]
    for version in versions_to_plot:
        if version == 'baseline':
            continue
        speedups = []
        valid_tokens = []
        for img_size, tokens in zip(IMAGE_SIZES, token_counts):
            if img_size in results and version in results[img_size] and 'baseline' in results[img_size]:
                baseline_time = results[img_size]['baseline'].get('attn_avg_time')
                version_time = results[img_size][version].get('attn_avg_time')
                if baseline_time and version_time and version_time > 0:
                    speedups.append(baseline_time / version_time)
                    valid_tokens.append(tokens)

        if speedups:
            ax4.plot(valid_tokens, speedups,
                    color=VERSION_COLORS.get(version, 'gray'),
                    marker=VERSION_MARKERS.get(version, 'o'),
                    markersize=10, linewidth=2, label=version)

    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='baseline (1.0x)')
    ax4.set_xlabel('Token Count (N)', fontsize=12)
    ax4.set_ylabel('Speedup vs Baseline', fontsize=12)
    ax4.set_title('Attention Kernel Speedup vs Baseline', fontsize=14)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(token_counts)
    ax4.set_xticklabels([f'{t}\n({s}px)' for t, s in zip(token_counts, IMAGE_SIZES)])

    plt.tight_layout()
    log_plot_file = RESULTS_DIR / f"scalability_plot_log{suffix}.png"
    plt.savefig(log_plot_file, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Log scale plot saved to: {log_plot_file}")


def plot_scalability(results):
    """Generate scalability plots - both with and without v0."""
    # Version 1: All versions including v0
    print("\n--- Generating plots WITH v0 ---")
    plot_scalability_single(results, VERSIONS, suffix="_with_v0")

    # Version 2: Without v0
    versions_no_v0 = [v for v in VERSIONS if v != 'v0']
    print("\n--- Generating plots WITHOUT v0 ---")
    plot_scalability_single(results, versions_no_v0, suffix="_no_v0")


def generate_report():
    """Generate comprehensive scalability report."""
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("ViT Self-Attention Kernel Scalability Report")
    report_lines.append("=" * 90)
    report_lines.append("")

    # Data structure: results[image_size][version] = {...}
    results = defaultdict(dict)

    # Collect all results
    for img_size in IMAGE_SIZES:
        num_tokens = (img_size // PATCH_SIZE) ** 2 + 1

        for version in VERSIONS:
            filepath = RESULTS_DIR / f"{version}_{img_size}.nsys-rep"
            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping...")
                continue

            csv_output = run_nsys_stats(filepath)
            kernels = parse_kernel_csv(csv_output)

            if not kernels:
                continue

            total_gpu_time = sum(k['total_time_ns'] for k in kernels)
            attn_total, attn_avg, attn_calls = get_attention_kernel_time(kernels, version)

            results[img_size][version] = {
                'total_gpu_time': total_gpu_time,
                'attn_total_time': attn_total,
                'attn_avg_time': attn_avg,
                'attn_calls': attn_calls,
                'num_tokens': num_tokens,
            }

    # Generate per-image-size tables
    for img_size in IMAGE_SIZES:
        num_tokens = (img_size // PATCH_SIZE) ** 2 + 1
        report_lines.append(f"\n{'='*90}")
        report_lines.append(f"Image Size: {img_size}x{img_size} | Tokens: {num_tokens}")
        report_lines.append(f"{'='*90}")

        if img_size not in results or not results[img_size]:
            report_lines.append("No data available")
            continue

        report_lines.append(f"\n{'Version':<12} {'Total GPU':>14} {'Attn Kernel':>14} {'Attn Avg':>12} {'Calls':>8}")
        report_lines.append("-" * 70)

        baseline_time = results[img_size].get('baseline', {}).get('total_gpu_time', 1)

        for version in VERSIONS:
            if version not in results[img_size]:
                continue
            r = results[img_size][version]
            speedup = baseline_time / r['total_gpu_time'] if r['total_gpu_time'] > 0 else 0
            report_lines.append(
                f"{version:<12} {format_time(r['total_gpu_time']):>14} "
                f"{format_time(r['attn_total_time']):>14} {format_time(r['attn_avg_time']):>12} "
                f"{r['attn_calls'] or 'N/A':>8}"
            )

    # Generate comparison table across image sizes
    report_lines.append(f"\n\n{'='*90}")
    report_lines.append("CROSS-SIZE COMPARISON: Attention Kernel Average Time (per call)")
    report_lines.append(f"{'='*90}")

    # Header
    header = f"{'Version':<12}"
    for img_size in IMAGE_SIZES:
        num_tokens = (img_size // PATCH_SIZE) ** 2 + 1
        header += f" {img_size} ({num_tokens}t):>18"
    report_lines.append("")
    report_lines.append(f"{'Version':<12} " + " ".join([f"{s}px ({(s//PATCH_SIZE)**2+1}t)".center(18) for s in IMAGE_SIZES]))
    report_lines.append("-" * (12 + 19 * len(IMAGE_SIZES)))

    for version in VERSIONS:
        row = f"{version:<12}"
        for img_size in IMAGE_SIZES:
            if img_size in results and version in results[img_size]:
                avg_time = results[img_size][version]['attn_avg_time']
                row += f" {format_time(avg_time):>17}"
            else:
                row += f" {'N/A':>17}"
        report_lines.append(row)

    # Scalability analysis
    report_lines.append(f"\n\n{'='*90}")
    report_lines.append("SCALABILITY ANALYSIS: Time Growth Factor (relative to 224px)")
    report_lines.append(f"{'='*90}")
    report_lines.append(f"\nToken count ratio: 224->384 = {((384//16)**2+1) / ((224//16)**2+1):.2f}x, "
                       f"224->512 = {((512//16)**2+1) / ((224//16)**2+1):.2f}x")
    report_lines.append(f"Theoretical O(N^2) growth: 384 = {(((384//16)**2+1) / ((224//16)**2+1))**2:.2f}x, "
                       f"512 = {(((512//16)**2+1) / ((224//16)**2+1))**2:.2f}x")
    report_lines.append("")
    report_lines.append(f"{'Version':<12} {'224->384 growth':>18} {'224->512 growth':>18}")
    report_lines.append("-" * 50)

    for version in VERSIONS:
        if 224 not in results or version not in results[224]:
            continue
        base_time = results[224][version].get('attn_avg_time')
        if base_time is None or base_time == 0:
            continue

        row = f"{version:<12}"
        for target_size in [384, 512]:
            if target_size in results and version in results[target_size]:
                target_time = results[target_size][version].get('attn_avg_time')
                if target_time:
                    growth = target_time / base_time
                    row += f" {growth:>17.2f}x"
                else:
                    row += f" {'N/A':>17}"
            else:
                row += f" {'N/A':>17}"
        report_lines.append(row)

    # Performance visualization
    report_lines.append(f"\n\n{'='*90}")
    report_lines.append("PERFORMANCE VISUALIZATION: Attention Kernel Time by Image Size")
    report_lines.append(f"{'='*90}")

    for img_size in IMAGE_SIZES:
        if img_size not in results:
            continue

        report_lines.append(f"\n--- {img_size}x{img_size} ({(img_size//PATCH_SIZE)**2+1} tokens) ---")

        # Find max time for this image size
        max_time = max(
            (r.get('attn_avg_time') or 0) for r in results[img_size].values()
        )
        if max_time == 0:
            continue

        bar_width = 50
        for version in VERSIONS:
            if version not in results[img_size]:
                continue
            avg_time = results[img_size][version].get('attn_avg_time')
            if avg_time is None:
                continue
            bar_len = int(bar_width * avg_time / max_time)
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            report_lines.append(f"  {version:>10}: [{bar}] {format_time(avg_time)}")

    report_lines.append(f"\n\n{'='*90}")
    report_lines.append("Report generated successfully!")
    report_lines.append(f"{'='*90}")

    # Write report
    report_text = "\n".join(report_lines)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n\nReport saved to: {OUTPUT_FILE}")

    # Generate plots
    if results:
        plot_scalability(results)

    return results


if __name__ == "__main__":
    generate_report()
