#!/usr/bin/env python3
"""
Analyze nsys profiling results and generate a summary report.
"""

import subprocess
import re
import os
import csv
import io
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_FILE = RESULTS_DIR / "benchmark_report.txt"

PROFILES = [
    ("baseline", "baseline_profile.nsys-rep"),
    ("v0", "custom_v0_profile.nsys-rep"),
    ("v1", "custom_v1_profile.nsys-rep"),
    ("v2", "custom_v2_profile.nsys-rep"),
    ("v3", "custom_v3_profile.nsys-rep"),
    ("v4", "custom_v4_profile.nsys-rep"),
    ("v5", "custom_v5_profile.nsys-rep"),
    ("v6", "custom_v6_profile.nsys-rep"),
]


def run_nsys_stats(nsys_rep_path, report_type):
    """Run nsys stats and return output."""
    cmd = [
        "/usr/local/cuda-12.8/bin/nsys", "stats",
        "--report", report_type,
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

    # Find the header line and data lines
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

    # Use csv module to properly parse (handles quoted fields with commas)
    reader = csv.reader(io.StringIO('\n'.join(data_lines)))
    header = next(reader, None)

    if not header:
        return kernels

    for row in reader:
        if len(row) >= 9:  # Time%, Total, Instances, Avg, Med, Min, Max, StdDev, Name
            try:
                time_pct = float(row[0])
                total_time_ns = int(row[1])
                instances = int(row[2])
                avg_ns = float(row[3])
                name = row[8]  # Name is the 9th column (index 8)
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


def get_softmax_kernels(kernels):
    """Filter for softmax-related kernels."""
    softmax_kernels = []
    for k in kernels:
        name_lower = k['name'].lower()
        if 'softmax' in name_lower or 'sa_' in name_lower or 'self_attention' in name_lower:
            softmax_kernels.append(k)
    return softmax_kernels


def get_attention_kernels(kernels):
    """Get kernels related to attention (GEMM, softmax, etc.)."""
    attention_related = []
    for k in kernels:
        name_lower = k['name'].lower()
        # GEMM kernels (Q*K, attn*V)
        if 'gemm' in name_lower or 'cutlass' in name_lower or 'magma' in name_lower:
            attention_related.append(k)
        # Softmax
        elif 'softmax' in name_lower:
            attention_related.append(k)
        # Custom kernels
        elif 'sa_' in name_lower:
            attention_related.append(k)
    return attention_related


def format_time(ns):
    """Format nanoseconds to readable format."""
    if ns >= 1e9:
        return f"{ns/1e9:.2f} s"
    elif ns >= 1e6:
        return f"{ns/1e6:.2f} ms"
    elif ns >= 1e3:
        return f"{ns/1e3:.2f} us"
    else:
        return f"{ns:.0f} ns"


def generate_report():
    """Generate comprehensive benchmark report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ViT Self-Attention CUDA Optimization Benchmark Report")
    report_lines.append("=" * 80)
    report_lines.append("")

    all_results = {}

    for name, filename in PROFILES:
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            report_lines.append(f"[{name}] File not found: {filepath}")
            continue

        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"Profile: {name.upper()}")
        report_lines.append(f"{'='*80}")

        # Get kernel summary
        csv_output = run_nsys_stats(filepath, "cuda_gpu_kern_sum")
        kernels = parse_kernel_csv(csv_output)

        if not kernels:
            report_lines.append("No kernel data found")
            continue

        # Calculate totals
        total_gpu_time = sum(k['total_time_ns'] for k in kernels)

        report_lines.append(f"\nTotal GPU Time: {format_time(total_gpu_time)}")
        report_lines.append(f"Total Kernel Calls: {sum(k['instances'] for k in kernels)}")

        # Top 10 kernels
        report_lines.append(f"\n--- Top 10 Kernels by Time ---")
        report_lines.append(f"{'Time %':>8} {'Total Time':>12} {'Calls':>8} {'Avg Time':>12}  Kernel Name")
        report_lines.append("-" * 100)

        for k in kernels[:10]:
            short_name = k['name'][:60] + "..." if len(k['name']) > 60 else k['name']
            report_lines.append(
                f"{k['time_pct']:>7.1f}% {format_time(k['total_time_ns']):>12} "
                f"{k['instances']:>8} {format_time(k['avg_ns']):>12}  {short_name}"
            )

        # Softmax / Self-Attention specific
        softmax_kernels = get_softmax_kernels(kernels)
        if softmax_kernels:
            report_lines.append(f"\n--- Softmax / Self-Attention Kernels ---")
            softmax_total = sum(k['total_time_ns'] for k in softmax_kernels)
            report_lines.append(f"Total Softmax Time: {format_time(softmax_total)} ({100*softmax_total/total_gpu_time:.1f}%)")
            for k in softmax_kernels:
                short_name = k['name'][:60] + "..." if len(k['name']) > 60 else k['name']
                report_lines.append(
                    f"  {k['time_pct']:>6.1f}% {format_time(k['total_time_ns']):>10} "
                    f"x{k['instances']:<6} avg {format_time(k['avg_ns']):>10}  {short_name}"
                )

        # Store for comparison
        all_results[name] = {
            'total_gpu_time': total_gpu_time,
            'kernels': kernels,
            'softmax_kernels': softmax_kernels,
            'softmax_time': sum(k['total_time_ns'] for k in softmax_kernels) if softmax_kernels else 0
        }

    # Comparison section
    if 'baseline' in all_results and len(all_results) > 1:
        report_lines.append(f"\n\n{'='*80}")
        report_lines.append("COMPARISON SUMMARY")
        report_lines.append(f"{'='*80}")

        baseline_time = all_results['baseline']['total_gpu_time']
        baseline_softmax = all_results['baseline']['softmax_time']

        report_lines.append(f"\n{'Version':<12} {'Total GPU Time':>15} {'Speedup':>10} {'Softmax Time':>15} {'Softmax Speedup':>15}")
        report_lines.append("-" * 70)

        for name in ['baseline', 'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']:
            if name not in all_results:
                continue
            r = all_results[name]
            speedup = baseline_time / r['total_gpu_time'] if r['total_gpu_time'] > 0 else 0
            softmax_speedup = baseline_softmax / r['softmax_time'] if r['softmax_time'] > 0 else float('inf')

            speedup_str = f"{speedup:.2f}x" if name != 'baseline' else "1.00x"
            softmax_speedup_str = f"{softmax_speedup:.2f}x" if name != 'baseline' and softmax_speedup != float('inf') else ("1.00x" if name == 'baseline' else "N/A")

            report_lines.append(
                f"{name:<12} {format_time(r['total_gpu_time']):>15} {speedup_str:>10} "
                f"{format_time(r['softmax_time']):>15} {softmax_speedup_str:>15}"
            )

        # Bar chart visualization
        report_lines.append(f"\n--- Performance Visualization (Total GPU Time) ---")
        max_time = max(r['total_gpu_time'] for r in all_results.values())
        bar_width = 50

        for name in ['baseline', 'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']:
            if name not in all_results:
                continue
            r = all_results[name]
            bar_len = int(bar_width * r['total_gpu_time'] / max_time)
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            report_lines.append(f"{name:>10}: [{bar}] {format_time(r['total_gpu_time'])}")

    report_lines.append(f"\n\n{'='*80}")
    report_lines.append("Report generated successfully!")
    report_lines.append(f"{'='*80}")

    # Write report
    report_text = "\n".join(report_lines)

    with open(OUTPUT_FILE, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_report()
