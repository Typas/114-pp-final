#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================="
echo "Loading environment variables from .env"
echo "========================================="
source .env

# Verify CUDA_HOME is set and is a directory
if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    echo "Error: CUDA_HOME is not set or is not a valid directory. Please check your .env file."
    exit 1
fi

# Verify CUDA is found
if ! command -v nvcc &> /dev/null
then
    echo "nvcc could not be found. Please check your .env file."
    exit 1
fi
nvcc --version

echo "========================================="
echo "Setting up results directory"
echo "========================================="
mkdir -p results

# =========================================
# Compile all Custom CUDA Kernels
# =========================================
echo "========================================="
echo "Compiling Custom CUDA Kernels"
echo "========================================="

echo "[v0] Compiling..."
(cd custom/v0 && uv run python setup.py install)

echo "[v1] Compiling..."
(cd custom/v1 && uv run python setup.py install)

echo "[v2] Compiling..."
(cd custom/v2 && uv run python setup.py install)

echo "[v3] Compiling..."
(cd custom/v3 && uv run python setup.py install)

# =========================================
# Run Baseline Benchmark
# =========================================
echo "========================================="
echo "Running Baseline Benchmark"
echo "========================================="
"$CUDA_HOME/bin/nsys" profile -o results/baseline_profile --force-overwrite true \
    uv run python base/inference_vit_pytorch_nvtx.py

# =========================================
# Run Custom Kernel Benchmarks (v0 - v3)
# =========================================
echo "========================================="
echo "Running Custom Kernel v0 Benchmark"
echo "========================================="
"$CUDA_HOME/bin/nsys" profile -o results/custom_v0_profile --force-overwrite true \
    uv run python custom/v0/inference_vit_pytorch_nvtx.py

echo "========================================="
echo "Running Custom Kernel v1 Benchmark"
echo "========================================="
"$CUDA_HOME/bin/nsys" profile -o results/custom_v1_profile --force-overwrite true \
    uv run python custom/v1/inference_vit_pytorch_nvtx_v1.py

echo "========================================="
echo "Running Custom Kernel v2 Benchmark"
echo "========================================="
"$CUDA_HOME/bin/nsys" profile -o results/custom_v2_profile --force-overwrite true \
    uv run python custom/v2/inference_vit_pytorch_nvtx_v2.py

echo "========================================="
echo "Running Custom Kernel v3 Benchmark"
echo "========================================="
"$CUDA_HOME/bin/nsys" profile -o results/custom_v3_profile --force-overwrite true \
    uv run python custom/v3/inference_vit_pytorch_nvtx_v3.py

# =========================================
# Summary
# =========================================
echo ""
echo "========================================="
echo "Benchmarking Complete!"
echo "========================================="
echo "Reports generated in results/ directory:"
echo "  - baseline_profile.nsys-rep"
echo "  - custom_v0_profile.nsys-rep"
echo "  - custom_v1_profile.nsys-rep"
echo "  - custom_v2_profile.nsys-rep"
echo "  - custom_v3_profile.nsys-rep"
echo ""
echo "To view reports:"
echo "  nsys-ui results/baseline_profile.nsys-rep"
echo "========================================="
