#!/bin/bash
#
# Scalability Benchmark Script
# Tests different kernel versions across multiple image sizes (token counts)
#

set -e

echo "========================================="
echo "Loading environment variables from .env"
echo "========================================="
source .env

# Verify CUDA_HOME is set
if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    echo "Error: CUDA_HOME is not set or is not a valid directory."
    exit 1
fi

echo "CUDA_HOME: $CUDA_HOME"
nvcc --version

# =========================================
# Configuration
# =========================================
IMAGE_SIZES=(224 384 512)
PATCH_SIZE=16
# Note: v0 is excluded due to extremely slow performance at large token counts
VERSIONS=(baseline v1 v2 v3 v4 v5 v6)
BATCH_SIZE=32
NUM_BATCHES=10

RESULTS_DIR="results/scalability"
mkdir -p "$RESULTS_DIR"

# =========================================
# Compile all Custom CUDA Kernels
# =========================================
echo ""
echo "========================================="
echo "Compiling Custom CUDA Kernels"
echo "========================================="

# Skip v0 due to slow performance
for ver in v1 v2 v3 v4 v5 v6; do
    echo "[${ver}] Compiling..."
    (cd custom/${ver} && uv run python setup.py install)
done

# =========================================
# Run Benchmarks
# =========================================
echo ""
echo "========================================="
echo "Running Scalability Benchmarks"
echo "========================================="

for img_size in "${IMAGE_SIZES[@]}"; do
    num_tokens=$(( (img_size / PATCH_SIZE) * (img_size / PATCH_SIZE) + 1 ))

    for ver in "${VERSIONS[@]}"; do
        echo ""
        echo "-----------------------------------------"
        echo "Testing: image_size=${img_size} (${num_tokens} tokens), version=${ver}"
        echo "-----------------------------------------"

        profile_name="${RESULTS_DIR}/${ver}_${img_size}"

        "$CUDA_HOME/bin/nsys" profile \
            -o "${profile_name}" \
            --force-overwrite true \
            uv run python scalability_bench.py \
                --image-size ${img_size} \
                --patch-size ${PATCH_SIZE} \
                --version ${ver} \
                --batch-size ${BATCH_SIZE} \
                --num-batches ${NUM_BATCHES}

        echo "Profile saved: ${profile_name}.nsys-rep"
    done
done

# =========================================
# Summary
# =========================================
echo ""
echo "========================================="
echo "Scalability Benchmark Complete!"
echo "========================================="
echo "Results saved in: ${RESULTS_DIR}/"
echo ""
echo "Generated profiles:"
for img_size in "${IMAGE_SIZES[@]}"; do
    for ver in "${VERSIONS[@]}"; do
        echo "  - ${ver}_${img_size}.nsys-rep"
    done
done
echo ""
echo "To analyze results, run:"
echo "  uv run python analyze_scalability.py"
echo "========================================="
