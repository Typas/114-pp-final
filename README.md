# Accelerating Vision Transformer Inference with Custom CUDA Kernels

This is a repository for NTU Parallel Programming Final Project. We aim to accelerate the Self-Attention mechanism in [Vision Transformer (ViT)](https://github.com/lucidrains/vit-pytorch) by implementing different custom CUDA kernels.

## Project Overview

TBD

## Project Structure

```
.
├── base/                           # Baseline implementation
│   ├── inference_vit_pytorch.py    # ViT inference using vit-pytorch
│   ├── train_vit_pytorch.py        # Training script for ViT on CIFAR-10
│   └── vit_cifar10.pth             # Pretrained weights
├── custom/                         
│   ├── v0                          
│   ├── v1                          
│   ├── v2                          
│   └── v3                          
├── data/                           # Dataset directory (CIFAR-10)
├── results/                        # Nsight profiling results
├── vit-pytorch/                    # vit-pytorch library (submodule)
├── bench.sh                        # Benchmark script
└── README.md
```

## Requirements

- Python 3.11+
- PyTorch 2.0+ with CUDA support
- CUDA Toolkit 12.x
- NVIDIA GPU with compute capability 7.0+

## Setup

TBD.

## Usage

TBD.

### Run Benchmark
```bash
bash bench.sh
```
This script will:
1. Compile the custom CUDA kernel
2. Run baseline inference with vit-pytorch (using PyTorch softmax)
3. Run optimized inference with custom CUDA.
4. Generate Nsight Systems profiling reports in `results/`

### Run Individual Scripts

**Baseline inference:**
```bash
python base/inference_vit_pytorch.py
```

**Custom CUDA kernel inference:**

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## License

This project is for educational purposes (NTU Parallel Programming Course).
