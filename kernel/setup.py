# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_cuda_kernels',
    ext_modules=[
        CUDAExtension(
            name='softmax_cuda',
            sources=['softmax_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    #'-gencode=arch=compute_120,code=sm_120',  # RTX 5090, Blackwell
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                    # '-gencode=arch=compute_86,code=sm_86',  # RTX 3090, Ampere
                    # '-gencode=arch=compute_75,code=sm_75',  # RTX 2080, Turing
                    # '-gencode=arch=compute_80,code=sm_80',  # A100
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
