from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sa_cuda',
    ext_modules=[
        CUDAExtension(
            name='sa_cuda_ext',
            sources=['sa_ext.cpp', 'sa_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
