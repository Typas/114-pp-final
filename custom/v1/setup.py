from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sa_v1_ext",
    ext_modules=[
        CUDAExtension(
            name="sa_v1_ext",
            sources=["sa_v1_ext.cpp", "sa_v1_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
