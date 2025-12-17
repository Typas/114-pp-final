from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sa_v2_ext",
    ext_modules=[
        CUDAExtension(
            name="sa_v2_ext",
            sources=["sa_v2_ext.cpp", "sa_v2_kernel.cu"],
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
