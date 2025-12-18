from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Get CUDA compute capability
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
else:
    arch = "90"  # Default to sm_90 (Hopper)

setup(
    name="sa_v4_ext",
    ext_modules=[
        CUDAExtension(
            name="sa_v4_ext",
            sources=["sa_v4_ext.cpp", "sa_v4_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-std=c++17",
                    f"-arch=sm_{arch}",
                    "--ptxas-options=-v",  # Show register usage
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
