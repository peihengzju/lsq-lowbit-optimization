import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def build_extensions():
    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    has_cuda = CUDA_HOME is not None

    if not has_cuda and not force_cuda:
        print("[setup.py] CUDA_HOME not found. Installing Python package without CUDA extension.")
        return []

    if not has_cuda and force_cuda:
        raise RuntimeError("FORCE_CUDA=1 is set but CUDA_HOME is not available")

    return [
        CUDAExtension(
            name="int4_int8_ext",
            sources=[
                "csrc/int4_int8_ext.cpp",
                "csrc/int4_int8_kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ]


README = Path(__file__).resolve().parent / "README.md"

setup(
    name="int4-cuda-inference-backend",
    version="0.1.0",
    description="Custom INT4 CUDA inference backend for quantized CNNs in PyTorch",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "benchmarks")),
    python_requires=">=3.10",
    ext_modules=build_extensions(),
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
