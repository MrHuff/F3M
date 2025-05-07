from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Set CUDA architecture list (optional but recommended)
os.environ['TORCH_CUDA_ARCH_LIST'] = "6.1;7.5;8.0"  # Example architectures

setup(
    name='ffm_3d_float',
    packages=find_packages(include=['F3M_src', 'F3M_src.*']),  # Adjust as needed
    ext_modules=[
        CUDAExtension(
            'ffm_3d_float',
            sources=['pybinder_setup.cu'],
            extra_compile_args={
                'nvcc': ['--use_fast_math'],
                'cxx': ['-O3'],  # Optimize C++ compilation (optional)
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
