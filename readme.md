**To install:**

* CUDA devtoolkit 10.1
* GCC 7+
* CMAKE
* LibTorch, download @ pytorch.org. Make sure to get the (cxx11 ABI) version.
* LibTorch needs cuBLAS and cuDNN. These can be downloaded and installed from Nvidia webpage
* cub; clone this from https://github.com/NVIDIA/cub.git (although this will likely be removed)

In cmake options, make sure to add -DCMAKE_PREFIX_PATH=/path_to_libtorch_download/libtorch-cxx11-abi-shared-with-deps-1.4.0/libtorch

There might be some modifications needed for each specific system. This is developed on Ubuntu 18.04.

Compile by using:

/usr/local/cuda-10.1/bin/nvcc  -DAT_PARALLEL_OPENMP=1 -isystem=/home/rhu/C_pp_libs/cub -isystem=/home/rhu/C_pp_libs/libtorch-cxx11-abi-shared-with-deps-1.5.0+cu101/libtorch/include -isystem=/home/rhu/C_pp_libs/libtorch-cxx11-abi-shared-with-deps-1.5.0+cu101/libtorch/include/torch/csrc/api/include  -g -Xcompiler=-fPIE   -Xptxas -O3 --use_fast_math -I/home/rhu/C_pp_libs/cub-1.8.0/cub -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14 -x cu -dc /home/rhu/Documents/phd_projects/cuda_n_body/main.cu -o CMakeFiles/cuda_n_body.dir/main.cu.o

replace the paths with your own system paths.