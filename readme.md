**To install:**

* CUDA devtoolkit 10.1
* GCC 7+
* CMAKE
* LibTorch, download @ pytorch.org. Make sure to get the (cxx11 ABI) version.
* LibTorch needs cuBLAS and cuDNN. These can be downloaded and installed from Nvidia webpage

In cmake options, make sure to add -DCMAKE_PREFIX_PATH=/path_to_libtorch_download/libtorch-cxx11-abi-shared-with-deps-1.4.0/libtorch

There might be some modifications needed for each specific system. This is developed on Ubuntu 18.04.

Compile by using:

1. Making an application directory, its recommended to use /cmake-build-debug
2. Build cmake: cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="/homes/rhu/C_pp_libs/libtorch" -G "CodeBlocks - Unix Makefiles" /homes/rhu/cuda_n_body
3. Build the actual application: cmake --build /homes/rhu/cuda_n_body/cmake-build-debug --target cuda_n_body -- -j8

Release mode:

2. cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/homes/rhu/C_pp_libs/libtorch" -G "CodeBlocks - Unix Makefiles" /homes/rhu/cuda_n_body
3. cmake --build /homes/rhu/cuda_n_body/cmake-build-release --target cuda_n_body -- -j8

replace the paths with your own system paths.