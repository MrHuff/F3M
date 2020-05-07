**To install:**

* CUDA devtoolkit 10.1
* GCC 7+
* CMAKE
* LibTorch, download @ pytorch.org
* LibTorch needs cuBLAS and cuDNN.

In my cmake options, make sure to add -DCMAKE_PREFIX_PATH=/path_to_libtorch_download/libtorch-cxx11-abi-shared-with-deps-1.4.0/libtorch

There might be some modifications needed for each specific system. This is developed on Ubuntu 18.04.
