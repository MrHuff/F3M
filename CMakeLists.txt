cmake_minimum_required(VERSION 3.16)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.1/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
set(CUB_LIBRARY_ROOT /home/rhu/C_pp_libs/cub)
if (NOT EXISTS ${CUB_LIBRARY_ROOT})
    message(WARNING "[${CTX}] CANNOT FIND CUB LIBRARY - MAKE ENTRY IN ROOT CMakeLists.txt")
else()
    message("[${CTX}] CUB_LIBRARY_ROOT ${CUB_LIBRARY_ROOT}")
endif()
include_directories(
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CUB_LIBRARY_ROOT}
        # ...
)
# ...
add_library(cub INTERFACE IMPORTED)
target_include_directories(cub INTERFACE
        ${CUB_LIBRARY_ROOT}
        # ...
        )
set_target_properties(cub PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${CUB_LIBRARY_ROOT}
        # ...
        )
project(cuda_n_body CUDA)
enable_language(C)
enable_language(CXX)
find_package(Torch REQUIRED)

add_executable(cuda_n_body main.cu)
set_target_properties(
        cuda_n_body
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(cuda_n_body cublas) #remember to add .lib for windows etc
target_link_libraries(cuda_n_body "${TORCH_LIBRARIES}")
target_link_libraries(cuda_n_body cub

        )
set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
target_compile_options(cuda_n_body PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Xptxas -O3
        --use_fast_math
        -I/home/rhu/C_pp_libs/cub-1.8.0/cub
        >)
