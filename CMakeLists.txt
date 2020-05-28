cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.1/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
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
set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")