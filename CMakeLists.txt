cmake_minimum_required(VERSION 3.16)
project(cuda_n_body CUDA)

set(CMAKE_CUDA_STANDARD 14)
add_executable(cuda_n_body main.cu)
set_target_properties(
        cuda_n_body
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(cuda_n_body cublas) #remember to add .lib for windows etc
