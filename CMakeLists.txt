cmake_minimum_required(VERSION 3.10)
project(QuatNet LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86) # Add this line

add_executable(quatnet
    src/main.cpp
    src/quatnet_layer.cpp
    src/hamprod_kernel.cu
)

target_link_libraries(quatnet ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
set_target_properties(quatnet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)