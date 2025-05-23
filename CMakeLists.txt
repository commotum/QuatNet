cmake_minimum_required(VERSION 3.18)
project(QuatNet LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Set target GPU architecture; adjust as needed
set(CMAKE_CUDA_ARCHITECTURES 86)

set(QUATNET_SOURCES
    src/hamprod_kernel.cu
    src/quat_ops.cu
    src/quatnet_layer.cu
)

add_library(quatnet_cuda STATIC ${QUATNET_SOURCES})
target_include_directories(quatnet_cuda PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(quatnet_cuda PUBLIC CUDA::cudart)
set_target_properties(quatnet_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(quatnet_example src/main.cpp)
target_link_libraries(quatnet_example PRIVATE quatnet_cuda CUDA::cudart)

add_executable(hamprod_tests tests/hamprod_tests.cpp)
target_link_libraries(hamprod_tests PRIVATE quatnet_cuda CUDA::cudart)
add_test(NAME hamprod_tests COMMAND hamprod_tests)
