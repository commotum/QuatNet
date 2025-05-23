cmake_minimum_required(VERSION 3.18)
project(QuatNetIsokawa LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (A6000 is sm_86)
set(CMAKE_CUDA_ARCHITECTURES 86) # Adjust if targeting other GPUs

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

# Add sources for the library
set(QUATNET_ISOKAWA_SRCS
    src/quaternion_math.cu # If it has global kernels or host functions needing CUDA compilation
    src/isokawa_layer.cu
)

# Create a shared library for Python bindings
add_library(isokawa_cuda SHARED ${QUATNET_ISOKAWA_SRCS})

# Link CUDA libraries
target_link_libraries(isokawa_cuda PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY}) # Or ${CUDA_CUDA_LIBRARY}

# Enable separable compilation if needed (usually good for larger projects)
set_target_properties(isokawa_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Optional: An executable for testing C++ components
# add_executable(isokawa_test src/main.cpp ${QUATNET_ISOKAWA_SRCS})
# target_link_libraries(isokawa_test PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
# set_target_properties(isokawa_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Installation (optional, for system-wide install or packaging)
# install(TARGETS isokawa_cuda LIBRARY DESTINATION lib)
# install(FILES src/quaternion_math.h src/isokawa_layer.h DESTINATION include/QuatNetIsokawa)
