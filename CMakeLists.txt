cmake_minimum_required(VERSION 3.18)
project(MarchingCubes LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA architectures - adjust based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find CUDA explicitly
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA found: ${CUDAToolkit_FOUND}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARY_DIR}")

# All source files in root
add_executable(marching_cubes
    src/main.cpp
    src/marching_cubes_cpu.cpp
    src/marching_cubes_gpu.cu
    src/utils.cpp

    src/common.h
    src/marching_cubes_cpu.h
    src/marching_cubes_gpu.cuh
    src/utils.h
)

# CUDA-specific compile options
target_compile_options(marching_cubes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        --expt-relaxed-constexpr
        -Xcompiler=/Zc:preprocessor
    >
)

# Link CUDA runtime explicitly
target_link_libraries(marching_cubes PRIVATE  ${CUDA_LIBRARIES})
target_include_directories(marching_cubes PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
)