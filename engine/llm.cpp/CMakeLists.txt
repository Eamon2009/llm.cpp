cmake_minimum_required(VERSION 3.18)

project(LLM_CUDA LANGUAGES CXX CUDA)

# Require C++17 for both Host C++ and NVCC CUDA compilers
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to Release build for max GPU/CPU optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
endif()

# Target your local GPU architecture automatically (e.g. Ampere, Ada, Hopper)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Find CUDA Toolkit (for cuBLAS & runtime) and OpenMP
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Main Executable Target
add_executable(llm llm.cu)

# Include project root so #include "include/..." and "config/..." work seamlessly
target_include_directories(llm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link cuBLAS, CUDA Runtime, and OpenMP
target_link_libraries(llm PRIVATE
    CUDA::cublas
    CUDA::cudart
    OpenMP::OpenMP_CXX
)

# Pass Windows & OpenMP flags correctly to NVCC host compiler
if(WIN32)
    target_compile_definitions(llm PRIVATE 
        NOMINMAX 
        WIN32_LEAN_AND_MEAN
    )
    target_compile_options(llm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp,/DNOMINMAX>
        $<$<COMPILE_LANGUAGE:CXX>:/openmp /DNOMINMAX>
    )
else()
    target_compile_options(llm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
        $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    )
endif()

# Enable high optimization level for CUDA kernels
target_compile_options(llm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)