#pragma once

#include "common.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace quadtrix {
namespace cuda {

constexpr int kWarpSize = 32;
constexpr int kDefaultBlockSize = 256;

inline int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

inline std::size_t ceil_div(std::size_t value, std::size_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline dim3 one_dim_grid(std::size_t n, int block_size = kDefaultBlockSize) {
    return dim3(static_cast<unsigned int>(ceil_div(n, static_cast<std::size_t>(block_size))));
}

#ifdef __CUDACC__
template <typename T>
__device__ __forceinline__ T warp_sum(T value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ T warp_max(T value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        T other = __shfl_down_sync(0xffffffffu, value, offset);
        value = value > other ? value : other;
    }
    return value;
}
#endif

}  // namespace cuda
}  // namespace quadtrix
