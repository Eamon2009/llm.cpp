#pragma once

#include "tensor.cuh"

#include <cuda_runtime.h>

namespace quadtrix {
namespace cuda {

Status global_norm_squared(
    const TensorView& grads,
    TensorView partial_sums,
    cudaStream_t stream = nullptr);

Status clip_gradients_by_global_norm(
    TensorView grads,
    float global_norm,
    float max_norm,
    cudaStream_t stream = nullptr);

inline float clip_scale(float global_norm, float max_norm) {
    return global_norm > max_norm && global_norm > 0.0f ? max_norm / global_norm : 1.0f;
}

}  // namespace cuda
}  // namespace quadtrix
