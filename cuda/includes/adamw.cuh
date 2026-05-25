#pragma once

#include "tensor.cuh"

#include <cuda_runtime.h>

namespace quadtrix {
namespace cuda {

struct AdamWConfig {
    float learning_rate = 1.0e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float epsilon = 1.0e-8f;
    float weight_decay = 0.0f;
    int step = 1;
};

Status adamw_update(
    TensorView params,
    const TensorView& grads,
    TensorView first_moment,
    TensorView second_moment,
    const AdamWConfig& config,
    float grad_scale = 1.0f,
    cudaStream_t stream = nullptr);

}  // namespace cuda
}  // namespace quadtrix
