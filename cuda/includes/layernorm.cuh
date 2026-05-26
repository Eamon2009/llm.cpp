#pragma once

#include "tensor.cuh"

#include <cuda_runtime.h>

namespace quadtrix {
namespace cuda {

Status layernorm_forward(
    const TensorView& input,
    const TensorView& gamma,
    const TensorView& beta,
    TensorView output,
    TensorView mean,
    TensorView rstd,
    float epsilon = 1.0e-5f,
    cudaStream_t stream = nullptr);

Status layernorm_backward(
    const TensorView& grad_output,
    const TensorView& input,
    const TensorView& gamma,
    const TensorView& mean,
    const TensorView& rstd,
    TensorView grad_input,
    TensorView grad_gamma,
    TensorView grad_beta,
    cudaStream_t stream = nullptr);

}  // namespace cuda
}  // namespace quadtrix
