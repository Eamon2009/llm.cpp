#pragma once

#include "tensor.cuh"

#include <cuda_runtime.h>

namespace quadtrix {
namespace cuda {

Status attention_forward(
    const TensorView& input_qkv,
    TensorView preatt,
    TensorView att,
    TensorView output,
    int num_heads,
    cudaStream_t stream = nullptr);

Status attention_backward(
    const TensorView& grad_output,
    const TensorView& input_qkv,
    const TensorView& att,
    TensorView grad_input_qkv,
    TensorView grad_preatt,
    TensorView grad_att,
    int num_heads,
    cudaStream_t stream = nullptr);

}  // namespace cuda
}  // namespace quadtrix
