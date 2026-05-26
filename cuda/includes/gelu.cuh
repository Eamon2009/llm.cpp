#pragma once

#include "tensor.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace quadtrix {
namespace cuda {

enum class GeluMode : std::uint8_t {
    Exact,
    Approximate,
};

Status gelu_forward(
    const TensorView& input,
    TensorView output,
    GeluMode mode = GeluMode::Approximate,
    cudaStream_t stream = nullptr);

Status gelu_backward(
    const TensorView& grad_output,
    const TensorView& input,
    TensorView grad_input,
    GeluMode mode = GeluMode::Approximate,
    cudaStream_t stream = nullptr);

}  // namespace cuda
}  // namespace quadtrix
