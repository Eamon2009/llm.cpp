#pragma once

#include "tensor.cuh"

#include <cuda_runtime.h>

namespace quadtrix {
namespace cuda {

Status permute_qkv_btc_to_bnhth(
    const TensorView& input_qkv,
    TensorView query,
    TensorView key,
    TensorView value,
    int num_heads,
    cudaStream_t stream = nullptr);

Status unpermute_bnhth_to_btc(
    const TensorView& input,
    TensorView output,
    cudaStream_t stream = nullptr);

}  // namespace cuda
}  // namespace quadtrix
