#include "../includes/gelu.cuh"

#include "../includes/runtime.cuh"
#include "../includes/utils.cuh"

#include <cmath>

namespace quadtrix {
namespace cuda {
namespace {

constexpr float kSqrtHalf = 0.70710678118654752440f;
constexpr float kSqrtTwoOverPi = 0.79788456080286535588f;
constexpr float kGeluCoeff = 0.044715f;

bool valid_unary_tensor_pair(const TensorView& input, const TensorView& output) {
    if (input.data == nullptr || output.data == nullptr || input.device != DeviceKind::CUDA ||
        output.device != DeviceKind::CUDA || input.device_id != output.device_id || input.dtype != DType::F32 ||
        output.dtype != DType::F32 || input.shape.rank != output.shape.rank || !input.shape.is_contiguous() ||
        !output.shape.is_contiguous() || input.numel() != output.numel()) {
        return false;
    }
    for (int i = 0; i < input.shape.rank; ++i) {
        if (input.shape.dims[i] != output.shape.dims[i]) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ float gelu_value(float x, GeluMode mode) {
    if (mode == GeluMode::Exact) {
        return 0.5f * x * (1.0f + erff(x * kSqrtHalf));
    }

    const float inner = kSqrtTwoOverPi * (x + kGeluCoeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gelu_forward_kernel(const float* __restrict__ input, float* __restrict__ output, std::size_t n, GeluMode mode) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gelu_value(input[idx], mode);
    }
}

}  // namespace

Status gelu_forward(const TensorView& input, TensorView output, GeluMode mode, cudaStream_t stream) {
    if (!valid_unary_tensor_pair(input, output)) {
        return Status::failure(cudaErrorInvalidValue, "invalid gelu_forward arguments");
    }

    DeviceGuard guard(input.device_id);
    const std::size_t n = input.numel();
    gelu_forward_kernel<<<one_dim_grid(n), kDefaultBlockSize, 0, stream>>>(
        input.data_as<const float>(),
        output.data_as<float>(),
        n,
        mode);
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace quadtrix
