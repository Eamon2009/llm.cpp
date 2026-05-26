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

bool valid_backward_tensors(const TensorView& grad_output, const TensorView& input, const TensorView& grad_input) {
    if (grad_output.data == nullptr || input.data == nullptr || grad_input.data == nullptr ||
        grad_output.device != DeviceKind::CUDA || input.device != DeviceKind::CUDA ||
        grad_input.device != DeviceKind::CUDA || grad_output.device_id != input.device_id ||
        grad_output.device_id != grad_input.device_id || grad_output.dtype != DType::F32 ||
        input.dtype != DType::F32 || grad_input.dtype != DType::F32 || grad_output.shape.rank != input.shape.rank ||
        grad_output.shape.rank != grad_input.shape.rank || !grad_output.shape.is_contiguous() ||
        !input.shape.is_contiguous() || !grad_input.shape.is_contiguous() || grad_output.numel() != input.numel() ||
        grad_output.numel() != grad_input.numel()) {
        return false;
    }
    for (int i = 0; i < input.shape.rank; ++i) {
        if (grad_output.shape.dims[i] != input.shape.dims[i] || grad_output.shape.dims[i] != grad_input.shape.dims[i]) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ float gelu_grad(float x, GeluMode mode) {
    if (mode == GeluMode::Exact) {
        const float cdf = 0.5f * (1.0f + erff(x * kSqrtHalf));
        const float pdf = 0.39894228040143267794f * expf(-0.5f * x * x);
        return cdf + x * pdf;
    }

    const float x2 = x * x;
    const float x3 = x2 * x;
    const float inner = kSqrtTwoOverPi * (x + kGeluCoeff * x3);
    const float t = tanhf(inner);
    const float sech2 = 1.0f - t * t;
    const float inner_grad = kSqrtTwoOverPi * (1.0f + 3.0f * kGeluCoeff * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * sech2 * inner_grad;
}

__global__ void gelu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    std::size_t n,
    GeluMode mode) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] += grad_output[idx] * gelu_grad(input[idx], mode);
    }
}

}  // namespace

Status gelu_backward(
    const TensorView& grad_output,
    const TensorView& input,
    TensorView grad_input,
    GeluMode mode,
    cudaStream_t stream) {
    if (!valid_backward_tensors(grad_output, input, grad_input)) {
        return Status::failure(cudaErrorInvalidValue, "invalid gelu_backward arguments");
    }

    DeviceGuard guard(input.device_id);
    const std::size_t n = input.numel();
    gelu_backward_kernel<<<one_dim_grid(n), kDefaultBlockSize, 0, stream>>>(
        grad_output.data_as<const float>(),
        input.data_as<const float>(),
        grad_input.data_as<float>(),
        n,
        mode);
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace quadtrix
