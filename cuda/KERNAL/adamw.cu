#include "../includes/adamw.cuh"

#include "../includes/runtime.cuh"
#include "../includes/utils.cuh"

#include <cmath>

namespace quadtrix {
namespace cuda {
namespace {

bool valid_same_shape_f32(const TensorView& a, const TensorView& b) {
    if (a.data == nullptr || b.data == nullptr || a.device != DeviceKind::CUDA || b.device != DeviceKind::CUDA ||
        a.device_id != b.device_id || a.dtype != DType::F32 || b.dtype != DType::F32 ||
        a.shape.rank != b.shape.rank || !a.shape.is_contiguous() || !b.shape.is_contiguous() ||
        a.numel() != b.numel()) {
        return false;
    }
    for (int i = 0; i < a.shape.rank; ++i) {
        if (a.shape.dims[i] != b.shape.dims[i]) {
            return false;
        }
    }
    return true;
}

__global__ void adamw_kernel(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ first_moment,
    float* __restrict__ second_moment,
    std::size_t n,
    float learning_rate,
    float beta1,
    float beta2,
    float beta1_correction,
    float beta2_correction,
    float epsilon,
    float weight_decay,
    float grad_scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const float param = params[idx];
    const float grad = grads[idx] * grad_scale;
    const float m = beta1 * first_moment[idx] + (1.0f - beta1) * grad;
    const float v = beta2 * second_moment[idx] + (1.0f - beta2) * grad * grad;
    const float m_hat = m / beta1_correction;
    const float v_hat = v / beta2_correction;
    first_moment[idx] = m;
    second_moment[idx] = v;
    params[idx] = param - learning_rate * (m_hat / (sqrtf(v_hat) + epsilon) + weight_decay * param);
}

}  // namespace

Status adamw_update(
    TensorView params,
    const TensorView& grads,
    TensorView first_moment,
    TensorView second_moment,
    const AdamWConfig& config,
    float grad_scale,
    cudaStream_t stream) {
    if (!valid_same_shape_f32(params, grads) || !valid_same_shape_f32(params, first_moment) ||
        !valid_same_shape_f32(params, second_moment) || config.step <= 0 || config.beta1 < 0.0f ||
        config.beta1 >= 1.0f || config.beta2 < 0.0f || config.beta2 >= 1.0f || config.epsilon <= 0.0f) {
        return Status::failure(cudaErrorInvalidValue, "invalid adamw_update arguments");
    }

    const float beta1_correction = 1.0f - powf(config.beta1, static_cast<float>(config.step));
    const float beta2_correction = 1.0f - powf(config.beta2, static_cast<float>(config.step));
    DeviceGuard guard(params.device_id);
    adamw_kernel<<<one_dim_grid(params.numel()), kDefaultBlockSize, 0, stream>>>(
        params.data_as<float>(),
        grads.data_as<const float>(),
        first_moment.data_as<float>(),
        second_moment.data_as<float>(),
        params.numel(),
        config.learning_rate,
        config.beta1,
        config.beta2,
        beta1_correction,
        beta2_correction,
        config.epsilon,
        config.weight_decay,
        grad_scale);
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace quadtrix
