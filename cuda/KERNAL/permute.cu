#include "../includes/permute.cuh"

#include "../includes/runtime.cuh"
#include "../includes/utils.cuh"

#include <limits>

namespace quadtrix {
namespace cuda {
namespace {

bool fits_int(std::int64_t value) {
    return value > 0 && value <= std::numeric_limits<int>::max();
}

bool valid_f32_cuda(const TensorView& tensor) {
    return tensor.data != nullptr && tensor.device == DeviceKind::CUDA && tensor.dtype == DType::F32 &&
           tensor.shape.is_contiguous();
}

__global__ void permute_qkv_kernel(
    const float* __restrict__ input_qkv,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    int total,
    int time,
    int channels,
    int head_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int i = idx % head_size;
    const int h = (idx / head_size) % (channels / head_size);
    const int t = (idx / head_size / (channels / head_size)) % time;
    const int b = idx / head_size / (channels / head_size) / time;
    const int c = h * head_size + i;
    const int src_base = b * time * 3 * channels + t * 3 * channels;
    query[idx] = input_qkv[src_base + c];
    key[idx] = input_qkv[src_base + channels + c];
    value[idx] = input_qkv[src_base + 2 * channels + c];
}

__global__ void unpermute_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total,
    int time,
    int channels,
    int head_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int i = idx % head_size;
    const int h = (idx / head_size) % (channels / head_size);
    const int t = (idx / head_size / (channels / head_size)) % time;
    const int b = idx / head_size / (channels / head_size) / time;
    const int c = h * head_size + i;
    output[b * time * channels + t * channels + c] = input[idx];
}

}  // namespace

Status permute_qkv_btc_to_bnhth(
    const TensorView& input_qkv,
    TensorView query,
    TensorView key,
    TensorView value,
    int num_heads,
    cudaStream_t stream) {
    if (!valid_f32_cuda(input_qkv) || !valid_f32_cuda(query) || !valid_f32_cuda(key) || !valid_f32_cuda(value) ||
        input_qkv.shape.rank != 3 || query.shape.rank != 4 || key.shape.rank != 4 || value.shape.rank != 4 ||
        query.device_id != input_qkv.device_id || key.device_id != input_qkv.device_id ||
        value.device_id != input_qkv.device_id) {
        return Status::failure(cudaErrorInvalidValue, "invalid permute_qkv tensors");
    }
    const std::int64_t batch = input_qkv.shape.dims[0];
    const std::int64_t time = input_qkv.shape.dims[1];
    const std::int64_t channels3 = input_qkv.shape.dims[2];
    if (num_heads <= 0 || channels3 % 3 != 0) {
        return Status::failure(cudaErrorInvalidValue, "invalid permute_qkv shape");
    }
    const std::int64_t channels = channels3 / 3;
    if (channels % num_heads != 0 || query.shape.dims[0] != batch || query.shape.dims[1] != num_heads ||
        query.shape.dims[2] != time || query.shape.dims[3] != channels / num_heads ||
        key.shape.dims != query.shape.dims || value.shape.dims != query.shape.dims ||
        !fits_int(query.numel())) {
        return Status::failure(cudaErrorInvalidValue, "invalid permute_qkv output shape");
    }

    DeviceGuard guard(input_qkv.device_id);
    const int total = static_cast<int>(query.numel());
    permute_qkv_kernel<<<one_dim_grid(total), kDefaultBlockSize, 0, stream>>>(
        input_qkv.data_as<const float>(),
        query.data_as<float>(),
        key.data_as<float>(),
        value.data_as<float>(),
        total,
        static_cast<int>(time),
        static_cast<int>(channels),
        static_cast<int>(channels / num_heads));
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

Status unpermute_bnhth_to_btc(const TensorView& input, TensorView output, cudaStream_t stream) {
    if (!valid_f32_cuda(input) || !valid_f32_cuda(output) || input.shape.rank != 4 || output.shape.rank != 3 ||
        input.device_id != output.device_id) {
        return Status::failure(cudaErrorInvalidValue, "invalid unpermute tensors");
    }
    const std::int64_t batch = input.shape.dims[0];
    const std::int64_t heads = input.shape.dims[1];
    const std::int64_t time = input.shape.dims[2];
    const std::int64_t head_size = input.shape.dims[3];
    const std::int64_t channels = heads * head_size;
    if (output.shape.dims[0] != batch || output.shape.dims[1] != time || output.shape.dims[2] != channels ||
        !fits_int(input.numel())) {
        return Status::failure(cudaErrorInvalidValue, "invalid unpermute output shape");
    }

    DeviceGuard guard(input.device_id);
    const int total = static_cast<int>(input.numel());
    unpermute_kernel<<<one_dim_grid(total), kDefaultBlockSize, 0, stream>>>(
        input.data_as<const float>(),
        output.data_as<float>(),
        total,
        static_cast<int>(time),
        static_cast<int>(channels),
        static_cast<int>(head_size));
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace quadtrix
