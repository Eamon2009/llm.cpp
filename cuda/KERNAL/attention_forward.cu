#include "../includes/attention.cuh"

#include "../includes/runtime.cuh"
#include "../includes/utils.cuh"

#include <cmath>
#include <limits>

namespace quadtrix {
namespace cuda {
namespace {

constexpr int kAttentionBlockSize = 256;

bool fits_int(std::int64_t value) {
    return value > 0 && value <= std::numeric_limits<int>::max();
}

bool valid_f32_cuda(const TensorView& tensor) {
    return tensor.data != nullptr && tensor.device == DeviceKind::CUDA && tensor.dtype == DType::F32 &&
           tensor.shape.is_contiguous();
}

__device__ float block_sum(float value, float* shared) {
    value = warp_sum(value);
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    const int warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;
    value = threadIdx.x < warp_count ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warp_sum(value);
    }
    if (threadIdx.x == 0) {
        shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

__device__ float block_max(float value, float* shared) {
    value = warp_max(value);
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    const int warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;
    value = threadIdx.x < warp_count ? shared[lane] : -INFINITY;
    if (warp == 0) {
        value = warp_max(value);
    }
    if (threadIdx.x == 0) {
        shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

__global__ void attention_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ preatt,
    float* __restrict__ att,
    float* __restrict__ output,
    int total_rows,
    int time,
    int channels,
    int num_heads,
    int head_size) {
    extern __shared__ float shared[];
    const int row_id = blockIdx.x;
    if (row_id >= total_rows) {
        return;
    }

    const int h = row_id % num_heads;
    const int t = (row_id / num_heads) % time;
    const int b = row_id / (num_heads * time);
    const float scale = rsqrtf(static_cast<float>(head_size));
    const int c3 = 3 * channels;
    const float* __restrict__ query = input + b * time * c3 + t * c3 + h * head_size;
    float* __restrict__ preatt_row = preatt + b * num_heads * time * time + h * time * time + t * time;
    float* __restrict__ att_row = att + b * num_heads * time * time + h * time * time + t * time;

    float local_max = -INFINITY;
    for (int t2 = threadIdx.x; t2 <= t; t2 += blockDim.x) {
        const float* __restrict__ key = input + b * time * c3 + t2 * c3 + channels + h * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; ++i) {
            score += query[i] * key[i];
        }
        score *= scale;
        preatt_row[t2] = score;
        local_max = fmaxf(local_max, score);
    }
    const float max_val = block_max(local_max, shared);

    float local_sum = 0.0f;
    for (int t2 = threadIdx.x; t2 <= t; t2 += blockDim.x) {
        const float value = expf(preatt_row[t2] - max_val);
        att_row[t2] = value;
        local_sum += value;
    }
    const float sum = block_sum(local_sum, shared);
    const float inv_sum = sum == 0.0f ? 0.0f : 1.0f / sum;

    for (int t2 = threadIdx.x; t2 < time; t2 += blockDim.x) {
        att_row[t2] = t2 <= t ? att_row[t2] * inv_sum : 0.0f;
        if (t2 > t) {
            preatt_row[t2] = 0.0f;
        }
    }
    __syncthreads();

    float* __restrict__ out = output + b * time * channels + t * channels + h * head_size;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float value = 0.0f;
        for (int t2 = 0; t2 <= t; ++t2) {
            const float* __restrict__ v = input + b * time * c3 + t2 * c3 + 2 * channels + h * head_size;
            value += att_row[t2] * v[i];
        }
        out[i] = value;
    }
}

}  // namespace

Status attention_forward(
    const TensorView& input_qkv,
    TensorView preatt,
    TensorView att,
    TensorView output,
    int num_heads,
    cudaStream_t stream) {
    if (!valid_f32_cuda(input_qkv) || !valid_f32_cuda(preatt) || !valid_f32_cuda(att) || !valid_f32_cuda(output) ||
        input_qkv.shape.rank != 3 || preatt.shape.rank != 4 || att.shape.rank != 4 || output.shape.rank != 3 ||
        input_qkv.device_id != preatt.device_id || input_qkv.device_id != att.device_id ||
        input_qkv.device_id != output.device_id) {
        return Status::failure(cudaErrorInvalidValue, "invalid attention_forward tensors");
    }

    const std::int64_t batch = input_qkv.shape.dims[0];
    const std::int64_t time = input_qkv.shape.dims[1];
    const std::int64_t channels3 = input_qkv.shape.dims[2];
    if (num_heads <= 0 || channels3 % 3 != 0) {
        return Status::failure(cudaErrorInvalidValue, "invalid attention_forward qkv shape");
    }
    const std::int64_t channels = channels3 / 3;
    if (channels % num_heads != 0 || output.shape.dims[0] != batch || output.shape.dims[1] != time ||
        output.shape.dims[2] != channels || preatt.shape.dims[0] != batch || preatt.shape.dims[1] != num_heads ||
        preatt.shape.dims[2] != time || preatt.shape.dims[3] != time || att.shape.dims != preatt.shape.dims ||
        !fits_int(batch) || !fits_int(time) || !fits_int(channels)) {
        return Status::failure(cudaErrorInvalidValue, "invalid attention_forward shape");
    }

    DeviceGuard guard(input_qkv.device_id);
    const int rows = static_cast<int>(batch * time * num_heads);
    const std::size_t shared_bytes = ((kAttentionBlockSize + kWarpSize - 1) / kWarpSize) * sizeof(float);
    attention_forward_kernel<<<rows, kAttentionBlockSize, shared_bytes, stream>>>(
        input_qkv.data_as<const float>(),
        preatt.data_as<float>(),
        att.data_as<float>(),
        output.data_as<float>(),
        rows,
        static_cast<int>(time),
        static_cast<int>(channels),
        num_heads,
        static_cast<int>(channels / num_heads));
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace quadtrix
