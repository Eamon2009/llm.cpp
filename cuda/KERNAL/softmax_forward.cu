#include "../includes/softmax.cuh"

#include "../includes/runtime.cuh"
#include "../includes/utils.cuh"

#include <cmath>
#include <limits>

namespace quadtrix {
namespace cuda {
namespace {

constexpr int kSoftmaxBlockSize = 256;

bool fits_int(std::int64_t value) {
    return value > 0 && value <= std::numeric_limits<int>::max();
}

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

__global__ void softmax_forward_kernel(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int rows,
    int cols,
    int valid_cols) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* __restrict__ logits_row = logits + row * cols;
    float* __restrict__ probs_row = probs + row * cols;
    float local_max = -INFINITY;
    for (int col = threadIdx.x; col < valid_cols; col += blockDim.x) {
        local_max = fmaxf(local_max, logits_row[col]);
    }
    const float max_val = block_max(local_max, shared);

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < valid_cols; col += blockDim.x) {
        const float value = expf(logits_row[col] - max_val);
        probs_row[col] = value;
        local_sum += value;
    }
    const float sum = block_sum(local_sum, shared);
    const float inv_sum = sum == 0.0f ? 0.0f : 1.0f / sum;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        probs_row[col] = col < valid_cols ? probs_row[col] * inv_sum : 0.0f;
    }
}

__global__ void causal_softmax_row_kernel(
    const float* __restrict__ preatt,
    float* __restrict__ att,
    int rows,
    int time) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int t = row % time;
    const int valid_cols = t + 1;
    const float* __restrict__ preatt_row = preatt + row * time;
    float* __restrict__ att_row = att + row * time;

    float local_max = -INFINITY;
    for (int col = threadIdx.x; col < valid_cols; col += blockDim.x) {
        local_max = fmaxf(local_max, preatt_row[col]);
    }
    const float max_val = block_max(local_max, shared);

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < valid_cols; col += blockDim.x) {
        const float value = expf(preatt_row[col] - max_val);
        att_row[col] = value;
        local_sum += value;
    }
    const float sum = block_sum(local_sum, shared);
    const float inv_sum = sum == 0.0f ? 0.0f : 1.0f / sum;

    for (int col = threadIdx.x; col < time; col += blockDim.x) {
        att_row[col] = col < valid_cols ? att_row[col] * inv_sum : 0.0f;
    }
}

}  // namespace

Status softmax_forward(const TensorView& logits, TensorView probs, int valid_cols, cudaStream_t stream) {
    if (!valid_same_shape_f32(logits, probs) || logits.shape.rank != 2 || !fits_int(logits.shape.dims[0]) ||
        !fits_int(logits.shape.dims[1])) {
        return Status::failure(cudaErrorInvalidValue, "invalid softmax_forward tensors");
    }
    const int rows = static_cast<int>(logits.shape.dims[0]);
    const int cols = static_cast<int>(logits.shape.dims[1]);
    if (valid_cols <= 0 || valid_cols > cols) {
        return Status::failure(cudaErrorInvalidValue, "invalid softmax_forward valid_cols");
    }

    DeviceGuard guard(logits.device_id);
    const std::size_t shared_bytes = ((kSoftmaxBlockSize + kWarpSize - 1) / kWarpSize) * sizeof(float);
    softmax_forward_kernel<<<rows, kSoftmaxBlockSize, shared_bytes, stream>>>(
        logits.data_as<const float>(),
        probs.data_as<float>(),
        rows,
        cols,
        valid_cols);
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

Status causal_softmax_forward(const TensorView& preatt, TensorView att, cudaStream_t stream) {
    if (!valid_same_shape_f32(preatt, att) || preatt.shape.rank != 4 || !fits_int(preatt.shape.dims[0]) ||
        !fits_int(preatt.shape.dims[1]) || !fits_int(preatt.shape.dims[2]) ||
        preatt.shape.dims[2] != preatt.shape.dims[3]) {
        return Status::failure(cudaErrorInvalidValue, "invalid causal_softmax_forward tensors");
    }
    const int rows = static_cast<int>(preatt.shape.dims[0] * preatt.shape.dims[1] * preatt.shape.dims[2]);
    const int time = static_cast<int>(preatt.shape.dims[2]);

    DeviceGuard guard(preatt.device_id);
    const std::size_t shared_bytes = ((kSoftmaxBlockSize + kWarpSize - 1) / kWarpSize) * sizeof(float);
    causal_softmax_row_kernel<<<rows, kSoftmaxBlockSize, shared_bytes, stream>>>(
        preatt.data_as<const float>(),
        att.data_as<float>(),
        rows,
        time);
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace quadtrix
