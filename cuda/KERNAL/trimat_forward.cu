#include "../includes/trimat.cuh"

#include "../includes/runtime.cuh"
#include "../includes/utils.cuh"

namespace quadtrix {
namespace cuda {
namespace {

bool valid_square_f32_cuda(const TensorView& matrix) {
    return matrix.data != nullptr && matrix.device == DeviceKind::CUDA && matrix.dtype == DType::F32 &&
           matrix.shape.rank == 4 && matrix.shape.is_contiguous() && matrix.shape.dims[2] == matrix.shape.dims[3];
}

__global__ void causal_mask_kernel(float* matrix, std::size_t n, int time, float masked_value) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int col = static_cast<int>(idx % time);
    const int row = static_cast<int>((idx / time) % time);
    if (col > row) {
        matrix[idx] = masked_value;
    }
}

}  // namespace

Status causal_mask_forward(TensorView matrix, float masked_value, cudaStream_t stream) {
    if (!valid_square_f32_cuda(matrix)) {
        return Status::failure(cudaErrorInvalidValue, "invalid causal_mask_forward tensor");
    }

    DeviceGuard guard(matrix.device_id);
    causal_mask_kernel<<<one_dim_grid(matrix.numel()), kDefaultBlockSize, 0, stream>>>(
        matrix.data_as<float>(),
        matrix.numel(),
        static_cast<int>(matrix.shape.dims[2]),
        masked_value);
    return QUADTRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace quadtrix
