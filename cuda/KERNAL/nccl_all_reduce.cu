#include "../includes/nccl_all_reduce.cuh"

#include "../includes/runtime.cuh"
#include "../includes/utils.cuh"

#include <cstdio>
#include <utility>

namespace quadtrix {
namespace cuda {
namespace {

#ifdef QUADTRIX_ENABLE_NCCL
ncclDataType_t to_nccl_dtype(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return ncclFloat32;
        case DType::F16:
            return ncclFloat16;
        case DType::I32:
            return ncclInt32;
        case DType::U8:
            return ncclUint8;
    }
    return ncclFloat32;
}

bool supports_nccl_dtype(DType dtype) {
    return dtype == DType::F32 || dtype == DType::F16 || dtype == DType::I32 || dtype == DType::U8
        ;
}
#else
bool supports_nccl_dtype(DType) {
    return false;
}
#endif

bool valid_reduce_tensor(const TensorView& tensor) {
    return tensor.data != nullptr && tensor.device == DeviceKind::CUDA && tensor.shape.is_contiguous() &&
           tensor.numel() > 0 && supports_nccl_dtype(tensor.dtype);
}

__global__ void scale_kernel(float* values, std::size_t n, float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        values[idx] *= scale;
    }
}

}  // namespace

const char* nccl_status_name(ncclResult_t status) {
#ifdef QUADTRIX_ENABLE_NCCL
    return ncclGetErrorString(status);
#else
    switch (status) {
        case ncclSuccess:
            return "ncclSuccess";
        case ncclUnhandledCudaError:
            return "ncclUnhandledCudaError";
        case ncclSystemError:
            return "ncclSystemError";
        case ncclInternalError:
            return "ncclInternalError";
        case ncclInvalidArgument:
            return "ncclInvalidArgument";
        case ncclInvalidUsage:
            return "ncclInvalidUsage";
        case ncclNumResults:
            return "ncclNumResults";
    }
    return "ncclUnknown";
#endif
}

NcclCommunicator::NcclCommunicator(ncclUniqueId unique_id, int world_size, int rank, int device_id)
    : world_size_(world_size), rank_(rank), device_id_(device_id) {
    DeviceGuard guard(device_id_);
#ifdef QUADTRIX_ENABLE_NCCL
    ncclResult_t status = ncclCommInitRank(&comm_, world_size_, unique_id, rank_);
    if (status != ncclSuccess) {
        std::fprintf(stderr, "Fatal NCCL error: ncclCommInitRank failed with %s\n", nccl_status_name(status));
        std::abort();
    }
#else
    std::fprintf(stderr, "Fatal NCCL error: build with QUADTRIX_ENABLE_NCCL and link NCCL to use NcclCommunicator\n");
    std::abort();
#endif
}

NcclStatus create_unique_id(ncclUniqueId* unique_id) {
    if (unique_id == nullptr) {
        return NcclStatus::failure(ncclInvalidArgument, "unique_id must not be null");
    }
#ifdef QUADTRIX_ENABLE_NCCL
    ncclResult_t status = ncclGetUniqueId(unique_id);
    if (status != ncclSuccess) {
        return NcclStatus::failure(status, "ncclGetUniqueId failed");
    }
    return NcclStatus::success();
#else
    return NcclStatus::failure(ncclInvalidUsage, "NCCL support is not enabled in this build");
#endif
}

NcclCommunicator::~NcclCommunicator() {
#ifdef QUADTRIX_ENABLE_NCCL
    if (comm_ != nullptr) {
        DeviceGuard guard(device_id_);
        ncclCommDestroy(comm_);
    }
#endif
}

NcclCommunicator::NcclCommunicator(NcclCommunicator&& other) noexcept
    : comm_(other.comm_), world_size_(other.world_size_), rank_(other.rank_), device_id_(other.device_id_) {
    other.comm_ = nullptr;
}

NcclCommunicator& NcclCommunicator::operator=(NcclCommunicator&& other) noexcept {
    if (this == &other) {
        return *this;
    }
#ifdef QUADTRIX_ENABLE_NCCL
    if (comm_ != nullptr) {
        DeviceGuard guard(device_id_);
        ncclCommDestroy(comm_);
    }
#endif
    comm_ = other.comm_;
    world_size_ = other.world_size_;
    rank_ = other.rank_;
    device_id_ = other.device_id_;
    other.comm_ = nullptr;
    return *this;
}

NcclStatus all_reduce_sum(NcclCommunicator& communicator, TensorView tensor, cudaStream_t stream) {
    if (!communicator.valid() || tensor.device_id != communicator.device_id() || !valid_reduce_tensor(tensor)) {
        return NcclStatus::failure(ncclInvalidArgument, "invalid all_reduce_sum arguments");
    }

#ifdef QUADTRIX_ENABLE_NCCL
    DeviceGuard guard(communicator.device_id());
    ncclResult_t status = ncclAllReduce(
        tensor.data,
        tensor.data,
        tensor.numel(),
        to_nccl_dtype(tensor.dtype),
        ncclSum,
        communicator.get(),
        stream);
    if (status != ncclSuccess) {
        return NcclStatus::failure(status, "ncclAllReduce failed");
    }
    return NcclStatus::success();
#else
    return NcclStatus::failure(ncclInvalidUsage, "NCCL support is not enabled in this build");
#endif
}

NcclStatus all_reduce_average(NcclCommunicator& communicator, TensorView tensor, cudaStream_t stream) {
    NcclStatus reduce = all_reduce_sum(communicator, tensor, stream);
    if (!reduce.ok) {
        return reduce;
    }
    if (tensor.dtype != DType::F32) {
        return NcclStatus::failure(ncclInvalidArgument, "all_reduce_average currently supports F32 tensors only");
    }

    DeviceGuard guard(communicator.device_id());
    scale_kernel<<<one_dim_grid(tensor.numel()), kDefaultBlockSize, 0, stream>>>(
        tensor.data_as<float>(),
        tensor.numel(),
        1.0f / static_cast<float>(communicator.world_size()));
    Status scale_status = QUADTRIX_CUDA_CHECK(cudaGetLastError());
    if (!scale_status.ok) {
        return NcclStatus::failure(ncclUnhandledCudaError, "all_reduce_average scale kernel failed");
    }
    return NcclStatus::success();
}

}  // namespace cuda
}  // namespace quadtrix
