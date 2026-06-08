#pragma once

#include "tensor.cuh"

#include <cuda_runtime.h>

#ifdef QUADTRIX_ENABLE_NCCL
#include <nccl.h>
#else
typedef struct {
    char internal[128];
} ncclUniqueId;
typedef struct ncclComm* ncclComm_t;
typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclNumResults = 6
} ncclResult_t;
#endif

namespace quadtrix {
namespace cuda {

struct NcclStatus {
    bool ok;
    ncclResult_t nccl_status;
    const char* message;

    static NcclStatus success() {
        return {true, ncclSuccess, "ok"};
    }

    static NcclStatus failure(ncclResult_t status, const char* message) {
        return {false, status, message};
    }
};

const char* nccl_status_name(ncclResult_t status);

class NcclCommunicator {
public:
    NcclCommunicator() = default;
    NcclCommunicator(ncclUniqueId unique_id, int world_size, int rank, int device_id);
    ~NcclCommunicator();

    NcclCommunicator(const NcclCommunicator&) = delete;
    NcclCommunicator& operator=(const NcclCommunicator&) = delete;

    NcclCommunicator(NcclCommunicator&& other) noexcept;
    NcclCommunicator& operator=(NcclCommunicator&& other) noexcept;

    ncclComm_t get() const {
        return comm_;
    }

    int world_size() const {
        return world_size_;
    }

    int rank() const {
        return rank_;
    }

    int device_id() const {
        return device_id_;
    }

    bool valid() const {
        return comm_ != nullptr;
    }

private:
    ncclComm_t comm_ = nullptr;
    int world_size_ = 1;
    int rank_ = 0;
    int device_id_ = 0;
};

NcclStatus create_unique_id(ncclUniqueId* unique_id);

NcclStatus all_reduce_sum(
    NcclCommunicator& communicator,
    TensorView tensor,
    cudaStream_t stream = nullptr);

NcclStatus all_reduce_average(
    NcclCommunicator& communicator,
    TensorView tensor,
    cudaStream_t stream = nullptr);

}  // namespace cuda
}  // namespace quadtrix
