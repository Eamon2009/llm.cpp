#pragma once

#include "common.h"
#include "runtime.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace quadtrix {
namespace cuda {

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t bytes, int device_id = -1) {
        allocate(bytes, device_id);
    }

    ~DeviceBuffer() {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        swap(other);
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            swap(other);
        }
        return *this;
    }

    void allocate(std::size_t bytes, int device_id = -1) {
        release();
        if (bytes == 0) {
            return;
        }
        if (device_id >= 0) {
            device_id_ = device_id;
            DeviceGuard guard(device_id);
            QUADTRIX_CUDA_ABORT(cudaMalloc(&ptr_, bytes));
        } else {
            device_id_ = current_device();
            QUADTRIX_CUDA_ABORT(cudaMalloc(&ptr_, bytes));
        }
        bytes_ = bytes;
    }

    void release() {
        if (ptr_ != nullptr) {
            if (device_id_ >= 0) {
                DeviceGuard guard(device_id_);
                cudaFree(ptr_);
            } else {
                cudaFree(ptr_);
            }
            ptr_ = nullptr;
            bytes_ = 0;
            device_id_ = -1;
        }
    }

    void* data() {
        return ptr_;
    }

    const void* data() const {
        return ptr_;
    }

    std::size_t bytes() const {
        return bytes_;
    }

    bool empty() const {
        return ptr_ == nullptr || bytes_ == 0;
    }

    int device_id() const {
        return device_id_;
    }

    void swap(DeviceBuffer& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(bytes_, other.bytes_);
        std::swap(device_id_, other.device_id_);
    }

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
    int device_id_ = -1;
};

inline Status copy_h2d(void* dst_device, const void* src_host, std::size_t bytes, cudaStream_t stream = nullptr) {
    return QUADTRIX_CUDA_CHECK(cudaMemcpyAsync(dst_device, src_host, bytes, cudaMemcpyHostToDevice, stream));
}

inline Status copy_d2h(void* dst_host, const void* src_device, std::size_t bytes, cudaStream_t stream = nullptr) {
    return QUADTRIX_CUDA_CHECK(cudaMemcpyAsync(dst_host, src_device, bytes, cudaMemcpyDeviceToHost, stream));
}

inline Status copy_d2d(void* dst_device, const void* src_device, std::size_t bytes, cudaStream_t stream = nullptr) {
    return QUADTRIX_CUDA_CHECK(cudaMemcpyAsync(dst_device, src_device, bytes, cudaMemcpyDeviceToDevice, stream));
}

inline Status memset_device(void* dst_device, int value, std::size_t bytes, cudaStream_t stream = nullptr) {
    return QUADTRIX_CUDA_CHECK(cudaMemsetAsync(dst_device, value, bytes, stream));
}

}  // namespace cuda
}  // namespace quadtrix
