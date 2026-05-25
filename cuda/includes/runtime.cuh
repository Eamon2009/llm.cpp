#pragma once

#include "common.h"

#include <cuda_runtime.h>

namespace quadtrix {
namespace cuda {

struct DeviceGuard {
    int previous_device = -1;
    bool changed = false;

    explicit DeviceGuard(int device_id) {
        QUADTRIX_CUDA_ABORT(cudaGetDevice(&previous_device));
        if (previous_device != device_id) {
            QUADTRIX_CUDA_ABORT(cudaSetDevice(device_id));
            changed = true;
        }
    }

    ~DeviceGuard() {
        if (changed) {
            cudaSetDevice(previous_device);
        }
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
};

struct Stream {
    cudaStream_t handle = nullptr;
    bool owns = false;

    Stream() = default;

    explicit Stream(cudaStream_t external_handle) : handle(external_handle), owns(false) {}

    static Stream create(unsigned int flags = cudaStreamNonBlocking) {
        Stream stream;
        QUADTRIX_CUDA_ABORT(cudaStreamCreateWithFlags(&stream.handle, flags));
        stream.owns = true;
        return stream;
    }

    ~Stream() {
        if (owns && handle != nullptr) {
            cudaStreamDestroy(handle);
        }
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept : handle(other.handle), owns(other.owns) {
        other.handle = nullptr;
        other.owns = false;
    }

    Stream& operator=(Stream&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (owns && handle != nullptr) {
            cudaStreamDestroy(handle);
        }
        handle = other.handle;
        owns = other.owns;
        other.handle = nullptr;
        other.owns = false;
        return *this;
    }

    void synchronize() const {
        QUADTRIX_CUDA_ABORT(cudaStreamSynchronize(handle));
    }
};

inline int current_device() {
    int device = 0;
    QUADTRIX_CUDA_ABORT(cudaGetDevice(&device));
    return device;
}

inline int device_count() {
    int count = 0;
    QUADTRIX_CUDA_ABORT(cudaGetDeviceCount(&count));
    return count;
}

}  // namespace cuda
}  // namespace quadtrix
