#pragma once

#include "common.h"
#include "memory.cuh"

#include <array>
#include <cstddef>
#include <cstdint>

namespace quadtrix {
namespace cuda {

constexpr int kMaxTensorDims = 8;

struct TensorShape {
    int rank = 0;
    std::array<std::int64_t, kMaxTensorDims> dims{};
    std::array<std::int64_t, kMaxTensorDims> strides{};

    static TensorShape contiguous(const std::int64_t* sizes, int ndim) {
        if (ndim < 1 || ndim > kMaxTensorDims) {
            std::fprintf(stderr, "Tensor rank %d is outside supported range [1, %d]\n", ndim, kMaxTensorDims);
            std::abort();
        }

        TensorShape shape;
        shape.rank = ndim;
        for (int i = 0; i < ndim; ++i) {
            if (sizes[i] <= 0) {
                std::fprintf(stderr, "Tensor dimension %d must be positive, got %lld\n", i, static_cast<long long>(sizes[i]));
                std::abort();
            }
            shape.dims[i] = sizes[i];
        }

        std::int64_t stride = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            shape.strides[i] = stride;
            stride *= shape.dims[i];
        }
        return shape;
    }

    std::size_t numel() const {
        std::size_t total = 1;
        for (int i = 0; i < rank; ++i) {
            if (dims[i] <= 0) {
                return 0;
            }
            std::size_t next = 0;
            if (!checked_mul(total, static_cast<std::size_t>(dims[i]), &next)) {
                return 0;
            }
            total = next;
        }
        return rank == 0 ? 0 : total;
    }

    bool is_contiguous() const {
        std::int64_t expected = 1;
        for (int i = rank - 1; i >= 0; --i) {
            if (strides[i] != expected) {
                return false;
            }
            expected *= dims[i];
        }
        return true;
    }
};

struct TensorView {
    void* data = nullptr;
    TensorShape shape;
    DType dtype = DType::F32;
    DeviceKind device = DeviceKind::CUDA;
    int device_id = 0;

    std::size_t numel() const {
        return shape.numel();
    }

    std::size_t bytes() const {
        std::size_t out = 0;
        if (!checked_mul(numel(), dtype_size(dtype), &out)) {
            return 0;
        }
        return out;
    }

    template <typename T>
    T* data_as() {
        return static_cast<T*>(data);
    }

    template <typename T>
    const T* data_as() const {
        return static_cast<const T*>(data);
    }
};

class Tensor {
public:
    Tensor() = default;

    Tensor(const std::int64_t* dims, int rank, DType dtype, int device_id = 0)
        : shape_(TensorShape::contiguous(dims, rank)), dtype_(dtype), device_id_(device_id) {
        allocate();
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    TensorView view() {
        return {storage_.data(), shape_, dtype_, DeviceKind::CUDA, device_id_};
    }

    TensorView view() const {
        return {const_cast<void*>(storage_.data()), shape_, dtype_, DeviceKind::CUDA, device_id_};
    }

    const TensorShape& shape() const {
        return shape_;
    }

    DType dtype() const {
        return dtype_;
    }

    int device_id() const {
        return device_id_;
    }

    std::size_t numel() const {
        return shape_.numel();
    }

    std::size_t bytes() const {
        return storage_.bytes();
    }

    void* data() {
        return storage_.data();
    }

    const void* data() const {
        return storage_.data();
    }

private:
    void allocate() {
        std::size_t bytes = 0;
        if (!checked_mul(shape_.numel(), dtype_size(dtype_), &bytes)) {
            std::fprintf(stderr, "Tensor allocation size overflow\n");
            std::abort();
        }
        storage_.allocate(bytes, device_id_);
    }

    TensorShape shape_;
    DType dtype_ = DType::F32;
    int device_id_ = 0;
    DeviceBuffer storage_;
};

}  // namespace cuda
}  // namespace quadtrix
