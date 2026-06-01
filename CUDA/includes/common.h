#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace quadtrix {
namespace cuda {

enum class DType : std::uint8_t {
    F32,
    F16,
    BF16,
    I32,
    U8,
};

enum class DeviceKind : std::uint8_t {
    CPU,
    CUDA,
};

struct Status {
    bool ok;
    cudaError_t cuda_error;
    const char* message;

    static Status success() {
        return {true, cudaSuccess, "ok"};
    }

    static Status failure(cudaError_t error, const char* message) {
        return {false, error, message};
    }
};

inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return "f32";
        case DType::F16:
            return "f16";
        case DType::BF16:
            return "bf16";
        case DType::I32:
            return "i32";
        case DType::U8:
            return "u8";
    }
    return "unknown";
}

inline std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return 4;
        case DType::F16:
            return 2;
        case DType::BF16:
            return 2;
        case DType::I32:
            return 4;
        case DType::U8:
            return 1;
    }

    std::fprintf(stderr, "Unknown CUDA dtype value %u\n", static_cast<unsigned int>(dtype));
    std::abort();
}

inline bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* out) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

inline Status check_cuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) {
        return Status::success();
    }

    std::fprintf(
        stderr,
        "CUDA error at %s:%d: %s failed with %s\n",
        file,
        line,
        expression,
        cudaGetErrorString(error));
    return Status::failure(error, expression);
}

inline void abort_on_cuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) {
        return;
    }

    std::fprintf(
        stderr,
        "Fatal CUDA error at %s:%d: %s failed with %s\n",
        file,
        line,
        expression,
        cudaGetErrorString(error));
    std::abort();
}

}  // namespace cuda
}  // namespace quadtrix

#define QUADTRIX_CUDA_CHECK(expr) \
    ::quadtrix::cuda::check_cuda((expr), #expr, __FILE__, __LINE__)

#define QUADTRIX_CUDA_ABORT(expr) \
    ::quadtrix::cuda::abort_on_cuda((expr), #expr, __FILE__, __LINE__)
