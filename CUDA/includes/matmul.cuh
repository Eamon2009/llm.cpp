#pragma once

#include "tensor.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace quadtrix {
namespace cuda {

enum class MatmulTranspose : std::uint8_t {
    None,
    Transpose,
};

struct BlasStatus {
    bool ok;
    cublasStatus_t cublas_status;
    const char* message;

    static BlasStatus success() {
        return {true, CUBLAS_STATUS_SUCCESS, "ok"};
    }

    static BlasStatus failure(cublasStatus_t status, const char* message) {
        return {false, status, message};
    }
};

const char* cublas_status_name(cublasStatus_t status);

class BlasHandle {
public:
    explicit BlasHandle(int device_id = 0);
    ~BlasHandle();

    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;

    BlasHandle(BlasHandle&& other) noexcept;
    BlasHandle& operator=(BlasHandle&& other) noexcept;

    cublasHandle_t get() const {
        return handle_;
    }

    int device_id() const {
        return device_id_;
    }

    BlasStatus set_stream(cudaStream_t stream);

private:
    cublasHandle_t handle_ = nullptr;
    int device_id_ = 0;
};

BlasStatus matmul(
    BlasHandle& handle,
    const TensorView& a,
    MatmulTranspose op_a,
    const TensorView& b,
    MatmulTranspose op_b,
    TensorView c,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr);

BlasStatus matmul_forward(
    BlasHandle& handle,
    const TensorView& input,
    const TensorView& weight,
    TensorView output,
    cudaStream_t stream = nullptr,
    float alpha = 1.0f,
    float beta = 0.0f);

BlasStatus matmul_backward_input(
    BlasHandle& handle,
    const TensorView& grad_output,
    const TensorView& weight,
    TensorView grad_input,
    cudaStream_t stream = nullptr,
    float alpha = 1.0f,
    float beta = 0.0f);

BlasStatus matmul_backward_weight(
    BlasHandle& handle,
    const TensorView& input,
    const TensorView& grad_output,
    TensorView grad_weight,
    cudaStream_t stream = nullptr,
    float alpha = 1.0f,
    float beta = 0.0f);

}  // namespace cuda
}  // namespace quadtrix
