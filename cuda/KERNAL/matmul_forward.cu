#include "../includes/matmul.cuh"

#include "../includes/runtime.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <cstdio>
#include <limits>

namespace quadtrix {
namespace cuda {
namespace {

const char* kInvalidMatmul = "invalid matmul arguments";

cublasOperation_t to_cublas_op(MatmulTranspose op) {
    return op == MatmulTranspose::Transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
}

bool fits_int(std::int64_t value) {
    return value > 0 && value <= std::numeric_limits<int>::max();
}

bool is_rank2_contiguous_cuda(const TensorView& tensor) {
    return tensor.data != nullptr && tensor.device == DeviceKind::CUDA && tensor.shape.rank == 2 &&
           tensor.shape.is_contiguous();
}

std::int64_t rows_after_op(const TensorView& tensor, MatmulTranspose op) {
    return op == MatmulTranspose::Transpose ? tensor.shape.dims[1] : tensor.shape.dims[0];
}

std::int64_t cols_after_op(const TensorView& tensor, MatmulTranspose op) {
    return op == MatmulTranspose::Transpose ? tensor.shape.dims[0] : tensor.shape.dims[1];
}

cudaDataType_t to_cuda_data_type(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return CUDA_R_32F;
        case DType::F16:
            return CUDA_R_16F;
        case DType::BF16:
            return CUDA_R_16BF;
        case DType::I32:
        case DType::U8:
            break;
    }
    return CUDA_R_32F;
}

cublasComputeType_t compute_type_for(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return CUBLAS_COMPUTE_32F;
        case DType::F16:
        case DType::BF16:
            return CUBLAS_COMPUTE_32F_FAST_16F;
        case DType::I32:
        case DType::U8:
            break;
    }
    return CUBLAS_COMPUTE_32F;
}

cublasGemmAlgo_t gemm_algo_for(DType dtype) {
    return dtype == DType::F32 ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
}

bool supports_gemm_dtype(DType dtype) {
    return dtype == DType::F32 || dtype == DType::F16 || dtype == DType::BF16;
}

}  // namespace

const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
#ifdef CUBLAS_STATUS_LICENSE_ERROR
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    }
    return "CUBLAS_STATUS_UNKNOWN";
}

BlasHandle::BlasHandle(int device_id) : device_id_(device_id) {
    DeviceGuard guard(device_id_);
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "Fatal cuBLAS error: cublasCreate failed with %s\n", cublas_status_name(status));
        std::abort();
    }
    status = cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "Fatal cuBLAS error: cublasSetMathMode failed with %s\n", cublas_status_name(status));
        std::abort();
    }
}

BlasHandle::~BlasHandle() {
    if (handle_ != nullptr) {
        DeviceGuard guard(device_id_);
        cublasDestroy(handle_);
    }
}

BlasHandle::BlasHandle(BlasHandle&& other) noexcept
    : handle_(other.handle_), device_id_(other.device_id_) {
    other.handle_ = nullptr;
}

BlasHandle& BlasHandle::operator=(BlasHandle&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (handle_ != nullptr) {
        DeviceGuard guard(device_id_);
        cublasDestroy(handle_);
    }

    handle_ = other.handle_;
    device_id_ = other.device_id_;
    other.handle_ = nullptr;
    return *this;
}

BlasStatus BlasHandle::set_stream(cudaStream_t stream) {
    cublasStatus_t status = cublasSetStream(handle_, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return BlasStatus::failure(status, "cublasSetStream failed");
    }
    return BlasStatus::success();
}

BlasStatus matmul(
    BlasHandle& handle,
    const TensorView& a,
    MatmulTranspose op_a,
    const TensorView& b,
    MatmulTranspose op_b,
    TensorView c,
    float alpha,
    float beta,
    cudaStream_t stream) {
    if (!is_rank2_contiguous_cuda(a) || !is_rank2_contiguous_cuda(b) || !is_rank2_contiguous_cuda(c)) {
        return BlasStatus::failure(CUBLAS_STATUS_INVALID_VALUE, kInvalidMatmul);
    }
    if (a.dtype != b.dtype || a.dtype != c.dtype || !supports_gemm_dtype(a.dtype)) {
        return BlasStatus::failure(CUBLAS_STATUS_NOT_SUPPORTED, "matmul dtype is unsupported or mismatched");
    }
    if (a.device_id != b.device_id || a.device_id != c.device_id || a.device_id != handle.device_id()) {
        return BlasStatus::failure(CUBLAS_STATUS_INVALID_VALUE, "matmul tensors and handle must share a device");
    }

    const std::int64_t m64 = rows_after_op(a, op_a);
    const std::int64_t k64 = cols_after_op(a, op_a);
    const std::int64_t b_k64 = rows_after_op(b, op_b);
    const std::int64_t n64 = cols_after_op(b, op_b);

    if (k64 != b_k64 || c.shape.dims[0] != m64 || c.shape.dims[1] != n64) {
        return BlasStatus::failure(CUBLAS_STATUS_INVALID_VALUE, "matmul shape mismatch");
    }
    if (!fits_int(m64) || !fits_int(n64) || !fits_int(k64) || !fits_int(a.shape.dims[1]) ||
        !fits_int(b.shape.dims[1]) || !fits_int(c.shape.dims[1])) {
        return BlasStatus::failure(CUBLAS_STATUS_INVALID_VALUE, "matmul dimensions exceed cuBLAS int range");
    }

    DeviceGuard guard(handle.device_id());
    BlasStatus stream_status = handle.set_stream(stream);
    if (!stream_status.ok) {
        return stream_status;
    }

    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int k = static_cast<int>(k64);
    const int lda = static_cast<int>(a.shape.dims[1]);
    const int ldb = static_cast<int>(b.shape.dims[1]);
    const int ldc = static_cast<int>(c.shape.dims[1]);
    const cudaDataType_t dtype = to_cuda_data_type(a.dtype);

    cublasStatus_t status = cublasGemmEx(
        handle.get(),
        to_cublas_op(op_b),
        to_cublas_op(op_a),
        n,
        m,
        k,
        &alpha,
        b.data,
        dtype,
        ldb,
        a.data,
        dtype,
        lda,
        &beta,
        c.data,
        dtype,
        ldc,
        compute_type_for(a.dtype),
        gemm_algo_for(a.dtype));

    if (status != CUBLAS_STATUS_SUCCESS) {
        return BlasStatus::failure(status, "cublasGemmEx failed");
    }
    return BlasStatus::success();
}

BlasStatus matmul_forward(
    BlasHandle& handle,
    const TensorView& input,
    const TensorView& weight,
    TensorView output,
    cudaStream_t stream,
    float alpha,
    float beta) {
    return matmul(handle, input, MatmulTranspose::None, weight, MatmulTranspose::None, output, alpha, beta, stream);
}

}  // namespace cuda
}  // namespace quadtrix
