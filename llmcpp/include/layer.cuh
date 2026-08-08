#ifndef CUDA_LAYERS_CUH
#define CUDA_LAYERS_CUH

#include "cuda_kernels.cuh"
#include "cuda_tensor.cuh"

// Fast Matrix Multiplication via cuBLAS GEMM
inline void
gpu_matmul(cublasHandle_t handle, CUDATensor &A, CUDATensor &W, CUDATensor &C, int M, int N, int K)
{
      A.to_device();
      W.to_device();
      C.allocate_device();

      float alpha = 1.0f;
      float beta = 0.0f;

      // C (M x N) = A (M x K) * W (K x N)
      CUBLAS_CHECK(cublasSgemm(handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
                               N,
                               M,
                               K,
                               &alpha,
                               W.d_data,
                               N,
                               A.d_data,
                               K,
                               &beta,
                               C.d_data,
                               N));
}

// Forward LayerNorm Wrapper
inline void gpu_layernorm(CUDATensor &X, CUDATensor &Gamma, CUDATensor &Beta, CUDATensor &Out)
{
      X.to_device();
      Gamma.to_device();
      Beta.to_device();
      Out.allocate_device();

      int total_rows = X.shape[0] * X.shape[1];
      int C = X.shape[2];

      int threads = 256;
      int blocks = total_rows;

      k_layernorm_forward<<<blocks, threads>>>(X.d_data,
                                               Gamma.d_data,
                                               Beta.d_data,
                                               Out.d_data,
                                               C,
                                               1e-5f);
      CUDA_CHECK(cudaGetLastError());
}

// Forward Softmax Wrapper
inline void gpu_softmax(CUDATensor &In, CUDATensor &Out)
{
      In.to_device();
      Out.allocate_device();

      int total_rows = In.numel() / In.shape.back();
      int C = In.shape.back();

      int threads = 256;
      int blocks = total_rows;

      k_softmax_forward<<<blocks, threads>>>(In.d_data, Out.d_data, C);
      CUDA_CHECK(cudaGetLastError());
}

#endif