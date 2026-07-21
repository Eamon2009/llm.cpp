#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                                           \
      do                                                                                           \
      {                                                                                            \
            cudaError_t err = call;                                                                \
            if (err != cudaSuccess)                                                                \
            {                                                                                      \
                  std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__     \
                            << ":" << __LINE__ << std::endl;                                       \
                  exit(EXIT_FAILURE);                                                              \
            }                                                                                      \
      } while (0)

#define CUBLAS_CHECK(call)                                                                         \
      do                                                                                           \
      {                                                                                            \
            cublasStatus_t stat = call;                                                            \
            if (stat != CUBLAS_STATUS_SUCCESS)                                                     \
            {                                                                                      \
                  std::cerr << "cuBLAS Error Code " << stat << " at " << __FILE__ << ":"           \
                            << __LINE__ << std::endl;                                              \
                  exit(EXIT_FAILURE);                                                              \
            }                                                                                      \
      } while (0)

#endif