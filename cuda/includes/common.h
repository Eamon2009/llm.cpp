// Quadtrix CUDA Engine — common.h
//  Tensor Runtime
// NO kernel logic, NO device code lives here.
// This is a pure host-side / shared header.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda_fp16.h> // __half
#include <cuda_bf16.h> // __nv_bfloat16
#include <cublas_v2.h>

// Version / build tags

#define QUADTRIX_VERSION_MAJOR 0
#define QUADTRIX_VERSION_MINOR 1
#define QUADTRIX_VERSION_PATCH 0
// Compiler / architecture hints
#define QX_INLINE __forceinline__
#define QX_HOST __host__
#define QX_DEVICE __device__
#define QX_HOST_DEVICE __host__ __device__
#define QX_GLOBAL __global__
#define QX_RESTRICT __restrict__

// Prevent unused-variable warnings in release builds
#define QX_UNUSED(x) ((void)(x))
// Integer math helpers (host + device safe)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define ROUND_UP(x, y) (CEIL_DIV((x), (y)) * (y))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(x, lo, hi) (MIN(MAX((x), (lo)), (hi)))
#define IS_POW2(x) (((x) & ((x) - 1)) == 0) // Power-of-two utilities
#define NEXT_POW2(x) \
      ([](uint32_t v) { v--; v|=v>>1; v|=v>>2; v|=v>>4; v|=v>>8; v|=v>>16; return v+1; }((uint32_t)(x)))
#define QX_ALIGN(x) __align__(x)
#define QX_ALIGN_16 QX_ALIGN(16)
#define QX_ALIGN_32 QX_ALIGN(32)
#define QX_ALIGN_128 QX_ALIGN(128) // cache-line

// Memory alignment for device allocations (128 bytes = 2 cache lines,
// ensures coalesced access on any architecture)
static constexpr size_t QX_MEM_ALIGN = 128;
// ---- CUDA runtime errors ----
#define CUDA_CHECK(call)                                   \
      do                                                   \
      {                                                    \
            cudaError_t _err = (call);                     \
            if (_err != cudaSuccess)                       \
            {                                              \
                  fprintf(stderr,                          \
                          "[CUDA ERROR] %s:%d  call: %s\n" \
                          "             %s\n",             \
                          __FILE__, __LINE__, #call,       \
                          cudaGetErrorString(_err));       \
                  exit(EXIT_FAILURE);                      \
            }                                              \
      } while (0)

// Variant that returns instead of exit (use in non-fatal paths)
#define CUDA_CHECK_RETURN(call, retval)                         \
      do                                                        \
      {                                                         \
            cudaError_t _err = (call);                          \
            if (_err != cudaSuccess)                            \
            {                                                   \
                  fprintf(stderr,                               \
                          "[CUDA WARN] %s:%d  call: %s — %s\n", \
                          __FILE__, __LINE__, #call,            \
                          cudaGetErrorString(_err));            \
                  return (retval);                              \
            }                                                   \
      } while (0)

// ---- cuBLAS errors ----
#define CUBLAS_CHECK(call)                                                \
      do                                                                  \
      {                                                                   \
            cublasStatus_t _st = (call);                                  \
            if (_st != CUBLAS_STATUS_SUCCESS)                             \
            {                                                             \
                  fprintf(stderr,                                         \
                          "[cuBLAS ERROR] %s:%d  call: %s  status: %d\n", \
                          __FILE__, __LINE__, #call, (int)_st);           \
                  exit(EXIT_FAILURE);                                     \
            }                                                             \
      } while (0)

// ---- NCCL errors (included only when NCCL is available) ----
#ifdef NCCL_MAJOR
#include <nccl.h>
#define NCCL_CHECK(call)                                   \
      do                                                   \
      {                                                    \
            ncclResult_t _r = (call);                      \
            if (_r != ncclSuccess)                         \
            {                                              \
                  fprintf(stderr,                          \
                          "[NCCL ERROR] %s:%d  call: %s\n" \
                          "             %s\n",             \
                          __FILE__, __LINE__, #call,       \
                          ncclGetErrorString(_r));         \
                  exit(EXIT_FAILURE);                      \
            }                                              \
      } while (0)
#endif

//  cuDNN errors
#ifdef CUDNN_MAJOR
#include <cudnn.h>
#define CUDNN_CHECK(call)                                   \
      do                                                    \
      {                                                     \
            cudnnStatus_t _st = (call);                     \
            if (_st != CUDNN_STATUS_SUCCESS)                \
            {                                               \
                  fprintf(stderr,                           \
                          "[cuDNN ERROR] %s:%d  call: %s\n" \
                          "              %s\n",             \
                          __FILE__, __LINE__, #call,        \
                          cudnnGetErrorString(_st));        \
                  exit(EXIT_FAILURE);                       \
            }                                               \
      } while (0)
#endif // CUDNN_MAJOR

// Convenience: sync + check after kernel launches
#define CUDA_KERNEL_CHECK()                      \
      do                                         \
      {                                          \
            CUDA_CHECK(cudaGetLastError());      \
            CUDA_CHECK(cudaDeviceSynchronize()); \
      } while (0)

// Debug-only sync (compiles away in release)
#ifdef QX_DEBUG
#define CUDA_KERNEL_CHECK_DEBUG() CUDA_KERNEL_CHECK()
#else
#define CUDA_KERNEL_CHECK_DEBUG() CUDA_CHECK(cudaGetLastError())
#endif
// Drives runtime dtype dispatch throughout the engine.
// Every tensor carries one of these.
typedef enum
{
      DTYPE_FLOAT32 = 0,  // float       — 4 bytes — default training dtype
      DTYPE_FLOAT16 = 1,  // __half      — 2 bytes — mixed-precision forward
      DTYPE_BFLOAT16 = 2, // __nv_bfloat16 — 2 bytes — preferred on A100/H100
      DTYPE_INT32 = 3,    // int32_t     — 4 bytes — indices, token IDs
      DTYPE_INT8 = 4,     // int8_t      — 1 byte  — quantized inference
      DTYPE_BOOL = 5,     // uint8_t     — 1 byte  — masks
      DTYPE_UNKNOWN = 255
} DType;

// Human-readable dtype name (for logging)
static inline const char *dtype_name(DType d)
{
      switch (d)
      {
      case DTYPE_FLOAT32:
            return "float32";
      case DTYPE_FLOAT16:
            return "float16";
      case DTYPE_BFLOAT16:
            return "bfloat16";
      case DTYPE_INT32:
            return "int32";
      case DTYPE_INT8:
            return "int8";
      case DTYPE_BOOL:
            return "bool";
      default:
            return "unknown";
      }
}

// Byte size per element for a given dtype
static inline size_t dtype_size(DType d)
{
      switch (d)
      {
      case DTYPE_FLOAT32:
            return 4;
      case DTYPE_FLOAT16:
            return 2;
      case DTYPE_BFLOAT16:
            return 2;
      case DTYPE_INT32:
            return 4;
      case DTYPE_INT8:
            return 1;
      case DTYPE_BOOL:
            return 1;
      default:
            fprintf(stderr, "[QX] dtype_size: unknown dtype %d\n", (int)d);
            exit(EXIT_FAILURE);
      }
}

// Map DType → cublas compute / data types (used in matmul wrappers)
static inline cudaDataType_t dtype_to_cuda(DType d)
{
      switch (d)
      {
      case DTYPE_FLOAT32:
            return CUDA_R_32F;
      case DTYPE_FLOAT16:
            return CUDA_R_16F;
      case DTYPE_BFLOAT16:
            return CUDA_R_16BF;
      case DTYPE_INT8:
            return CUDA_R_8I;
      default:
            fprintf(stderr, "[QX] dtype_to_cuda: unsupported dtype %d\n", (int)d);
            exit(EXIT_FAILURE);
      }
}

static constexpr int QX_MAX_DIMS = 5;    // max tensor rank supported
static constexpr int QX_MAX_DEVICES = 8; // max GPUs in a node
static constexpr int QX_BLOCK_SIZE_SMALL = 128;
static constexpr int QX_BLOCK_SIZE = 256;
static constexpr int QX_BLOCK_SIZE_LARGE = 512;
static constexpr int QX_BLOCK_SIZE_MAX = 1024;
static constexpr int QX_WARP_SIZE = 32;
static constexpr int QX_MAX_GRID_X = 65535;
// Memory location enum
//   Tracks where a tensor buffer actually lives.
typedef enum
{
      MEM_HOST = 0,        // CPU
      MEM_HOST_PINNED = 1, // CPU pinned — fast H2D/D2H transfers
      MEM_DEVICE = 2,      // GPU global memory
      MEM_MANAGED = 3,     // CUDA unified / managed memory
      MEM_UNKNOWN = 255
} MemLocation;

//
// Training phase enum
// Passed into forward/backward launchers so they can skip
// gradient computation during inference.
typedef enum
{
      PHASE_TRAIN = 0,
      PHASE_EVAL = 1,
      PHASE_INFER = 2
} TrainingPhase;
// Reduction operation enum
//   Used by reduction kernels (global_norm, softmax, layernorm, etc.)
typedef enum
{
      REDUCE_SUM = 0,
      REDUCE_MAX = 1,
      REDUCE_MIN = 2,
      REDUCE_MEAN = 3,
      REDUCE_PROD = 4
} ReduceOp;
static constexpr float QX_EPS_F32 = 1e-6f; // layernorm / division guard
static constexpr float QX_EPS_F16 = 1e-4f; // fp16 is less precise
static constexpr float QX_INF_F32 = 1e38f; // softmax init sentinel
static constexpr float QX_NEG_INF_F32 = -1e38f;
// GELU approximation constant (tanh variant)
static constexpr float QX_GELU_COEFF = 0.044715f;
static constexpr float QX_SQRT_2_PI = 0.7978845608f; // sqrt(2/pi)
static constexpr float QX_ATT_SCALE_AUTO = -1.0f;
static_assert(sizeof(float) == 4, "float must be 4 bytes");
static_assert(sizeof(__half) == 2, "__half must be 2 bytes");
static_assert(sizeof(__nv_bfloat16) == 2, "bfloat16 must be 2 bytes");
static_assert(QX_WARP_SIZE == 32, "warp size invariant broken");