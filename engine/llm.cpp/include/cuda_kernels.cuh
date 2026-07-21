#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cmath>
#include <cuda_runtime.h>

// 1. Element-wise Bias Addition Kernel
__global__ void k_add_bias(float *out, const float *bias, int N, int E)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < N * E)
      {
            int e = idx % E;
            out[idx] += bias[e];
      }
}

// 2. Parallel ReLU Kernel
__global__ void k_relu(const float *in, float *out, int N)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < N)
      {
            out[idx] = fmaxf(0.0f, in[idx]);
      }
}

// 3. Parallel Layer Normalization Kernel
__global__ void k_layernorm_forward(const float *x,
                                    const float *gamma,
                                    const float *beta,
                                    float *out,
                                    int C,
                                    float eps)
{
      int row = blockIdx.x; // Each CUDA block handles one row (Batch * Sequence)
      const float *row_x = x + row * C;
      float *row_out = out + row * C;

      __shared__ float s_mean;
      __shared__ float s_var;

      if (threadIdx.x == 0)
      {
            s_mean = 0.0f;
            s_var = 0.0f;
      }
      __syncthreads();

      // Sum mean
      float local_sum = 0.0f;
      for (int c = threadIdx.x; c < C; c += blockDim.x)
      {
            local_sum += row_x[c];
      }
      atomicAdd(&s_mean, local_sum / C);
      __syncthreads();

      // Sum variance
      float local_var = 0.0f;
      for (int c = threadIdx.x; c < C; c += blockDim.x)
      {
            float diff = row_x[c] - s_mean;
            local_var += diff * diff;
      }
      atomicAdd(&s_var, local_var / C);
      __syncthreads();

      // Normalize and scale
      float inv_std = rsqrtf(s_var + eps);
      for (int c = threadIdx.x; c < C; c += blockDim.x)
      {
            row_out[c] = (row_x[c] - s_mean) * inv_std * gamma[c] + beta[c];
      }
}

// 4. Parallel Softmax Kernel
__global__ void k_softmax_forward(const float *in, float *out, int C)
{
      int row = blockIdx.x;
      const float *row_in = in + row * C;
      float *row_out = out + row * C;

      __shared__ float s_max;
      __shared__ float s_sum;

      if (threadIdx.x == 0)
      {
            s_max = -1e30f;
            s_sum = 0.0f;
      }
      __syncthreads();

      // Find max value
      float local_max = -1e30f;
      for (int c = threadIdx.x; c < C; c += blockDim.x)
      {
            local_max = fmaxf(local_max, row_in[c]);
      }
      // Atomic max via integer conversion
      atomicMax((int *)&s_max, __float_as_int(local_max));
      __syncthreads();

      // Exponentiate and sum
      float local_sum = 0.0f;
      for (int c = threadIdx.x; c < C; c += blockDim.x)
      {
            float e = expf(row_in[c] - s_max);
            row_out[c] = e;
            local_sum += e;
      }
      atomicAdd(&s_sum, local_sum);
      __syncthreads();

      // Normalize
      float inv_sum = 1.0f / s_sum;
      for (int c = threadIdx.x; c < C; c += blockDim.x)
      {
            row_out[c] *= inv_sum;
      }
}

#endif