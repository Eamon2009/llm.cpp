#pragma once

/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2026 Eamon Sippy
 *
 * Flash Attention CUDA – Multi-Head Attention
 * No single-headed Head struct.  Native multi-head fused kernels.
 * FP32, causal masking, online softmax, recomputation backward.
 * NOTE: This is for now a test the concept implementation.
 *       The GEMMs are simple loop kernels, not cuBLAS.
 *       For production, swap the matmul helpers with cublasGemmEx.
 */

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace fa
{
// parameters
// ------------------------------------------------------------------
constexpr int FWD_BR = 64; // queries per block (forward)
constexpr int FWD_BC = 64; // keys/values per tile (forward)
constexpr int BWD_BR = 64; // queries per block (backward)
constexpr int BWD_BC = 64; // keys per block (backward)

// ------------------------------------------------------------------
__global__ void matmul_fwd_kernel(const float *__restrict__ A, const float *__restrict__ W,
                                  const float *__restrict__ bias, float *__restrict__ C, int M,
                                  int K, int N)
{
      int row = blockIdx.y * blockDim.y + threadIdx.y;
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (row >= M || col >= N)
            return;
      float sum = (bias) ? bias[col] : 0.0f;
      for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * W[col * K + k];
      C[row * N + col] = sum;
}

__global__ void matmul_bwd_dA_kernel(const float *__restrict__ dC, const float *__restrict__ W,
                                     float *__restrict__ dA, int M, int K, int N)
{
      int row = blockIdx.y * blockDim.y + threadIdx.y;
      int k = blockIdx.x * blockDim.x + threadIdx.x;
      if (row >= M || k >= K)
            return;
      float sum = 0.0f;
      for (int col = 0; col < N; ++col)
            sum += dC[row * N + col] * W[col * K + k];
      dA[row * K + k] = sum;
}

__global__ void matmul_bwd_dW_kernel(const float *__restrict__ A, const float *__restrict__ dC,
                                     float *__restrict__ dW, int M, int K, int N)
{
      int col = blockIdx.y * blockDim.y + threadIdx.y;
      int k = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= N || k >= K)
            return;
      float sum = 0.0f;
      for (int row = 0; row < M; ++row)
            sum += A[row * K + k] * dC[row * N + col];
      dW[col * K + k] = sum;
}

__global__ void bias_bwd_kernel(const float *__restrict__ dC, float *__restrict__ db, int M, int N)
{
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= N)
            return;
      float sum = 0.0f;
      for (int row = 0; row < M; ++row)
            sum += dC[row * N + col];
      atomicAdd(&db[col], sum);
}

__global__ void sgd_step_kernel(float *__restrict__ w, const float *__restrict__ dw, float lr,
                                int n)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n)
            w[idx] -= lr * dw[idx];
}
__global__ void reshape_to_bhtd_kernel(const float *__restrict__ in, float *__restrict__ out, int B,
                                       int T, int n_head, int D)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = B * T * n_head * D;
      if (idx >= total)
            return;

      int d = idx % D;
      int h = (idx / D) % n_head;
      int t = (idx / (D * n_head)) % T;
      int b = idx / (D * n_head * T);

      int out_idx = ((b * n_head + h) * T + t) * D + d;
      out[out_idx] = in[idx];
}

__global__ void reshape_from_bhtd_kernel(const float *__restrict__ in, float *__restrict__ out,
                                         int B, int T, int n_head, int D)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = B * T * n_head * D;
      if (idx >= total)
            return;

      int d = idx % D;
      int h = (idx / D) % n_head;
      int t = (idx / (D * n_head)) % T;
      int b = idx / (D * n_head * T);

      int in_idx = ((b * n_head + h) * T + t) * D + d;
      out[idx] = in[in_idx];
}

__global__ void dropout_fwd_kernel(const float *__restrict__ in, float *__restrict__ out,
                                   float *__restrict__ mask, int n, float p, unsigned int seed)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= n)
            return;
      curandState state;
      curand_init(seed, idx, 0, &state);
      float r = curand_uniform(&state);
      float keep = (r >= p) ? (1.0f / (1.0f - p)) : 0.0f;
      mask[idx] = keep;
      out[idx] = in[idx] * keep;
}

__global__ void dropout_bwd_kernel(const float *__restrict__ dout, const float *__restrict__ mask,
                                   float *__restrict__ din, int n)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= n)
            return;
      din[idx] = dout[idx] * mask[idx];
}

__global__ void elemwise_add_3in1_kernel(const float *__restrict__ a, const float *__restrict__ b,
                                         const float *__restrict__ c, float *__restrict__ out,
                                         int n)
{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= n)
            return;
      out[idx] = a[idx] + b[idx] + c[idx];
}

template <int Bc, int D>
__global__ void flash_attn_fwd_kernel(const float *__restrict__ Q, const float *__restrict__ K,
                                      const float *__restrict__ V, float *__restrict__ O,
                                      float *__restrict__ LSE, int N, float scale)
{
      int bh = blockIdx.x;
      int q_tile = blockIdx.y;
      int q_idx = q_tile * blockDim.x + threadIdx.x;
      if (q_idx >= N)
            return;

      const float *q_ptr = Q + bh * N * D;
      const float *k_ptr = K + bh * N * D;
      const float *v_ptr = V + bh * N * D;
      float *o_ptr = O + bh * N * D;
      float *lse_ptr = LSE + bh * N;
      float q_reg[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
            q_reg[d] = q_ptr[q_idx * D + d] * scale;

      float m = -INFINITY;
      float l = 0.0f;
      float o_reg[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
            o_reg[d] = 0.0f;

      __shared__ float K_smem[Bc * D];
      __shared__ float V_smem[Bc * D];

      int num_kv_tiles = (N + Bc - 1) / Bc;
      for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile)
      {
            int kv_start = kv_tile * Bc;
            for (int i = threadIdx.x; i < Bc * D; i += blockDim.x)
            {
                  int k_seq = kv_start + i / D;
                  int d = i % D;
                  if (k_seq < N)
                  {
                        K_smem[i] = k_ptr[k_seq * D + d];
                        V_smem[i] = v_ptr[k_seq * D + d];
                  }
                  else
                  {
                        K_smem[i] = 0.0f;
                        V_smem[i] = 0.0f;
                  }
            }
            __syncthreads();
            float s_local[Bc];
#pragma unroll
            for (int k_j = 0; k_j < Bc; ++k_j)
            {
                  int global_k = kv_start + k_j;
                  if (global_k >= N || global_k > q_idx)
                  {
                        s_local[k_j] = -INFINITY;
                  }
                  else
                  {
                        float acc = 0.0f;
#pragma unroll
                        for (int d = 0; d < D; ++d)
                              acc += q_reg[d] * K_smem[k_j * D + d];
                        s_local[k_j] = acc;
                  }
            }
            float m_prev = m;
            float m_new = m_prev;
#pragma unroll
            for (int k_j = 0; k_j < Bc; ++k_j)
                  m_new = fmaxf(m_new, s_local[k_j]);

            float l_new = 0.0f;
#pragma unroll
            for (int k_j = 0; k_j < Bc; ++k_j)
                  if (s_local[k_j] != -INFINITY)
                        l_new += expf(s_local[k_j] - m_new);
            l_new += expf(m_prev - m_new) * l;

            float scale_o = expf(m_prev - m_new);
#pragma unroll
            for (int d = 0; d < D; ++d)
                  o_reg[d] = o_reg[d] * scale_o;

#pragma unroll
            for (int k_j = 0; k_j < Bc; ++k_j)
            {
                  if (s_local[k_j] != -INFINITY)
                  {
                        float p = expf(s_local[k_j] - m_new);
#pragma unroll
                        for (int d = 0; d < D; ++d)
                              o_reg[d] += p * V_smem[k_j * D + d];
                  }
            }

            m = m_new;
            l = l_new;
            __syncthreads();
      }

      float l_inv = 1.0f / l;
#pragma unroll
      for (int d = 0; d < D; ++d)
            o_ptr[q_idx * D + d] = o_reg[d] * l_inv;

      lse_ptr[q_idx] = m + logf(l);
}

template <int Bc, int D>
__global__ void flash_attn_bwd_dq_kernel(const float *__restrict__ Q, const float *__restrict__ K,
                                         const float *__restrict__ V, const float *__restrict__ O,
                                         const float *__restrict__ dO,
                                         const float *__restrict__ LSE, float *__restrict__ dQ,
                                         int N, float scale)
{
      int bh = blockIdx.x;
      int q_idx = blockIdx.y * blockDim.x + threadIdx.x;
      if (q_idx >= N)
            return;

      const float *q_ptr = Q + bh * N * D;
      const float *k_ptr = K + bh * N * D;
      const float *v_ptr = V + bh * N * D;
      const float *o_ptr = O + bh * N * D;
      const float *do_ptr = dO + bh * N * D;
      const float *lse_ptr = LSE + bh * N;
      float *dq_ptr = dQ + bh * N * D;

      float q_reg[D], o_reg[D], do_reg[D];
      float D_acc = 0.0f;
#pragma unroll
      for (int d = 0; d < D; ++d)
      {
            q_reg[d] = q_ptr[q_idx * D + d] * scale;
            o_reg[d] = o_ptr[q_idx * D + d];
            do_reg[d] = do_ptr[q_idx * D + d];
            D_acc += do_reg[d] * o_reg[d];
      }

      float lse = lse_ptr[q_idx];
      float dq_acc[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
            dq_acc[d] = 0.0f;

      __shared__ float K_smem[Bc * D];
      __shared__ float V_smem[Bc * D];

      int num_kv_tiles = (N + Bc - 1) / Bc;
      for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile)
      {
            int kv_start = kv_tile * Bc;
            for (int i = threadIdx.x; i < Bc * D; i += blockDim.x)
            {
                  int k_seq = kv_start + i / D;
                  int d = i % D;
                  if (k_seq < N)
                  {
                        K_smem[i] = k_ptr[k_seq * D + d];
                        V_smem[i] = v_ptr[k_seq * D + d];
                  }
                  else
                  {
                        K_smem[i] = 0.0f;
                        V_smem[i] = 0.0f;
                  }
            }
            __syncthreads();

#pragma unroll
            for (int k_j = 0; k_j < Bc; ++k_j)
            {
                  int global_k = kv_start + k_j;
                  if (global_k >= N || global_k > q_idx)
                        continue;

                  float s = 0.0f;
#pragma unroll
                  for (int d = 0; d < D; ++d)
                        s += q_reg[d] * K_smem[k_j * D + d];

                  float p = expf(s - lse);
                  float dP = 0.0f;
#pragma unroll
                  for (int d = 0; d < D; ++d)
                        dP += do_reg[d] * V_smem[k_j * D + d];

                  float dS = p * (dP - D_acc);
#pragma unroll
                  for (int d = 0; d < D; ++d)
                        dq_acc[d] += dS * K_smem[k_j * D + d];
            }
            __syncthreads();
      }

#pragma unroll
      for (int d = 0; d < D; ++d)
            dq_ptr[q_idx * D + d] = dq_acc[d] * scale;
}
template <int Br, int D>
__global__ void flash_attn_bwd_dkdv_kernel(const float *__restrict__ Q, const float *__restrict__ K,
                                           const float *__restrict__ V, const float *__restrict__ O,
                                           const float *__restrict__ dO,
                                           const float *__restrict__ LSE, float *__restrict__ dK,
                                           float *__restrict__ dV, int N, float scale)
{
      int bh = blockIdx.x;
      int k_idx = blockIdx.y * blockDim.x + threadIdx.x;
      if (k_idx >= N)
            return;

      const float *q_ptr = Q + bh * N * D;
      const float *k_ptr = K + bh * N * D;
      const float *v_ptr = V + bh * N * D;
      const float *o_ptr = O + bh * N * D;
      const float *do_ptr = dO + bh * N * D;
      const float *lse_ptr = LSE + bh * N;
      float *dk_ptr = dK + bh * N * D;
      float *dv_ptr = dV + bh * N * D;

      float k_reg[D], v_reg[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
      {
            k_reg[d] = k_ptr[k_idx * D + d];
            v_reg[d] = v_ptr[k_idx * D + d];
      }

      float dk_acc[D], dv_acc[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
      {
            dk_acc[d] = 0.0f;
            dv_acc[d] = 0.0f;
      }

      __shared__ float Q_smem[Br * D];
      __shared__ float O_smem[Br * D];
      __shared__ float dO_smem[Br * D];
      __shared__ float LSE_smem[Br];

      int num_q_tiles = (N + Br - 1) / Br;
      for (int q_tile = 0; q_tile < num_q_tiles; ++q_tile)
      {
            int q_start = q_tile * Br;

            for (int i = threadIdx.x; i < Br; i += blockDim.x)
            {
                  int q = q_start + i;
                  if (q < N)
                  {
                        LSE_smem[i] = lse_ptr[q];
#pragma unroll
                        for (int d = 0; d < D; ++d)
                        {
                              Q_smem[i * D + d] = q_ptr[q * D + d] * scale;
                              O_smem[i * D + d] = o_ptr[q * D + d];
                              dO_smem[i * D + d] = do_ptr[q * D + d];
                        }
                  }
                  else
                  {
                        LSE_smem[i] = 0.0f;
#pragma unroll
                        for (int d = 0; d < D; ++d)
                        {
                              Q_smem[i * D + d] = 0.0f;
                              O_smem[i * D + d] = 0.0f;
                              dO_smem[i * D + d] = 0.0f;
                        }
                  }
            }
            __syncthreads();

#pragma unroll
            for (int q_i = 0; q_i < Br; ++q_i)
            {
                  int global_q = q_start + q_i;
                  if (global_q >= N || k_idx > global_q)
                        continue;

                  float s = 0.0f;
#pragma unroll
                  for (int d = 0; d < D; ++d)
                        s += Q_smem[q_i * D + d] * k_reg[d];

                  float lse = LSE_smem[q_i];
                  float p = expf(s - lse);

                  float dP = 0.0f;
#pragma unroll
                  for (int d = 0; d < D; ++d)
                        dP += dO_smem[q_i * D + d] * v_reg[d];

                  float D_acc = 0.0f;
#pragma unroll
                  for (int d = 0; d < D; ++d)
                        D_acc += dO_smem[q_i * D + d] * O_smem[q_i * D + d];

                  float dS = p * (dP - D_acc);
#pragma unroll
                  for (int d = 0; d < D; ++d)
                  {
                        dk_acc[d] += dS * Q_smem[q_i * D + d];
                        dv_acc[d] += p * dO_smem[q_i * D + d];
                  }
            }
            __syncthreads();
      }

#pragma unroll
      for (int d = 0; d < D; ++d)
      {
            dk_ptr[k_idx * D + d] = dk_acc[d];
            dv_ptr[k_idx * D + d] = dv_acc[d];
      }
}
// ------------------------------------------------------------------
template <int D>
inline void launch_flash_attn_fwd(const float *Q, const float *K, const float *V, float *O,
                                  float *LSE, int B, int n_head, int T, float scale,
                                  cudaStream_t stream)
{
      int total_heads = B * n_head;
      int num_q_tiles = (T + FWD_BR - 1) / FWD_BR;
      dim3 grid(total_heads, num_q_tiles);
      dim3 block(FWD_BR);
      flash_attn_fwd_kernel<FWD_BC, D><<<grid, block, 0, stream>>>(Q, K, V, O, LSE, T, scale);
}

template <int D>
inline void launch_flash_attn_bwd(const float *Q, const float *K, const float *V, const float *O,
                                  const float *dO, const float *LSE, float *dQ, float *dK,
                                  float *dV, int B, int n_head, int T, float scale,
                                  cudaStream_t stream)
{
      int total_heads = B * n_head;

      int num_q_tiles = (T + BWD_BR - 1) / BWD_BR;
      dim3 grid_dq(total_heads, num_q_tiles);
      dim3 block_dq(BWD_BR);
      flash_attn_bwd_dq_kernel<BWD_BC, D>
          <<<grid_dq, block_dq, 0, stream>>>(Q, K, V, O, dO, LSE, dQ, T, scale);

      int num_k_tiles = (T + BWD_BC - 1) / BWD_BC;
      dim3 grid_dkv(total_heads, num_k_tiles);
      dim3 block_dkv(BWD_BC);
      flash_attn_bwd_dkdv_kernel<BWD_BR, D>
          <<<grid_dkv, block_dkv, 0, stream>>>(Q, K, V, O, dO, LSE, dK, dV, T, scale);
}

// ------------------------------------------------------------------
struct DeviceLinear
{
      int in_features = 0;
      int out_features = 0;
      bool use_bias = false;

      float *weight = nullptr;
      float *bias = nullptr;
      float *d_weight = nullptr;
      float *d_bias = nullptr;

      DeviceLinear() = default;
      DeviceLinear(int in_, int out_, bool bias_, std::mt19937 &rng)
          : in_features(in_), out_features(out_), use_bias(bias_)
      {
            size_t w_sz = out_features * in_features;
            size_t b_sz = use_bias ? out_features : 0;
            cudaMalloc(&weight, w_sz * sizeof(float));
            cudaMalloc(&d_weight, w_sz * sizeof(float));
            if (use_bias)
            {
                  cudaMalloc(&bias, b_sz * sizeof(float));
                  cudaMalloc(&d_bias, b_sz * sizeof(float));
            }

            std::vector<float> h_w(w_sz);
            float limit = std::sqrt(6.0f / (in_features + out_features));
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (auto &w : h_w)
                  w = dist(rng);
            cudaMemcpy(weight, h_w.data(), w_sz * sizeof(float), cudaMemcpyHostToDevice);

            if (use_bias)
            {
                  std::vector<float> h_b(b_sz, 0.0f);
                  cudaMemcpy(bias, h_b.data(), b_sz * sizeof(float), cudaMemcpyHostToDevice);
            }
            zero_grad(nullptr);
      }

      ~DeviceLinear()
      {
            if (weight)
                  cudaFree(weight);
            if (d_weight)
                  cudaFree(d_weight);
            if (bias)
                  cudaFree(bias);
            if (d_bias)
                  cudaFree(d_bias);
      }

      int num_params() const { return out_features * in_features + (use_bias ? out_features : 0); }

      void zero_grad(cudaStream_t stream)
      {
            cudaMemsetAsync(d_weight, 0, out_features * in_features * sizeof(float), stream);
            if (use_bias && d_bias)
                  cudaMemsetAsync(d_bias, 0, out_features * sizeof(float), stream);
      }

      void forward(const float *x, float *out, int M, cudaStream_t stream)
      {
            dim3 block(16, 16);
            dim3 grid((out_features + 15) / 16, (M + 15) / 16);
            matmul_fwd_kernel<<<grid, block, 0, stream>>>(x, weight, bias, out, M, in_features,
                                                          out_features);
      }

      void backward(const float *x, const float *dout, float *dx, int M, cudaStream_t stream)
      {
            if (dx)
            {
                  dim3 block(16, 16);
                  dim3 grid((in_features + 15) / 16, (M + 15) / 16);
                  matmul_bwd_dA_kernel<<<grid, block, 0, stream>>>(dout, weight, dx, M, in_features,
                                                                   out_features);
            }
            {
                  dim3 block(16, 16);
                  dim3 grid((in_features + 15) / 16, (out_features + 15) / 16);
                  matmul_bwd_dW_kernel<<<grid, block, 0, stream>>>(x, dout, d_weight, M,
                                                                   in_features, out_features);
            }
            if (use_bias && d_bias)
            {
                  int blocks = (out_features + 255) / 256;
                  bias_bwd_kernel<<<blocks, 256, 0, stream>>>(dout, d_bias, M, out_features);
            }
      }

      void step(float lr, cudaStream_t stream)
      {
            int n = out_features * in_features;
            sgd_step_kernel<<<(n + 255) / 256, 256, 0, stream>>>(weight, d_weight, lr, n);
            if (use_bias && bias && d_bias)
            {
                  int nb = out_features;
                  sgd_step_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(bias, d_bias, lr, nb);
            }
      }

      void save(std::ofstream &f) const
      {
            int in_ = in_features, out_ = out_features;
            bool b_ = use_bias;
            f.write(reinterpret_cast<const char *>(&in_), sizeof(int));
            f.write(reinterpret_cast<const char *>(&out_), sizeof(int));
            f.write(reinterpret_cast<const char *>(&b_), sizeof(bool));

            std::vector<float> h_w(out_features * in_features);
            cudaMemcpy(h_w.data(), weight, h_w.size() * sizeof(float), cudaMemcpyDeviceToHost);
            f.write(reinterpret_cast<const char *>(h_w.data()), h_w.size() * sizeof(float));

            if (use_bias)
            {
                  std::vector<float> h_b(out_features);
                  cudaMemcpy(h_b.data(), bias, h_b.size() * sizeof(float), cudaMemcpyDeviceToHost);
                  f.write(reinterpret_cast<const char *>(h_b.data()), h_b.size() * sizeof(float));
            }
      }

      void load(std::ifstream &f)
      {
            int in_, out_;
            bool b_;
            f.read(reinterpret_cast<char *>(&in_), sizeof(int));
            f.read(reinterpret_cast<char *>(&out_), sizeof(int));
            f.read(reinterpret_cast<char *>(&b_), sizeof(bool));
            if (in_ != in_features || out_ != out_features || b_ != use_bias)
                  throw std::runtime_error("DeviceLinear load dimension mismatch");

            std::vector<float> h_w(out_features * in_features);
            f.read(reinterpret_cast<char *>(h_w.data()), h_w.size() * sizeof(float));
            cudaMemcpy(weight, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice);

            if (use_bias)
            {
                  std::vector<float> h_b(out_features);
                  f.read(reinterpret_cast<char *>(h_b.data()), h_b.size() * sizeof(float));
                  cudaMemcpy(bias, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice);
            }
      }
};
/*
 * Multi-Head Attention
 */
// Multi-Head Attention

struct MultiHeadAttentionCUDA
{
      int n_embd = 0;
      int n_head = 0;
      int head_dim = 0;
      float scale = 0.0f;
      float dropout_p = 0.0f;

      DeviceLinear q_proj;
      DeviceLinear k_proj;
      DeviceLinear v_proj;
      DeviceLinear o_proj;

      // Training buffers (allocated on demand)
      float *d_q = nullptr; // [B, T, n_embd]  raw linear output
      float *d_k = nullptr;
      float *d_v = nullptr;
      float *d_attn_out = nullptr;  // [B, T, n_embd]  before output proj
      float *d_lse = nullptr;       // [B, n_head, T]
      float *d_input = nullptr;     // [B, T, n_embd]  saved input
      float *d_drop_mask = nullptr; // [B, T, n_embd]

      int curr_B = 0, curr_T = 0;

      MultiHeadAttentionCUDA() = default;
      MultiHeadAttentionCUDA(int n_embd_, int n_head_, int head_dim_, float drop_,
                             std::mt19937 &rng)
          : n_embd(n_embd_), n_head(n_head_), head_dim(head_dim_), dropout_p(drop_)
      {
            if (n_embd % n_head != 0)
                  throw std::runtime_error("n_embd must be divisible by n_head");
            scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            q_proj = DeviceLinear(n_embd, n_embd, true, rng);
            k_proj = DeviceLinear(n_embd, n_embd, true, rng);
            v_proj = DeviceLinear(n_embd, n_embd, true, rng);
            o_proj = DeviceLinear(n_embd, n_embd, true, rng);
      }

      ~MultiHeadAttentionCUDA() { free_buffers(); }

      int num_params() const
      {
            return q_proj.num_params() + k_proj.num_params() + v_proj.num_params() +
                   o_proj.num_params();
      }

      void forward(const float *d_x, float *d_out, int B, int T, bool training,
                   cudaStream_t stream = 0)
      {
            allocate_buffers(B, T, training);
            const int BT = B * T;
            const int BHTD = B * n_head * T * head_dim;

            if (training)
                  cudaMemcpyAsync(d_input, d_x, BT * n_embd * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream);
            q_proj.forward(d_x, d_q, BT, stream);
            k_proj.forward(d_x, d_k, BT, stream);
            v_proj.forward(d_x, d_v, BT, stream);
            float *d_q_perm = nullptr, *d_k_perm = nullptr, *d_v_perm = nullptr;
            float *d_attn_perm = nullptr;
            cudaMalloc(&d_q_perm, BHTD * sizeof(float));
            cudaMalloc(&d_k_perm, BHTD * sizeof(float));
            cudaMalloc(&d_v_perm, BHTD * sizeof(float));
            cudaMalloc(&d_attn_perm, BHTD * sizeof(float));

            int threads = 256;
            int blocks = (BHTD + 255) / 256;
            reshape_to_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_q, d_q_perm, B, T, n_head,
                                                                   head_dim);
            reshape_to_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_k, d_k_perm, B, T, n_head,
                                                                   head_dim);
            reshape_to_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_v, d_v_perm, B, T, n_head,
                                                                   head_dim);
            if (head_dim == 64)
            {
                  launch_flash_attn_fwd<64>(d_q_perm, d_k_perm, d_v_perm, d_attn_perm, d_lse, B,
                                            n_head, T, scale, stream);
            }
            else
            {
                  throw std::runtime_error(
                      "Unsupported head_dim in forward. Add template instantiation.");
            }

            reshape_from_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_attn_perm, d_attn_out, B, T,
                                                                     n_head, head_dim);
            o_proj.forward(d_attn_out, d_out, BT, stream);
            if (training && dropout_p > 0.0f)
            {
                  int n = BT * n_embd;
                  dropout_fwd_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_out, d_out, d_drop_mask,
                                                                          n, dropout_p, 1234u);
            }

            cudaFree(d_q_perm);
            cudaFree(d_k_perm);
            cudaFree(d_v_perm);
            cudaFree(d_attn_perm);
      }

      void backward(const float *d_dout, float *d_dx, int B, int T, cudaStream_t stream = 0)
      {
            const int BT = B * T;
            const int BHTD = B * n_head * T * head_dim;
            const int threads = 256;
            const int blocks = (BHTD + 255) / 256;
            float *d_dout_clean = nullptr;
            if (dropout_p > 0.0f && d_drop_mask)
            {
                  cudaMalloc(&d_dout_clean, BT * n_embd * sizeof(float));
                  dropout_bwd_kernel<<<(BT * n_embd + 255) / 256, 256, 0, stream>>>(
                      d_dout, d_drop_mask, d_dout_clean, BT * n_embd);
            }
            else
            {
                  d_dout_clean = const_cast<float *>(d_dout);
            }

            float *d_dattn_raw = nullptr;
            cudaMalloc(&d_dattn_raw, BT * n_embd * sizeof(float));
            o_proj.backward(d_attn_out, d_dout_clean, d_dattn_raw, BT, stream);
            float *d_dattn_perm = nullptr;
            cudaMalloc(&d_dattn_perm, BHTD * sizeof(float));
            reshape_to_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_dattn_raw, d_dattn_perm, B, T,
                                                                   n_head, head_dim);
            float *d_q_perm = nullptr, *d_k_perm = nullptr, *d_v_perm = nullptr,
                  *d_attn_perm = nullptr;
            cudaMalloc(&d_q_perm, BHTD * sizeof(float));
            cudaMalloc(&d_k_perm, BHTD * sizeof(float));
            cudaMalloc(&d_v_perm, BHTD * sizeof(float));
            cudaMalloc(&d_attn_perm, BHTD * sizeof(float));

            reshape_to_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_q, d_q_perm, B, T, n_head,
                                                                   head_dim);
            reshape_to_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_k, d_k_perm, B, T, n_head,
                                                                   head_dim);
            reshape_to_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_v, d_v_perm, B, T, n_head,
                                                                   head_dim);
            reshape_to_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_attn_out, d_attn_perm, B, T,
                                                                   n_head, head_dim);

            // backward
            float *d_dq_perm = nullptr, *d_dk_perm = nullptr, *d_dv_perm = nullptr;
            cudaMalloc(&d_dq_perm, BHTD * sizeof(float));
            cudaMalloc(&d_dk_perm, BHTD * sizeof(float));
            cudaMalloc(&d_dv_perm, BHTD * sizeof(float));
            cudaMemsetAsync(d_dq_perm, 0, BHTD * sizeof(float), stream);
            cudaMemsetAsync(d_dk_perm, 0, BHTD * sizeof(float), stream);
            cudaMemsetAsync(d_dv_perm, 0, BHTD * sizeof(float), stream);
            if (head_dim == 64)
            {
                  launch_flash_attn_bwd<64>(d_q_perm, d_k_perm, d_v_perm, d_attn_perm, d_dattn_perm,
                                            d_lse, d_dq_perm, d_dk_perm, d_dv_perm, B, n_head, T,
                                            scale, stream);
            }
            else
            {
                  throw std::runtime_error(
                      "Unsupported head_dim in backward. Add template instantiation.");
            }

            float *d_dq_raw = nullptr, *d_dk_raw = nullptr, *d_dv_raw = nullptr;
            cudaMalloc(&d_dq_raw, BHTD * sizeof(float));
            cudaMalloc(&d_dk_raw, BHTD * sizeof(float));
            cudaMalloc(&d_dv_raw, BHTD * sizeof(float));

            reshape_from_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_dq_perm, d_dq_raw, B, T,
                                                                     n_head, head_dim);
            reshape_from_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_dk_perm, d_dk_raw, B, T,
                                                                     n_head, head_dim);
            reshape_from_bhtd_kernel<<<blocks, threads, 0, stream>>>(d_dv_perm, d_dv_raw, B, T,
                                                                     n_head, head_dim);
            float *d_dx_q = nullptr;
            float *d_dx_k = nullptr;
            float *d_dx_v = nullptr;
            cudaMalloc(&d_dx_q, BT * n_embd * sizeof(float));
            cudaMalloc(&d_dx_k, BT * n_embd * sizeof(float));
            cudaMalloc(&d_dx_v, BT * n_embd * sizeof(float));
            q_proj.backward(d_input, d_dq_raw, d_dx_q, BT, stream);
            k_proj.backward(d_input, d_dk_raw, d_dx_k, BT, stream);
            v_proj.backward(d_input, d_dv_raw, d_dx_v, BT, stream);
            int n = BT * n_embd;
            elemwise_add_3in1_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_dx_q, d_dx_k, d_dx_v,
                                                                          d_dx, n);
            cudaFree(d_dout_clean);
            cudaFree(d_dattn_raw);
            cudaFree(d_dattn_perm);
            cudaFree(d_q_perm);
            cudaFree(d_k_perm);
            cudaFree(d_v_perm);
            cudaFree(d_attn_perm);
            cudaFree(d_dq_perm);
            cudaFree(d_dk_perm);
            cudaFree(d_dv_perm);
            cudaFree(d_dq_raw);
            cudaFree(d_dk_raw);
            cudaFree(d_dv_raw);
            cudaFree(d_dx_q);
            cudaFree(d_dx_k);
            cudaFree(d_dx_v);
      }

      void zero_grad(cudaStream_t stream = 0)
      {
            q_proj.zero_grad(stream);
            k_proj.zero_grad(stream);
            v_proj.zero_grad(stream);
            o_proj.zero_grad(stream);
      }

      void step(float lr, cudaStream_t stream = 0)
      {
            q_proj.step(lr, stream);
            k_proj.step(lr, stream);
            v_proj.step(lr, stream);
            o_proj.step(lr, stream);
      }

      void save(std::ofstream &f) const
      {
            f.write(reinterpret_cast<const char *>(&n_embd), sizeof(int));
            f.write(reinterpret_cast<const char *>(&n_head), sizeof(int));
            f.write(reinterpret_cast<const char *>(&head_dim), sizeof(int));
            f.write(reinterpret_cast<const char *>(&dropout_p), sizeof(float));
            q_proj.save(f);
            k_proj.save(f);
            v_proj.save(f);
            o_proj.save(f);
      }

      void load(std::ifstream &f)
      {
            int ne, nh, hd;
            float dp;
            f.read(reinterpret_cast<char *>(&ne), sizeof(int));
            f.read(reinterpret_cast<char *>(&nh), sizeof(int));
            f.read(reinterpret_cast<char *>(&hd), sizeof(int));
            f.read(reinterpret_cast<char *>(&dp), sizeof(float));
            if (ne != n_embd || nh != n_head || hd != head_dim || dp != dropout_p)
                  throw std::runtime_error("MultiHeadAttentionCUDA load mismatch");
            q_proj.load(f);
            k_proj.load(f);
            v_proj.load(f);
            o_proj.load(f);
      }

    private:
      void allocate_buffers(int B, int T, bool training)
      {
            if (B == curr_B && T == curr_T)
                  return;
            free_buffers();
            curr_B = B;
            curr_T = T;

            const int BT = B * T;
            const int BHTD = B * n_head * T * head_dim;

            cudaMalloc(&d_q, BHTD * sizeof(float));
            cudaMalloc(&d_k, BHTD * sizeof(float));
            cudaMalloc(&d_v, BHTD * sizeof(float));
            cudaMalloc(&d_attn_out, BHTD * sizeof(float));

            if (training)
            {
                  cudaMalloc(&d_lse, B * n_head * T * sizeof(float));
                  cudaMalloc(&d_input, BT * n_embd * sizeof(float));
                  if (dropout_p > 0.0f)
                        cudaMalloc(&d_drop_mask, BT * n_embd * sizeof(float));
            }
      }

      void free_buffers()
      {
            auto free_if = [](float *&p)
            {
                  if (p)
                  {
                        cudaFree(p);
                        p = nullptr;
                  }
            };
            free_if(d_q);
            free_if(d_k);
            free_if(d_v);
            free_if(d_attn_out);
            free_if(d_lse);
            free_if(d_input);
            free_if(d_drop_mask);
            curr_B = 0;
            curr_T = 0;
      }
};

} // namespace fa