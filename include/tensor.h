/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2026 Eamon Sippy
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif

/**
 * @brief Dense multi-dimensional float tensor.
 *
 * Row-major layout. Supports 1D, 2D, and 3D indexing via at().
 * AVX2 / SSE / OpenMP vectorization paths available at compile time.
 */
struct Tensor
{
      std::vector<int> shape;
      std::vector<float> data;

      /**
       * @brief Default constructor. Empty tensor.
       */
      Tensor() = default;

      /**
       * @brief Allocate tensor with fill value.
       *
       * @param sh   Shape vector.
       * @param fill Initial value for all elements.
       */
      Tensor(std::vector<int> sh, float fill = 0.0f) : shape(std::move(sh))
      {
            int total = 1;
            for (int d : shape)
                  total *= d;
            data.reserve(total);
            data.assign(total, fill);
      }

      Tensor(const Tensor &) = default;
      Tensor(Tensor &&) noexcept = default;
      Tensor &operator=(const Tensor &) = default;
      Tensor &operator=(Tensor &&) noexcept = default;

      /**
       * @brief Total element count.
       * @return Product of all shape dimensions.
       */
      int numel() const
      {
            int n = 1;
            for (int d : shape)
                  n *= d;
            return n;
      }

      /**
       * @brief Number of dimensions.
       * @return shape.size().
       */
      int ndim() const { return (int)shape.size(); }

      float &at(int i) { return data[i]; }
      float at(int i) const { return data[i]; }

      float &at(int r, int c) { return data[r * shape[1] + c]; }
      float at(int r, int c) const { return data[r * shape[1] + c]; }

      float &at(int b, int r, int c) { return data[b * shape[1] * shape[2] + r * shape[2] + c]; }
      float at(int b, int r, int c) const
      {
            return data[b * shape[1] * shape[2] + r * shape[2] + c];
      }

      /**
       * @brief Allocate zero-filled tensor.
       *
       * @param sh Shape vector.
       * @return   Zero-filled Tensor.
       */
      static Tensor zeros(std::vector<int> sh) { return Tensor(sh, 0.0f); }

      /**
       * @brief Allocate one-filled tensor.
       *
       * @param sh Shape vector.
       * @return   One-filled Tensor.
       */
      static Tensor ones(std::vector<int> sh) { return Tensor(sh, 1.0f); }

      /**
       * @brief Allocate tensor with normal-distributed values.
       *
       * @param sh    Shape vector.
       * @param mean  Distribution mean.
       * @param std   Distribution standard deviation.
       * @param rng   Seeded MT19937.
       * @return      Normally-distributed Tensor.
       */
      static Tensor randn(std::vector<int> sh, float mean, float std, std::mt19937 &rng)
      {
            std::normal_distribution<float> dist(mean, std);
            Tensor t(sh);
            for (auto &v : t.data)
                  v = dist(rng);
            return t;
      }

      /**
       * @brief Fill all elements with value.
       *
       * @param v Fill value.
       */
      void fill(float v) { std::fill(data.begin(), data.end(), v); }

      /**
       * @brief Print shape to stdout.
       *
       * @param name Optional label prefix.
       */
      void print_shape(const std::string &name = "") const
      {
            if (!name.empty())
                  std::cout << name << ": ";
            std::cout << "[";
            for (int i = 0; i < (int)shape.size(); ++i)
            {
                  std::cout << shape[i];
                  if (i + 1 < (int)shape.size())
                        std::cout << ", ";
            }
            std::cout << "]" << std::endl;
      }
};

/**
 * @brief Element-wise addition.
 *
 * @param a LHS tensor.
 * @param b RHS tensor. Same shape as a.
 * @return  Element-wise sum. New allocation.
 */
inline Tensor add(const Tensor &a, const Tensor &b)
{
      assert(a.data.size() == b.data.size());
      Tensor c(a.shape);
      size_t n = a.data.size();

#ifdef __AVX__
      size_t vec_end = n & ~7ULL;
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 8)
      {
            __m256 va = _mm256_loadu_ps(&a.data[i]);
            __m256 vb = _mm256_loadu_ps(&b.data[i]);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&c.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            c.data[i] = a.data[i] + b.data[i];
#elif defined(__SSE__)
      size_t vec_end = n & ~3ULL;
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 4)
      {
            __m128 va = _mm_loadu_ps(&a.data[i]);
            __m128 vb = _mm_loadu_ps(&b.data[i]);
            __m128 vc = _mm_add_ps(va, vb);
            _mm_storeu_ps(&c.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            c.data[i] = a.data[i] + b.data[i];
#else
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < n; ++i)
            c.data[i] = a.data[i] + b.data[i];
#endif
      return c;
}

/**
 * @brief In-place element-wise addition.
 *
 * @param a Destination tensor. Modified in-place.
 * @param b RHS tensor. Same shape as a.
 */
inline void add_inplace(Tensor &a, const Tensor &b)
{
      assert(a.data.size() == b.data.size());
      size_t n = a.data.size();

#ifdef __AVX__
      size_t vec_end = n & ~7ULL;
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 8)
      {
            __m256 va = _mm256_loadu_ps(&a.data[i]);
            __m256 vb = _mm256_loadu_ps(&b.data[i]);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&a.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            a.data[i] += b.data[i];
#elif defined(__SSE__)
      size_t vec_end = n & ~3ULL;
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 4)
      {
            __m128 va = _mm_loadu_ps(&a.data[i]);
            __m128 vb = _mm_loadu_ps(&b.data[i]);
            __m128 vc = _mm_add_ps(va, vb);
            _mm_storeu_ps(&a.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            a.data[i] += b.data[i];
#else
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < n; ++i)
            a.data[i] += b.data[i];
#endif
}

/**
 * @brief Element-wise scalar multiplication.
 *
 * @param a Input tensor.
 * @param s Scalar multiplier.
 * @return  Scaled tensor. New allocation.
 */
inline Tensor scale(const Tensor &a, float s)
{
      Tensor c(a.shape);
      size_t n = a.data.size();

#ifdef __AVX__
      size_t vec_end = n & ~7ULL;
      __m256 vs = _mm256_set1_ps(s);
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 8)
      {
            __m256 va = _mm256_loadu_ps(&a.data[i]);
            __m256 vc = _mm256_mul_ps(va, vs);
            _mm256_storeu_ps(&c.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            c.data[i] = a.data[i] * s;
#elif defined(__SSE__)
      size_t vec_end = n & ~3ULL;
      __m128 vs = _mm_set1_ps(s);
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 4)
      {
            __m128 va = _mm_loadu_ps(&a.data[i]);
            __m128 vc = _mm_mul_ps(va, vs);
            _mm_storeu_ps(&c.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            c.data[i] = a.data[i] * s;
#else
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < n; ++i)
            c.data[i] = a.data[i] * s;
#endif
      return c;
}

/**
 * @brief In-place element-wise scalar multiplication.
 *
 * @param a Destination tensor. Modified in-place.
 * @param s Scalar multiplier.
 */
inline void scale_inplace(Tensor &a, float s)
{
      size_t n = a.data.size();

#ifdef __AVX__
      size_t vec_end = n & ~7ULL;
      __m256 vs = _mm256_set1_ps(s);
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 8)
      {
            __m256 va = _mm256_loadu_ps(&a.data[i]);
            __m256 vc = _mm256_mul_ps(va, vs);
            _mm256_storeu_ps(&a.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            a.data[i] *= s;
#elif defined(__SSE__)
      size_t vec_end = n & ~3ULL;
      __m128 vs = _mm_set1_ps(s);
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 4)
      {
            __m128 va = _mm_loadu_ps(&a.data[i]);
            __m128 vc = _mm_mul_ps(va, vs);
            _mm_storeu_ps(&a.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            a.data[i] *= s;
#else
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < n; ++i)
            a.data[i] *= s;
#endif
}

/**
 * @brief Element-wise ReLU.
 *
 * @param a Input tensor.
 * @return  ReLU-activated tensor. New allocation.
 */
inline Tensor relu(const Tensor &a)
{
      Tensor c(a.shape);
      size_t n = a.data.size();

#ifdef __AVX__
      size_t vec_end = n & ~7ULL;
      __m256 zero = _mm256_setzero_ps();
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 8)
      {
            __m256 va = _mm256_loadu_ps(&a.data[i]);
            __m256 vc = _mm256_max_ps(va, zero);
            _mm256_storeu_ps(&c.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            c.data[i] = std::max(0.0f, a.data[i]);
#elif defined(__SSE__)
      size_t vec_end = n & ~3ULL;
      __m128 zero = _mm_setzero_ps();
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 4)
      {
            __m128 va = _mm_loadu_ps(&a.data[i]);
            __m128 vc = _mm_max_ps(va, zero);
            _mm_storeu_ps(&c.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            c.data[i] = std::max(0.0f, a.data[i]);
#else
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < n; ++i)
            c.data[i] = std::max(0.0f, a.data[i]);
#endif
      return c;
}

/**
 * @brief In-place element-wise ReLU.
 *
 * @param a Destination tensor. Modified in-place.
 */
inline void relu_inplace(Tensor &a)
{
      size_t n = a.data.size();

#ifdef __AVX__
      size_t vec_end = n & ~7ULL;
      __m256 zero = _mm256_setzero_ps();
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 8)
      {
            __m256 va = _mm256_loadu_ps(&a.data[i]);
            __m256 vc = _mm256_max_ps(va, zero);
            _mm256_storeu_ps(&a.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            a.data[i] = std::max(0.0f, a.data[i]);
#elif defined(__SSE__)
      size_t vec_end = n & ~3ULL;
      __m128 zero = _mm_setzero_ps();
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < vec_end; i += 4)
      {
            __m128 va = _mm_loadu_ps(&a.data[i]);
            __m128 vc = _mm_max_ps(va, zero);
            _mm_storeu_ps(&a.data[i], vc);
      }
      for (size_t i = vec_end; i < n; ++i)
            a.data[i] = std::max(0.0f, a.data[i]);
#else
#ifdef _OPENMP
#pragma omp parallel for if (n > 10000)
#endif
      for (size_t i = 0; i < n; ++i)
            a.data[i] = std::max(0.0f, a.data[i]);
#endif
}

/**
 * @brief 3D softmax over last dimension.
 *
 * @param a [B, T, C].
 * @return  [B, T, C]. Softmax probabilities. New allocation.
 */
inline Tensor softmax3d(const Tensor &a)
{
      int B = a.shape[0], T = a.shape[1], C = a.shape[2];
      Tensor out(a.shape);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (B * T > 64)
#endif
      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  float maxv = -1e30f;
                  for (int c = 0; c < C; ++c)
                        maxv = std::max(maxv, a.at(b, t, c));

                  float sumv = 0.0f;
                  for (int c = 0; c < C; ++c)
                  {
                        float e = std::exp(a.at(b, t, c) - maxv);
                        out.at(b, t, c) = e;
                        sumv += e;
                  }

                  float inv_sum = 1.0f / sumv;
                  for (int c = 0; c < C; ++c)
                        out.at(b, t, c) *= inv_sum;
            }
      }
      return out;
}

/**
 * @brief 2D softmax over last dimension.
 *
 * @param a [T, C].
 * @return  [T, C]. Softmax probabilities. New allocation.
 */
inline Tensor softmax2d(const Tensor &a)
{
      int T = a.shape[0], C = a.shape[1];
      Tensor out(a.shape);

#ifdef _OPENMP
#pragma omp parallel for if (T > 128)
#endif
      for (int t = 0; t < T; ++t)
      {
            float maxv = -1e30f;
            for (int c = 0; c < C; ++c)
                  maxv = std::max(maxv, a.at(t, c));

            float sumv = 0.0f;
            for (int c = 0; c < C; ++c)
            {
                  float e = std::exp(a.at(t, c) - maxv);
                  out.at(t, c) = e;
                  sumv += e;
            }

            float inv_sum = 1.0f / sumv;
            for (int c = 0; c < C; ++c)
                  out.at(t, c) *= inv_sum;
      }
      return out;
}

/**
 * @brief Layer normalization.
 *
 * @param x     [B, T, C].
 * @param gamma [C]. Scale.
 * @param beta  [C]. Shift.
 * @param eps   Epsilon for numerical stability.
 * @return      [B, T, C]. Normalized output. New allocation.
 */
inline Tensor layer_norm(const Tensor &x, const Tensor &gamma, const Tensor &beta,
                         float eps = 1e-5f)
{
      int B = x.shape[0], T = x.shape[1], C = x.shape[2];
      Tensor out(x.shape);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (B * T > 64)
#endif
      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  float mu = 0.0f;
                  for (int c = 0; c < C; ++c)
                        mu += x.at(b, t, c);
                  mu /= C;

                  float var = 0.0f;
                  for (int c = 0; c < C; ++c)
                  {
                        float d = x.at(b, t, c) - mu;
                        var += d * d;
                  }
                  var /= C;

                  float inv = 1.0f / std::sqrt(var + eps);
                  for (int c = 0; c < C; ++c)
                        out.at(b, t, c) = (x.at(b, t, c) - mu) * inv * gamma.at(c) + beta.at(c);
            }
      }
      return out;
}

/**
 * @brief Batched matrix multiplication (3D @ 2D).
 *
 * @param a [B, T, D].
 * @param w [D, E].
 * @return  [B, T, E]. New allocation.
 */
inline Tensor matmul(const Tensor &a, const Tensor &w)
{
      assert(a.ndim() == 3 && w.ndim() == 2);
      int B = a.shape[0], T = a.shape[1], D = a.shape[2];
      int E = w.shape[1];
      assert(w.shape[0] == D);

      Tensor out({B, T, E}, 0.0f);

      const int TILE_T = 32;
      const int TILE_E = 32;
      const int TILE_D = 32;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (B * T * E * D > 100000)
#endif
      for (int b = 0; b < B; ++b)
      {
            for (int t0 = 0; t0 < T; t0 += TILE_T)
            {
                  int t_end = std::min(t0 + TILE_T, T);
                  for (int e0 = 0; e0 < E; e0 += TILE_E)
                  {
                        int e_end = std::min(e0 + TILE_E, E);
                        for (int d0 = 0; d0 < D; d0 += TILE_D)
                        {
                              int d_end = std::min(d0 + TILE_D, D);
                              for (int t = t0; t < t_end; ++t)
                              {
                                    for (int e = e0; e < e_end; ++e)
                                    {
                                          float s = out.at(b, t, e);
                                          for (int d = d0; d < d_end; ++d)
                                                s += a.at(b, t, d) * w.at(d, e);
                                          out.at(b, t, e) = s;
                                    }
                              }
                        }
                  }
            }
      }
      return out;
}

/**
 * @brief Add bias vector to last dimension.
 *
 * @param x    [..., E].
 * @param bias [E].
 * @return     Bias-added tensor. New allocation.
 */
inline Tensor add_bias(const Tensor &x, const Tensor &bias)
{
      assert(x.shape.back() == bias.shape[0]);
      Tensor out = x;
      int E = bias.shape[0];
      int stride = E;
      int n = x.numel() / E;

#ifdef _OPENMP
#pragma omp parallel for if (n * E > 10000)
#endif
      for (int i = 0; i < n; ++i)
      {
            for (int e = 0; e < E; ++e)
                  out.data[i * stride + e] += bias.data[e];
      }
      return out;
}

/**
 * @brief Batched matrix multiplication (3D @ 3D).
 *
 * @param a [B, T, D].
 * @param b [B, D, T2].
 * @return  [B, T, T2]. New allocation.
 */
inline Tensor bmm(const Tensor &a, const Tensor &b)
{
      assert(a.ndim() == 3 && b.ndim() == 3);
      int B = a.shape[0], T = a.shape[1], D = a.shape[2];
      int T2 = b.shape[2];
      assert(b.shape[0] == B && b.shape[1] == D);

      Tensor out({B, T, T2}, 0.0f);

      const int TILE = 32;

#ifdef _OPENMP
#pragma omp parallel for if (B * T * T2 * D > 100000)
#endif
      for (int bb = 0; bb < B; ++bb)
      {
            for (int t0 = 0; t0 < T; t0 += TILE)
            {
                  int t_end = std::min(t0 + TILE, T);
                  for (int t2_0 = 0; t2_0 < T2; t2_0 += TILE)
                  {
                        int t2_end = std::min(t2_0 + TILE, T2);
                        for (int d0 = 0; d0 < D; d0 += TILE)
                        {
                              int d_end = std::min(d0 + TILE, D);
                              for (int t = t0; t < t_end; ++t)
                              {
                                    for (int t2 = t2_0; t2 < t2_end; ++t2)
                                    {
                                          float s = out.at(bb, t, t2);
                                          for (int d = d0; d < d_end; ++d)
                                                s += a.at(bb, t, d) * b.at(bb, d, t2);
                                          out.at(bb, t, t2) = s;
                                    }
                              }
                        }
                  }
            }
      }
      return out;
}

/**
 * @brief Transpose last two dimensions of 3D tensor.
 *
 * @param a [B, T, D].
 * @return  [B, D, T]. New allocation.
 */
inline Tensor transpose23(const Tensor &a)
{
      int B = a.shape[0], T = a.shape[1], D = a.shape[2];
      Tensor out({B, D, T});

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (B * T * D > 10000)
#endif
      for (int b = 0; b < B; ++b)
      {
            for (int d = 0; d < D; ++d)
            {
                  for (int t = 0; t < T; ++t)
                        out.at(b, d, t) = a.at(b, t, d);
            }
      }
      return out;
}

/**
 * @brief Concatenate tensors along last dimension.
 *
 * @param ts Vector of [B, T, *] tensors. B and T must match across all.
 * @return   [B, T, sum(D_i)]. New allocation.
 */
inline Tensor cat_last(const std::vector<Tensor> &ts)
{
      int B = ts[0].shape[0], T = ts[0].shape[1];
      int total = 0;
      for (auto &t : ts)
            total += t.shape[2];

      Tensor out({B, T, total}, 0.0f);

      int offset = 0;
      for (auto &t : ts)
      {
            int D = t.shape[2];
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (B * T * D > 10000)
#endif
            for (int b = 0; b < B; ++b)
            {
                  for (int tt = 0; tt < T; ++tt)
                  {
                        for (int d = 0; d < D; ++d)
                              out.at(b, tt, offset + d) = t.at(b, tt, d);
                  }
            }
            offset += D;
      }
      return out;
}

/**
 * @brief Inverted dropout.
 *
 * @param x       Input tensor.
 * @param p       Dropout probability.
 * @param training No-op when false.
 * @param rng     Thread-local MT19937.
 * @return        Scaled tensor (1/(1-p) when kept, 0 when dropped). New allocation when training.
 */
inline Tensor dropout(const Tensor &x, float p, bool training, std::mt19937 &rng)
{
      if (!training || p == 0.0f)
            return x;

      std::bernoulli_distribution dist(1.0f - p);
      Tensor out = x;
      float scale_v = 1.0f / (1.0f - p);

      for (auto &v : out.data)
            v = dist(rng) ? v * scale_v : 0.0f;

      return out;
}