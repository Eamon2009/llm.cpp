/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2026 Eamon Sippy
 */

#pragma once

#include "config/config.h"
#include "linear.h"
#include "tensor.h"

#include <fstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Single-head causal scaled dot-product self-attention
 *
 * Causal mask: upper-triangle set to -1e30. Not thread-safe across forward calls.
 */
struct Head
{
      int head_size;
      Linear key, query, value;

      /**
       * @brief Default constructor. Leaves head uninitialized.
       */
      Head() = default;

      /**
       * @brief Construct and initialize Q, K, V projection weights.
       *
       * @param n_embd Input embedding dimension.
       * @param hs     Head dimension (head_size).
       * @param rng    Seeded MT19937 for weight init.
       */
      Head(int n_embd, int hs, std::mt19937 &rng)
          : head_size(hs), key(n_embd, hs, false, rng), query(n_embd, hs, false, rng),
            value(n_embd, hs, false, rng)
      {
      }

      /**
       * @brief Forward pass.
       *
       * @param x        [B, T, n_embd].
       * @param training Enables dropout when true.
       * @param rng      Thread-local MT19937 for stochastic ops.
       * @return         [B, T, head_size]. New allocation.
       */
      Tensor forward(const Tensor &x, bool training, std::mt19937 &rng) const
      {
            int B = x.shape[0], T = x.shape[1];

            Tensor k = key.forward(x);   // [B, T, head_size]
            Tensor q = query.forward(x); // [B, T, head_size]
            Tensor v = value.forward(x); // [B, T, head_size]

            float scale = 1.0f / std::sqrt((float)head_size);
            Tensor kt = transpose23(k); // [B, head_size, T]
            Tensor wei = bmm(q, kt);    // [B, T, T]

            // Causal mask: upper-triangle to -inf
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (B * T * T > 10000)
#endif
            for (int b = 0; b < B; ++b)
            {
                  for (int i = 0; i < T; ++i)
                  {
                        for (int j = i + 1; j < T; ++j)
                        {
                              wei.at(b, i, j) = -1e30f;
                        }
                  }
            }

            // In-place scale to avoid extra allocation
            wei = scale3d_inplace(std::move(wei), scale);
            wei = softmax3d(wei);
            wei = dropout(wei, DROPOUT, training, rng);
            return bmm(wei, v); // [B, T, head_size]
      }

      /**
       * @brief Total trainable parameters.
       * @return Sum of key, query, and value parameter counts.
       */
      int num_params() const { return key.num_params() + query.num_params() + value.num_params(); }

      /**
       * @brief Serialize weights to binary stream.
       * @param f Open binary output stream.
       */
      void save(std::ofstream &f) const
      {
            key.save(f);
            query.save(f);
            value.save(f);
      }

      /**
       * @brief Deserialize weights from binary stream.
       * @param f Open binary input stream.
       */
      void load(std::ifstream &f)
      {
            key.load(f);
            query.load(f);
            value.load(f);
      }

    private:
      /**
       * @brief In-place scalar multiply via move semantics.
       *
       * @param t Tensor to scale. Consumed by move.
       * @param s Scale factor.
       * @return  Scaled tensor. Same shape as input.
       */
      static Tensor scale3d_inplace(Tensor &&t, float s)
      {
#ifdef _OPENMP
#pragma omp parallel for if (t.data.size() > 10000)
#endif
            for (size_t i = 0; i < t.data.size(); ++i)
            {
                  t.data[i] *= s;
            }
            return std::move(t);
      }
};

/**
 * @brief Multi-Head Attention (MHA) module.
 *
 * Heads run sequentially for PRNG safety. Underlying matmul uses OpenMP.
 * Throws if n_embd % n_head != 0 (enforced in Block constructor).
 */
struct MultiHeadAttention
{
      int num_heads, head_size, n_embd;
      std::vector<Head> heads;
      Linear proj; // output projection [n_head * hs, n_embd]

      /**
       * @brief Default constructor. Leaves MHA uninitialized.
       */
      MultiHeadAttention() = default;

      /**
       * @brief Construct heads and output projection.
       *
       * @param n_embd_ Model embedding dimension.
       * @param num_h   Number of parallel attention heads.
       * @param hs      Head dimension. n_embd_ must be divisible by num_h.
       * @param rng     Seeded MT19937 for weight init.
       */
      MultiHeadAttention(int n_embd_, int num_h, int hs, std::mt19937 &rng)
          : num_heads(num_h), head_size(hs), n_embd(n_embd_), proj(num_h * hs, n_embd_, true, rng)
      {
            for (int i = 0; i < num_h; ++i)
                  heads.emplace_back(n_embd_, hs, rng);
      }

      /**
       * @brief Forward pass.
       *
       * @param x        [B, T, n_embd].
       * @param training Enables dropout when true.
       * @param rng      Thread-local MT19937 for stochastic ops.
       * @return         [B, T, n_embd]. New allocation.
       */
      Tensor forward(const Tensor &x, bool training, std::mt19937 &rng) const
      {
            std::vector<Tensor> head_outs(num_heads);

            // Sequential head execution for PRNG safety
            for (int i = 0; i < num_heads; ++i)
            {
                  head_outs[i] = heads[i].forward(x, training, rng);
            }

            Tensor concat = cat_last(head_outs); // [B, T, num_heads * head_size]
            Tensor out = proj.forward(concat);   // [B, T, n_embd]
            return dropout(out, DROPOUT, training, rng);
      }

      /**
       * @brief Total trainable parameters.
       * @return Sum of all head and projection parameter counts.
       */
      int num_params() const
      {
            int n = proj.num_params();
            for (auto &h : heads)
                  n += h.num_params();
            return n;
      }

      /**
       * @brief Serialize weights to binary stream.
       * @param f Open binary output stream.
       */
      void save(std::ofstream &f) const
      {
            for (auto &h : heads)
                  h.save(f);
            proj.save(f);
      }

      /**
       * @brief Deserialize weights from binary stream.
       * @param f Open binary input stream.
       */
      void load(std::ifstream &f)
      {
            for (auto &h : heads)
                  h.load(f);
            proj.load(f);
      }
};