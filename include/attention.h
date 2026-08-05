#pragma once

#include "config/config.h"
#include "linear.h"
#include "tensor.h"

#include <fstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Head
{
      int head_size;
      Linear key, query, value;

      Head() = default;

      Head(int n_embd, int hs, std::mt19937 &rng)
            : head_size(hs), key(n_embd, hs, false, rng), query(n_embd, hs, false, rng),
              value(n_embd, hs, false, rng)
      {
      }

      Tensor forward(const Tensor &x, bool training, std::mt19937 &rng) const
      {
            int B = x.shape[0], T = x.shape[1];

            Tensor k = key.forward(x);   // [B, T, hs]
            Tensor q = query.forward(x); // [B, T, hs]
            Tensor v = value.forward(x); // [B, T, hs]

            // scaled dot-product  wei = q @ k^T * scale
            float scale = 1.0f / std::sqrt((float)head_size);
            Tensor kt = transpose23(k); // [B, hs, T]
            Tensor wei = bmm(q, kt);    // [B, T, T]

            // Parallelized causal mask application for all CPU cores
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

            // scale then softmax (using move semantics)
            wei = scale3d_inplace(std::move(wei), scale);
            wei = softmax3d(wei);

            // dropout on attention weights
            wei = dropout(wei, DROPOUT, training, rng);
            return bmm(wei, v);
      }

      int num_params() const
      {
            return key.num_params() + query.num_params() + value.num_params();
      }

      void save(std::ofstream &f) const
      {
            key.save(f);
            query.save(f);
            value.save(f);
      }
      void load(std::ifstream &f)
      {
            key.load(f);
            query.load(f);
            value.load(f);
      }

    private:
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

// Multi-head causal attention
struct MultiHeadAttention
{
      int num_heads, head_size, n_embd;
      std::vector<Head> heads;
      Linear proj;

      MultiHeadAttention() = default;

      MultiHeadAttention(int n_embd_, int num_h, int hs, std::mt19937 &rng)
            : num_heads(num_h), head_size(hs), n_embd(n_embd_), proj(num_h * hs, n_embd_, true, rng)
      {
            for (int i = 0; i < num_h; ++i)
                  heads.emplace_back(n_embd_, hs, rng);
      }

      Tensor forward(const Tensor &x, bool training, std::mt19937 &rng) const
      {
            std::vector<Tensor> head_outs(num_heads);

            // Sequential across heads to keep std::mt19937 thread-safe,
            // while underlying tensor operations utilize OpenMP multi-core execution.
            for (int i = 0; i < num_heads; ++i)
            {
                  head_outs[i] = heads[i].forward(x, training, rng);
            }

            Tensor concat = cat_last(head_outs); // [B, T, num_heads*hs]
            Tensor out = proj.forward(concat);   // [B, T, n_embd]
            return dropout(out, DROPOUT, training, rng);
      }

      int num_params() const
      {
            int n = proj.num_params();
            for (auto &h : heads)
                  n += h.num_params();
            return n;
      }

      void save(std::ofstream &f) const
      {
            for (auto &h : heads)
                  h.save(f);
            proj.save(f);
      }
      void load(std::ifstream &f)
      {
            for (auto &h : heads)
                  h.load(f);
            proj.load(f);
      }
};