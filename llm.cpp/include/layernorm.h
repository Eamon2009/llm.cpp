#pragma once
// ============================================================
//  include/layernorm.h  –  Layer normalisation
//  Mirrors: nn.LayerNorm(n_embd)
// ============================================================

#include "tensor.h"
#include <fstream>

struct LayerNorm
{
      int n_embd;
      Tensor gamma; // scale  [n_embd]  (weight)
      Tensor beta;  // shift  [n_embd]  (bias)

      LayerNorm() = default;

      explicit LayerNorm(int embd)
          : n_embd(embd),
            gamma(Tensor::ones({embd})),
            beta(Tensor::zeros({embd})) {}

      // x: [B, T, n_embd]  →  [B, T, n_embd]
      Tensor forward(const Tensor &x) const
      {
            return layer_norm(x, gamma, beta);
      }

      int num_params() const { return gamma.numel() + beta.numel(); }

      void save(std::ofstream &f) const
      {
            f.write(reinterpret_cast<const char *>(gamma.data.data()),
                    gamma.numel() * sizeof(float));
            f.write(reinterpret_cast<const char *>(beta.data.data()),
                    beta.numel() * sizeof(float));
      }
      void load(std::ifstream &f)
      {
            f.read(reinterpret_cast<char *>(gamma.data.data()),
                   gamma.numel() * sizeof(float));
            f.read(reinterpret_cast<char *>(beta.data.data()),
                   beta.numel() * sizeof(float));
      }
};