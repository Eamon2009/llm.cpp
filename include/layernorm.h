/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2026 Eamon Sippy
 */

#pragma once

#include "tensor.h"

#include <fstream>

/**
 * @brief Layer normalization.
 *
 * Per-row: (x - mu) / sqrt(var + eps) * gamma + beta.
 */
struct LayerNorm
{
      int n_embd;
      Tensor gamma; // [n_embd]
      Tensor beta;  // [n_embd]

      /**
       * @brief Default constructor. Leaves layer uninitialized.
       */
      LayerNorm() = default;

      /**
       * @brief Construct with gamma=1, beta=0.
       *
       * @param embd Channel dimension.
       */
      explicit LayerNorm(int embd)
          : n_embd(embd), gamma(Tensor::ones({embd})), beta(Tensor::zeros({embd}))
      {
      }

      /**
       * @brief Forward pass.
       *
       * @param x [B, T, n_embd].
       * @return  [B, T, n_embd]. New allocation.
       */
      Tensor forward(const Tensor &x) const { return layer_norm(x, gamma, beta); }

      /**
       * @brief Total trainable parameters.
       * @return gamma.numel() + beta.numel().
       */
      int num_params() const { return gamma.numel() + beta.numel(); }

      /**
       * @brief Serialize weights to binary stream.
       * @param f Open binary output stream.
       */
      void save(std::ofstream &f) const
      {
            f.write(reinterpret_cast<const char *>(gamma.data.data()),
                    gamma.numel() * sizeof(float));
            f.write(reinterpret_cast<const char *>(beta.data.data()), beta.numel() * sizeof(float));
      }

      /**
       * @brief Deserialize weights from binary stream.
       * @param f Open binary input stream.
       */
      void load(std::ifstream &f)
      {
            f.read(reinterpret_cast<char *>(gamma.data.data()), gamma.numel() * sizeof(float));
            f.read(reinterpret_cast<char *>(beta.data.data()), beta.numel() * sizeof(float));
      }
};