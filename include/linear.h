/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2026 Eamon Sippy
 */

#pragma once

#include "tensor.h"

#include <fstream>

/**
 * @brief Dense linear projection (Y = X @ W + b).
 */
struct Linear
{
      int in_features, out_features;
      bool has_bias;
      Tensor weight; // [in_features, out_features]
      Tensor bias;   // [out_features] (empty when has_bias=false)

      /**
       * @brief Default constructor. Leaves layer uninitialized.
       */
      Linear() = default;

      /**
       * @brief Construct and initialize weights.
       *
       * @param in_f     Input feature dimension.
       * @param out_f    Output feature dimension.
       * @param use_bias True to allocate and use bias vector.
       * @param rng      Seeded MT19937 for weight init (N(0, 0.02)).
       */
      Linear(int in_f, int out_f, bool use_bias, std::mt19937 &rng)
          : in_features(in_f), out_features(out_f), has_bias(use_bias)
      {
            weight = Tensor::randn({in_f, out_f}, 0.0f, 0.02f, rng);
            if (has_bias)
                  bias = Tensor({out_f}, 0.0f);
      }

      /**
       * @brief Forward pass.
       *
       * @param x [B, T, in_features].
       * @return  [B, T, out_features]. New allocation.
       */
      Tensor forward(const Tensor &x) const
      {
            Tensor out = matmul(x, weight);
            if (has_bias)
                  out = add_bias(out, bias);
            return out;
      }

      /**
       * @brief Total trainable parameters.
       * @return weight.numel() + (has_bias ? bias.numel() : 0).
       */
      int num_params() const { return weight.numel() + (has_bias ? bias.numel() : 0); }

      /**
       * @brief Serialize weights to binary stream.
       * @param f Open binary output stream.
       */
      void save(std::ofstream &f) const
      {
            f.write(reinterpret_cast<const char *>(weight.data.data()),
                    weight.numel() * sizeof(float));
            if (has_bias)
                  f.write(reinterpret_cast<const char *>(bias.data.data()),
                          bias.numel() * sizeof(float));
      }

      /**
       * @brief Deserialize weights from binary stream.
       * @param f Open binary input stream.
       */
      void load(std::ifstream &f)
      {
            f.read(reinterpret_cast<char *>(weight.data.data()), weight.numel() * sizeof(float));
            if (has_bias)
                  f.read(reinterpret_cast<char *>(bias.data.data()), bias.numel() * sizeof(float));
      }
};