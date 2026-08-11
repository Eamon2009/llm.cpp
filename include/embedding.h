/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2026 Eamon Sippy
 */

#pragma once

#include "tensor.h"

#include <fstream>
#include <random>
#include <vector>

/**
 * @brief Learnable embedding lookup table.
 */
struct Embedding
{
      int num_embeddings; // Vocabulary size or max sequence length
      int embedding_dim;
      Tensor weight; // [num_embeddings, embedding_dim]

      /**
       * @brief Default constructor. Leaves layer uninitialized.
       */
      Embedding() = default;

      /**
       * @brief Construct and initialize weights.
       *
       * @param num_emb Total embeddings (vocab size or max positions).
       * @param emb_dim Embedding dimension.
       * @param rng     Seeded MT19937 for weight init (N(0, 0.02)).
       */
      Embedding(int num_emb, int emb_dim, std::mt19937 &rng)
          : num_embeddings(num_emb), embedding_dim(emb_dim)
      {
            weight = Tensor::randn({num_emb, emb_dim}, 0.0f, 0.02f, rng);
      }

      /**
       * @brief Token embedding lookup.
       *
       * @param idx [B * T]. Flat token indices.
       * @param B   Batch size.
       * @param T   Sequence length.
       * @return    [B, T, embedding_dim]. New allocation.
       */
      Tensor forward(const std::vector<int> &idx, int B, int T) const
      {
            Tensor out({B, T, embedding_dim});
            for (int b = 0; b < B; ++b)
            {
                  for (int t = 0; t < T; ++t)
                  {
                        int token = idx[b * T + t];
                        for (int d = 0; d < embedding_dim; ++d)
                        {
                              out.at(b, t, d) = weight.at(token, d);
                        }
                  }
            }
            return out;
      }

      /**
       * @brief Positional embedding lookup.
       *
       * @param T Number of positions to retrieve.
       * @return  [1, T, embedding_dim]. New allocation.
       */
      Tensor forward_pos(int T) const
      {
            Tensor out({1, T, embedding_dim});
            for (int t = 0; t < T; ++t)
            {
                  for (int d = 0; d < embedding_dim; ++d)
                  {
                        out.at(0, t, d) = weight.at(t, d);
                  }
            }
            return out;
      }

      /**
       * @brief Total trainable parameters.
       * @return weight.numel().
       */
      int num_params() const { return weight.numel(); }

      /**
       * @brief Serialize weights to binary stream.
       * @param f Open binary output stream.
       */
      void save(std::ofstream &f) const
      {
            f.write(reinterpret_cast<const char *>(weight.data.data()),
                    weight.numel() * sizeof(float));
      }

      /**
       * @brief Deserialize weights from binary stream.
       * @param f Open binary input stream.
       */
      void load(std::ifstream &f)
      {
            f.read(reinterpret_cast<char *>(weight.data.data()), weight.numel() * sizeof(float));
      }
};