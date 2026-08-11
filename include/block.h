/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2026 Eamon Sippy
 */

#pragma once

#include "attention.h"
#include "feedforward.h"
#include "layernorm.h"

#include <fstream>

/**
 * @brief Pre-norm residual Transformer block.
 *
 * Layout: ln1 -> MHA -> residual -> ln2 -> FFN -> residual.
 * Throws if n_embd % n_head != 0.
 */
struct Block
{
      // ***Inner Layers***
      MultiHeadAttention sa;
      FeedForward ffwd;
      LayerNorm ln1; // pre-attention norm
      LayerNorm ln2; // pre-ffn norm

      /**
       * @brief Default constructor. Leaves block uninitialized.
       */
      Block() = default;

      /**
       * @brief Construct and initialize weights.
       *
       * @param n_embd Embedding dimension. Must be divisible by n_head.
       * @param n_head Parallel attention heads.
       * @param rng    Seeded MT19937 for weight init.
       *
       * @throws std::invalid_argument if n_embd % n_head != 0.
       */
      Block(int n_embd, int n_head, std::mt19937 &rng)
          : sa(n_embd, n_head, n_embd / n_head, rng), ffwd(n_embd, rng), ln1(n_embd), ln2(n_embd)
      {
      }

      /**
       * @brief Forward pass.
       *
       * @param x        [batch, seq, n_embd].
       * @param training Enables dropout when true. Inference path is deterministic.
       * @param rng      Thread-local MT19937 for stochastic ops.
       * @return         [batch, seq, n_embd]. New allocation.
       */
      Tensor forward(const Tensor &x, bool training, std::mt19937 &rng) const
      {
            Tensor xn1 = ln1.forward(x);
            Tensor attn = sa.forward(xn1, training, rng);
            Tensor x2 = add(x, attn);

            Tensor xn2 = ln2.forward(x2);
            Tensor ffn = ffwd.forward(xn2, training, rng);
            return add(x2, ffn);
      }

      /**
       * @brief Total trainable parameters.
       * @return Sum of all submodule parameter counts.
       */
      int num_params() const
      {
            return sa.num_params() + ffwd.num_params() + ln1.num_params() + ln2.num_params();
      }

      /**
       * @brief Serialize weights to binary stream.
       * @param f Open binary output stream.
       */
      void save(std::ofstream &f) const
      {
            sa.save(f);
            ffwd.save(f);
            ln1.save(f);
            ln2.save(f);
      }

      /**
       * @brief Deserialize weights from binary stream.
       * @param f Open binary input stream.
       */
      void load(std::ifstream &f)
      {
            sa.load(f);
            ffwd.load(f);
            ln1.load(f);
            ln2.load(f);
      }
};