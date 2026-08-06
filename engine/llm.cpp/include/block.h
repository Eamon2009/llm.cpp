#pragma once
// ============================================================
//  include/block.h  –  Single Transformer block
//  Mirrors: class Block in Python
//  Architecture: pre-LN - MHA - residual -pre-LN - FFN -residual
// ============================================================

#include "attention.h"
#include "feedforward.h"
#include "layernorm.h"
#include <fstream>

struct Block
{
      MultiHeadAttention sa; // self-attention
      FeedForward ffwd;      // feed-forward
      LayerNorm ln1;         // before attention
      LayerNorm ln2;         // before ffn

      Block() = default;

      Block(int n_embd, int n_head, std::mt19937 &rng)
          : sa(n_embd, n_head, n_embd / n_head, rng),
            ffwd(n_embd, rng),
            ln1(n_embd),
            ln2(n_embd) {}

      // x: [B, T, n_embd]  →  [B, T, n_embd]
      Tensor forward(const Tensor &x, bool training, std::mt19937 &rng) const
      {
            // x = x + sa(ln1(x))
            Tensor xn1 = ln1.forward(x);
            Tensor attn = sa.forward(xn1, training, rng);
            Tensor x2 = add(x, attn);

            // x = x + ffwd(ln2(x))
            Tensor xn2 = ln2.forward(x2);
            Tensor ffn = ffwd.forward(xn2, training, rng);
            return add(x2, ffn);
      }

      int num_params() const
      {
            return sa.num_params() + ffwd.num_params() + ln1.num_params() + ln2.num_params();
      }

      void save(std::ofstream &f) const
      {
            sa.save(f);
            ffwd.save(f);
            ln1.save(f);
            ln2.save(f);
      }
      void load(std::ifstream &f)
      {
            sa.load(f);
            ffwd.load(f);
            ln1.load(f);
            ln2.load(f);
      }
};