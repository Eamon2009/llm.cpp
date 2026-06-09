#pragma once
// ============================================================
//  include/feedforward.h  –  Position-wise feed-forward network
//  Mirrors: class FeedFoward in Python
// ============================================================

#include "tensor.h"
#include "linear.h"
#include "config/config.h"
#include <fstream>

struct FeedForward
{
      Linear fc1; // n_embd  → 4*n_embd
      Linear fc2; // 4*n_embd → n_embd

      FeedForward() = default;

      FeedForward(int n_embd, std::mt19937 &rng)
          : fc1(n_embd, 4 * n_embd, true, rng),
            fc2(4 * n_embd, n_embd, true, rng) {}

      // x: [B, T, n_embd]  →  [B, T, n_embd]
      Tensor forward(const Tensor &x, bool training, std::mt19937 &rng) const
      {
            Tensor h = fc1.forward(x); // [B, T, 4*n_embd]
            h = relu(h);               // ReLU
            h = fc2.forward(h);        // [B, T, n_embd]
            return dropout(h, DROPOUT, training, rng);
      }

      int num_params() const
      {
            return fc1.num_params() + fc2.num_params();
      }

      void save(std::ofstream &f) const
      {
            fc1.save(f);
            fc2.save(f);
      }
      void load(std::ifstream &f)
      {
            fc1.load(f);
            fc2.load(f);
      }
};