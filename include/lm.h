/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2026 Eamon Sippy
 */

#pragma once

#include "block.h"
#include "config/config.h"
#include "embedding.h"
#include "layernorm.h"
#include "linear.h"
#include "sampler.h"
#include "tensor.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

/**
 * @brief Cross-entropy loss for language modeling.
 *
 * @param logits  [B * T, V]. Raw logits.
 * @param targets [B * T]. Ground-truth token indices.
 * @return        Scalar mean loss.
 */
inline float cross_entropy(const Tensor &logits, const std::vector<int> &targets)
{
      int BT = logits.shape[0];
      int V = logits.shape[1];
      float loss = 0.0f;
      for (int i = 0; i < BT; ++i)
      {
            float maxv = -1e30f;
            for (int v = 0; v < V; ++v)
                  maxv = std::max(maxv, logits.at(i, v));
            float sumexp = 0.0f;
            for (int v = 0; v < V; ++v)
                  sumexp += std::exp(logits.at(i, v) - maxv);
            float log_prob = logits.at(i, targets[i]) - maxv - std::log(sumexp);
            loss -= log_prob;
      }
      return loss / (float)BT;
}

/**
 * @brief AdamW optimizer with first and second moment estimates.
 */
struct AdamW
{
      float lr, beta1, beta2, eps, weight_decay;
      int step_count;
      std::vector<float *> params;
      std::vector<int> sizes;
      std::vector<std::vector<float>> m, v;

      /**
       * @brief Construct with hyperparameters.
       *
       * @param lr_    Learning rate.
       * @param beta1_ First moment decay.
       * @param beta2_ Second moment decay.
       * @param eps_   Epsilon for numerical stability.
       * @param wd     Weight decay coefficient.
       */
      AdamW(float lr_ = 3e-4f, float beta1_ = 0.9f, float beta2_ = 0.999f, float eps_ = 1e-8f,
            float wd = 0.0f)
          : lr(lr_), beta1(beta1_), beta2(beta2_), eps(eps_), weight_decay(wd), step_count(0)
      {
      }

      /**
       * @brief Register a parameter vector for optimization.
       *
       * @param p Parameter vector. Pointer stored; caller owns memory.
       */
      void add_param(std::vector<float> &p)
      {
            params.push_back(p.data());
            sizes.push_back((int)p.size());
            m.emplace_back(p.size(), 0.0f);
            v.emplace_back(p.size(), 0.0f);
      }

      /**
       * @brief Single optimization step.
       *
       * @param grads Per-parameter gradients. Same order as add_param() calls.
       */
      void step(std::vector<std::vector<float>> &grads)
      {
            ++step_count;
            float bc1 = 1.0f - std::pow(beta1, step_count);
            float bc2 = 1.0f - std::pow(beta2, step_count);
            for (int pi = 0; pi < (int)params.size(); ++pi)
            {
                  for (int i = 0; i < sizes[pi]; ++i)
                  {
                        float g = grads[pi][i];
                        m[pi][i] = beta1 * m[pi][i] + (1.0f - beta1) * g;
                        v[pi][i] = beta2 * v[pi][i] + (1.0f - beta2) * g * g;
                        float mhat = m[pi][i] / bc1;
                        float vhat = v[pi][i] / bc2;
                        params[pi][i] -= lr * mhat / (std::sqrt(vhat) + eps);
                        if (weight_decay > 0.0f)
                              params[pi][i] -= lr * weight_decay * params[pi][i];
                  }
            }
      }
};

/**
 * @brief GPT language model.
 *
 * Token + position embeddings, N Transformer blocks, final LayerNorm,
 * and output projection. Not thread-safe across forward/generate calls.
 */
struct GPTLanguageModel
{
      std::mt19937 rng;
      int vocab_size, n_embd, n_head, n_layer, block_size;

      Embedding token_emb;
      Embedding pos_emb;
      std::vector<Block> blocks;
      LayerNorm ln_f;
      Linear lm_head;

      /**
       * @brief Construct and initialize all weights.
       *
       * @param vocab  Vocabulary size.
       * @param embd   Embedding dimension.
       * @param heads  Number of attention heads.
       * @param layers Number of Transformer blocks.
       * @param blk_sz Maximum sequence length (block size).
       * @param seed   PRNG seed for weight init.
       *
       * @throws std::invalid_argument if embd % heads != 0.
       */
      GPTLanguageModel(int vocab, int embd, int heads, int layers, int blk_sz, unsigned int seed)
          : vocab_size(vocab), n_embd(embd), n_head(heads), n_layer(layers), block_size(blk_sz),
            rng(seed), token_emb(vocab, embd, rng), pos_emb(blk_sz, embd, rng), ln_f(embd),
            lm_head(embd, vocab, true, rng)
      {
            for (int i = 0; i < layers; ++i)
                  blocks.emplace_back(embd, heads, rng);
      }

      /**
       * @brief Total trainable parameters.
       * @return Sum of all submodule parameter counts.
       */
      int num_params() const
      {
            int n = token_emb.num_params() + pos_emb.num_params() + ln_f.num_params() +
                    lm_head.num_params();
            for (auto &b : blocks)
                  n += b.num_params();
            return n;
      }

      /**
       * @brief Forward pass.
       *
       * @param idx      [B * T]. Flat token indices.
       * @param B        Batch size.
       * @param T        Sequence length.
       * @param targets  [B * T]. Ground-truth next-token indices. Empty for inference.
       * @param training Enables dropout when true.
       * @return         Pair of {logits [B * T, vocab_size], loss}. Loss=0 when no targets.
       */
      std::pair<Tensor, float> forward(const std::vector<int> &idx, int B, int T,
                                       const std::vector<int> &targets, bool training)
      {
            Tensor tok = token_emb.forward(idx, B, T);
            Tensor pos = pos_emb.forward_pos(T);

            Tensor x({B, T, n_embd});
            for (int b = 0; b < B; ++b)
                  for (int t = 0; t < T; ++t)
                        for (int d = 0; d < n_embd; ++d)
                              x.at(b, t, d) = tok.at(b, t, d) + pos.at(0, t, d);

            for (auto &blk : blocks)
                  x = blk.forward(x, training, rng);

            x = ln_f.forward(x);

            Tensor logits3d = lm_head.forward(x);

            Tensor logits({B * T, vocab_size});
            for (int i = 0; i < B * T; ++i)
                  for (int v = 0; v < vocab_size; ++v)
                        logits.at(i, v) = logits3d.data[i * vocab_size + v];

            float loss = 0.0f;
            if (!targets.empty())
                  loss = cross_entropy(logits, targets);

            return {logits, loss};
      }

      /**
       * @brief Autoregressive token generation with repetition penalty.
       *
       * @param context        Initial token sequence.
       * @param max_new_tokens Number of tokens to generate.
       * @param params         SamplerParams. Defaults to config.h values.
       * @return               Extended token sequence (context + generated).
       */
      std::vector<int> generate(std::vector<int> context, int max_new_tokens,
                                const SamplerParams &params = SamplerParams())
      {
            std::uniform_real_distribution<float> udist(0.0f, 1.0f);

            for (int step = 0; step < max_new_tokens; ++step)
            {
                  int T = std::min((int)context.size(), block_size);
                  std::vector<int> ctx(context.end() - T, context.end());

                  Tensor logits = forward(ctx, 1, T, std::vector<int>(), false).first;

                  int offset = (T - 1) * vocab_size;
                  std::vector<float> last(vocab_size);
                  for (int v = 0; v < vocab_size; ++v)
                        last[v] = logits.data[offset + v];

                  apply_rep_penalty(last, context, params);

                  float maxv = *std::max_element(last.begin(), last.end());
                  float sumv = 0.0f;
                  for (auto &lv : last)
                  {
                        lv = std::exp(lv - maxv);
                        sumv += lv;
                  }
                  for (auto &lv : last)
                        lv /= sumv;

                  float r = udist(rng);
                  float cumsum = 0.0f;
                  int next_tok = vocab_size - 1;
                  for (int v = 0; v < vocab_size; ++v)
                  {
                        cumsum += last[v];
                        if (r < cumsum)
                        {
                              next_tok = v;
                              break;
                        }
                  }
                  context.push_back(next_tok);
            }
            return context;
      }

      /**
       * @brief Serialize all weights to binary file.
       *
       * @param path Output file path. Overwrites if exists.
       */
      void save(const std::string &path) const
      {
            std::ofstream f(path, std::ios::binary);
            if (!f)
            {
                  std::cerr << "[ERROR] Cannot open " << path << " for writing\n";
                  return;
            }
            token_emb.save(f);
            pos_emb.save(f);
            for (auto &b : blocks)
                  b.save(f);
            ln_f.save(f);
            lm_head.save(f);
            std::cout << "[SAVE]  Weights written to " << path << "\n";
      }

      /**
       * @brief Deserialize all weights from binary file.
       *
       * @param path Input file path.
       */
      void load(const std::string &path)
      {
            std::ifstream f(path, std::ios::binary);
            if (!f)
            {
                  std::cerr << "[ERROR] Cannot open " << path << " for reading\n";
                  return;
            }
            token_emb.load(f);
            pos_emb.load(f);
            for (auto &b : blocks)
                  b.load(f);
            ln_f.load(f);
            lm_head.load(f);
            std::cout << "[LOAD]  Weights loaded from " << path << "\n";
      }
};