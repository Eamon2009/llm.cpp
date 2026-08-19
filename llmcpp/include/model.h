//  include/gpt.h  –  GPT Language Model
//  Mirrors: class GPTLanguageModel in Python

#pragma once
#include "block.h"
#include "config/config.h"
#include "embedding.h"
#include "layernorm.h"
#include "linear.h"
#include "tensor.cuh"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// ------------------------------------------------------------------
// Cross-entropy loss for language modelling
//   logits : [B*T, vocab_size]   (flat)
//   targets: flat integer indices, length B*T
// ------------------------------------------------------------------
inline float cross_entropy(const Tensor &logits, const std::vector<int> &targets)
{
      int BT = logits.shape[0];
      int V = logits.shape[1];
      float loss = 0.0f;
      for (int i = 0; i < BT; ++i)
      {
            // log-softmax + NLL
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

// AdamW optimiser (simple, stateful)
struct AdamW
{
      float lr, beta1, beta2, eps, weight_decay;
      int step_count;
      std::vector<float *> params;
      std::vector<int> sizes;
      std::vector<std::vector<float>> m, v; // first/second moment

      AdamW(float lr_ = 3e-4f,
            float beta1_ = 0.9f,
            float beta2_ = 0.999f,
            float eps_ = 1e-8f,
            float wd = 0.0f)
            : lr(lr_), beta1(beta1_), beta2(beta2_), eps(eps_), weight_decay(wd), step_count(0)
      {
      }

      void add_param(std::vector<float> &p)
      {
            params.push_back(p.data());
            sizes.push_back((int)p.size());
            m.emplace_back(p.size(), 0.0f);
            v.emplace_back(p.size(), 0.0f);
      }

      // grads must be passed in the same order as params were added
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

// GPTLanguageModel
// ------------------------------------------------------------------
struct GPTLanguageModel
{
      std::mt19937 rng;
      int vocab_size, n_embd, n_head, n_layer, block_size;

      Embedding token_emb; // [vocab_size, n_embd]
      Embedding pos_emb;   // [block_size, n_embd]
      std::vector<Block> blocks;
      LayerNorm ln_f;
      Linear lm_head; // [n_embd, vocab_size]

      GPTLanguageModel(int vocab, int embd, int heads, int layers, int blk_sz, unsigned int seed)
            : vocab_size(vocab), n_embd(embd), n_head(heads), n_layer(layers), block_size(blk_sz),
              rng(seed), token_emb(vocab, embd, rng), pos_emb(blk_sz, embd, rng), ln_f(embd),
              lm_head(embd, vocab, true, rng)
      {
            for (int i = 0; i < layers; ++i)
                  blocks.emplace_back(embd, heads, rng);
      }

      int num_params() const
      {
            int n = token_emb.num_params() + pos_emb.num_params() + ln_f.num_params() +
                    lm_head.num_params();
            for (auto &b : blocks)
                  n += b.num_params();
            return n;
      }

      // ----------------------------------------------------------------
      // forward
      //   idx    : flat [B*T] token indices
      //   B, T   : batch and sequence length
      //   targets: flat [B*T] next-token indices (empty = inference)
      //   returns: (logits [B*T, vocab], loss)  loss=0 when no targets
      std::pair<Tensor, float> forward(const std::vector<int> &idx,
                                       int B,
                                       int T,
                                       const std::vector<int> &targets,
                                       bool training)
      {
            // token + position embeddings
            Tensor tok = token_emb.forward(idx, B, T); // [B, T, n_embd]
            Tensor pos = pos_emb.forward_pos(T);       // [1, T, n_embd]

            // broadcast pos over batch
            Tensor x({B, T, n_embd});
            for (int b = 0; b < B; ++b)
                  for (int t = 0; t < T; ++t)
                        for (int d = 0; d < n_embd; ++d)
                              x.at(b, t, d) = tok.at(b, t, d) + pos.at(0, t, d);

            // transformer blocks
            for (auto &blk : blocks)
                  x = blk.forward(x, training, rng);

            // final layer norm
            x = ln_f.forward(x);

            // lm_head  logits [B, T, vocab]
            Tensor logits3d = lm_head.forward(x); // [B, T, vocab]

            // reshape to [B*T, vocab]
            Tensor logits({B * T, vocab_size});
            for (int i = 0; i < B * T; ++i)
                  for (int v = 0; v < vocab_size; ++v)
                        logits.at(i, v) = logits3d.data[i * vocab_size + v];

            float loss = 0.0f;
            if (!targets.empty())
                  loss = cross_entropy(logits, targets);

            return {logits, loss};
      }

      std::vector<int> generate(std::vector<int> context, int max_new_tokens)
      {
            std::uniform_real_distribution<float> udist(0.0f, 1.0f);

            for (int step = 0; step < max_new_tokens; ++step)
            {
                  // crop context to block_size
                  int T = std::min((int)context.size(), block_size);
                  std::vector<int> ctx(context.end() - T, context.end());

                  std::pair<Tensor, float> forward_result =
                        forward(ctx, 1, T, std::vector<int>(), false);
                  Tensor logits = forward_result.first;

                  // pick last time-step logits  [vocab]
                  int offset = (T - 1) * vocab_size;
                  std::vector<float> last_logits(vocab_size);
                  for (int v = 0; v < vocab_size; ++v)
                        last_logits[v] = logits.data[offset + v];

                  // softmax
                  float maxv = *std::max_element(last_logits.begin(), last_logits.end());
                  float sumv = 0.0f;
                  for (auto &lv : last_logits)
                  {
                        lv = std::exp(lv - maxv);
                        sumv += lv;
                  }
                  for (auto &lv : last_logits)
                        lv /= sumv;

                  // multinomial sample
                  float r = udist(rng);
                  float cumsum = 0.0f;
                  int next_tok = vocab_size - 1;
                  for (int v = 0; v < vocab_size; ++v)
                  {
                        cumsum += last_logits[v];
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

            // save / load weights
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
