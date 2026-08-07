/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2026 Eamon Sippy
 */

#pragma once

#include "config/config.h"
#include "tensor.h"

#include <cassert>
#include <cmath>
#include <vector>

/**
 * @brief Gradient accumulator for Linear projection weights and bias.
 */
struct GradLinear
{
      Tensor dW; // [in_features, out_features]
      Tensor db; // [out_features] (empty when has_bias=false)
      bool has_bias;

      /**
       * @brief Default constructor. Leaves gradients uninitialized.
       */
      GradLinear() : has_bias(false) {}

      /**
       * @brief Allocate zero-initialized gradient buffers.
       *
       * @param in_f  Input feature dimension.
       * @param out_f Output feature dimension.
       * @param bias  True if layer has bias term.
       */
      GradLinear(int in_f, int out_f, bool bias)
          : dW({in_f, out_f}, 0.0f), db(bias ? Tensor({out_f}, 0.0f) : Tensor()), has_bias(bias)
      {
      }

      /**
       * @brief Zero all gradient buffers.
       */
      void zero()
      {
            dW.fill(0.0f);
            if (has_bias)
                  db.fill(0.0f);
      }
};

/**
 * @brief Gradient accumulator for embedding lookup weights.
 */
struct GradEmbedding
{
      Tensor dW; // [num_embeddings, embedding_dim]

      /**
       * @brief Default constructor. Leaves gradients uninitialized.
       */
      GradEmbedding() = default;

      /**
       * @brief Allocate zero-initialized gradient buffer.
       *
       * @param n Number of embeddings.
       * @param d Embedding dimension.
       */
      GradEmbedding(int n, int d) : dW({n, d}, 0.0f) {}

      /**
       * @brief Zero gradient buffer.
       */
      void zero() { dW.fill(0.0f); }
};

/**
 * @brief Gradient accumulator for LayerNorm scale and shift parameters.
 */
struct GradLayerNorm
{
      Tensor dgamma; // [channels]
      Tensor dbeta;  // [channels]

      /**
       * @brief Default constructor. Leaves gradients uninitialized.
       */
      GradLayerNorm() = default;

      /**
       * @brief Allocate zero-initialized gradient buffers.
       *
       * @param C Channel dimension.
       */
      GradLayerNorm(int C) : dgamma({C}, 0.0f), dbeta({C}, 0.0f) {}

      /**
       * @brief Zero all gradient buffers.
       */
      void zero()
      {
            dgamma.fill(0.0f);
            dbeta.fill(0.0f);
      }
};

/**
 * @brief Gradient accumulator for single attention head (Q, K, V projections).
 */
struct GradHead
{
      GradLinear dkey, dquery, dvalue;

      /**
       * @brief Default constructor. Leaves gradients uninitialized.
       */
      GradHead() = default;

      /**
       * @brief Allocate zero-initialized gradient buffers for all projections.
       *
       * @param n_embd Input embedding dimension.
       * @param hs     Head dimension.
       */
      GradHead(int n_embd, int hs)
          : dkey(n_embd, hs, false), dquery(n_embd, hs, false), dvalue(n_embd, hs, false)
      {
      }

      /**
       * @brief Zero all gradient buffers.
       */
      void zero()
      {
            dkey.zero();
            dquery.zero();
            dvalue.zero();
      }
};

/**
 * @brief Gradient accumulator for Multi-Head Attention (all heads + output projection).
 */
struct GradMHA
{
      std::vector<GradHead> heads;
      GradLinear proj;

      /**
       * @brief Default constructor. Leaves gradients uninitialized.
       */
      GradMHA() = default;

      /**
       * @brief Allocate zero-initialized gradient buffers.
       *
       * @param n_embd Embedding dimension.
       * @param n_head Number of attention heads.
       * @param hs     Head dimension.
       */
      GradMHA(int n_embd, int n_head, int hs) : proj(n_head * hs, n_embd, true)
      {
            for (int i = 0; i < n_head; ++i)
                  heads.emplace_back(n_embd, hs);
      }

      /**
       * @brief Zero all gradient buffers.
       */
      void zero()
      {
            for (auto &h : heads)
                  h.zero();
            proj.zero();
      }
};

/**
 * @brief Gradient accumulator for Feed-Forward Network (FC1 + FC2).
 */
struct GradFFN
{
      GradLinear dfc1, dfc2;

      /**
       * @brief Default constructor. Leaves gradients uninitialized.
       */
      GradFFN() = default;

      /**
       * @brief Allocate zero-initialized gradient buffers.
       *
       * @param n_embd Embedding dimension. FC1 expands to 4 * n_embd.
       */
      GradFFN(int n_embd) : dfc1(n_embd, 4 * n_embd, true), dfc2(4 * n_embd, n_embd, true) {}

      /**
       * @brief Zero all gradient buffers.
       */
      void zero()
      {
            dfc1.zero();
            dfc2.zero();
      }
};

/**
 * @brief Gradient accumulator for a single Transformer Block.
 */
struct GradBlock
{
      GradMHA sa;
      GradFFN ffwd;
      GradLayerNorm ln1, ln2;

      /**
       * @brief Default constructor. Leaves gradients uninitialized.
       */
      GradBlock() = default;

      /**
       * @brief Allocate zero-initialized gradient buffers for all submodules.
       *
       * @param n_embd Embedding dimension.
       * @param n_head Number of attention heads.
       * @param hs     Head dimension.
       */
      GradBlock(int n_embd, int n_head, int hs)
          : sa(n_embd, n_head, hs), ffwd(n_embd), ln1(n_embd), ln2(n_embd)
      {
      }

      /**
       * @brief Zero all gradient buffers.
       */
      void zero()
      {
            sa.zero();
            ffwd.zero();
            ln1.zero();
            ln2.zero();
      }
};

/**
 * @brief Master gradient container for the full GPT model.
 */
struct Grads
{
      GradEmbedding tok_emb, pos_emb;
      std::vector<GradBlock> blocks;
      GradLayerNorm ln_f;
      GradLinear lm_head;

      /**
       * @brief Default constructor. Leaves gradients uninitialized.
       */
      Grads() = default;

      /**
       * @brief Allocate zero-initialized gradient buffers for all model parameters.
       *
       * @param vocab_size  Vocabulary size.
       * @param n_embd      Embedding dimension.
       * @param n_head      Number of attention heads.
       * @param n_layer     Number of Transformer blocks.
       * @param block_size  Maximum sequence length.
       */
      Grads(int vocab_size, int n_embd, int n_head, int n_layer, int block_size)
          : tok_emb(vocab_size, n_embd), pos_emb(block_size, n_embd), ln_f(n_embd),
            lm_head(n_embd, vocab_size, true)
      {
            int hs = n_embd / n_head;
            for (int i = 0; i < n_layer; ++i)
                  blocks.emplace_back(n_embd, n_head, hs);
      }

      /**
       * @brief Zero all gradient buffers across all model layers.
       */
      void zero()
      {
            tok_emb.zero();
            pos_emb.zero();
            for (auto &b : blocks)
                  b.zero();
            ln_f.zero();
            lm_head.zero();
      }
};

/**
 * @brief Activation cache for single-head attention backward.
 */
struct SavedHead
{
      Tensor x;            // [B, T, n_embd]
      Tensor k, q, v;      // [B, T, head_size]
      Tensor wei_pre;      // Unnormalized logits [B, T, T]
      Tensor wei;          // Post-softmax probabilities [B, T, T]
      Tensor dropout_mask; // [B, T, T]
      bool used_dropout;
};

/**
 * @brief Activation cache for Multi-Head Attention backward.
 */
struct SavedMHA
{
      std::vector<SavedHead> heads;
      Tensor concat;       // [B, T, n_head * head_size]
      Tensor proj_out;     // [B, T, n_embd]
      Tensor dropout_mask; // Projection dropout mask
      bool used_dropout;
};

/**
 * @brief Activation cache for Feed-Forward Network backward.
 */
struct SavedFFN
{
      Tensor x;            // [B, T, n_embd]
      Tensor h_pre;        // FC1 pre-ReLU [B, T, 4 * n_embd]
      Tensor h;            // Post-ReLU [B, T, 4 * n_embd]
      Tensor out_pre;      // FC2 pre-dropout [B, T, n_embd]
      Tensor dropout_mask; // FFN dropout mask
      bool used_dropout;
};

/**
 * @brief Activation cache for LayerNorm backward.
 */
struct SavedLN
{
      Tensor x;                              // [B, T, C]
      Tensor xhat;                           // Normalized [B, T, C]
      Tensor inv_std;                        // [B, T, 1]
      std::vector<float> mu_vec, invstd_vec; // [B * T]
};

/**
 * @brief Activation cache for a single Transformer Block.
 */
struct SavedBlock
{
      SavedLN ln1, ln2;
      SavedMHA mha;
      SavedFFN ffn;
      Tensor x_in;        // [B, T, C]
      Tensor x_after_mha; // [B, T, C]
};

/**
 * @brief Root activation cache from training forward pass.
 */
struct SavedForward
{
      std::vector<int> idx; // [B * T]
      int B, T;
      Tensor tok_out; // [B, T, C]
      Tensor pos_out; // [1, T, C]
      Tensor emb_sum; // [B, T, C]

      std::vector<SavedBlock> blocks;

      SavedLN ln_f;
      Tensor lm_in;    // [B, T, C]
      Tensor logits3d; // [B, T, V]
      Tensor logits2d; // [B * T, V]
      std::vector<int> targets;
};

/**
 * @brief Cross-entropy + softmax backward.
 *
 * @param logits2d [B * T, V]. Raw logits.
 * @param targets  [B * T]. Ground-truth token indices.
 * @return         [B * T, V]. Gradient w.r.t. logits. New allocation.
 */
inline Tensor backward_cross_entropy(const Tensor &logits2d, const std::vector<int> &targets)
{
      int BT = logits2d.shape[0], V = logits2d.shape[1];
      Tensor dlogits({BT, V}, 0.0f);

      for (int i = 0; i < BT; ++i)
      {
            // Numerically stable softmax
            float maxv = -1e30f;
            for (int v = 0; v < V; ++v)
                  maxv = std::max(maxv, logits2d.at(i, v));

            float sumv = 0.0f;
            for (int v = 0; v < V; ++v)
            {
                  dlogits.at(i, v) = std::exp(logits2d.at(i, v) - maxv);
                  sumv += dlogits.at(i, v);
            }
            for (int v = 0; v < V; ++v)
                  dlogits.at(i, v) /= sumv;

            dlogits.at(i, targets[i]) -= 1.0f;

            for (int v = 0; v < V; ++v)
                  dlogits.at(i, v) /= (float)BT;
      }
      return dlogits;
}

/**
 * @brief Linear projection backward.
 *
 * @param dOut [B, T, E]. Upstream gradient.
 * @param x    [B, T, D]. Forward input.
 * @param W    [D, E]. Weight matrix.
 * @param g    Gradient accumulator. dW and db are accumulated (not overwritten).
 * @return     [B, T, D]. Input gradient. New allocation.
 */
inline Tensor backward_linear(const Tensor &dOut, const Tensor &x, const Tensor &W, GradLinear &g)
{
      int B = dOut.shape[0], T = dOut.shape[1], E = dOut.shape[2];
      int D = W.shape[0];
      assert(E == W.shape[1]);

      Tensor dX({B, T, D}, 0.0f);
      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < D; ++d)
                  {
                        float s = 0.0f;
                        for (int e = 0; e < E; ++e)
                              s += dOut.at(b, t, e) * W.at(d, e);
                        dX.at(b, t, d) += s;
                  }

      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < D; ++d)
                        for (int e = 0; e < E; ++e)
                              g.dW.at(d, e) += x.at(b, t, d) * dOut.at(b, t, e);

      if (g.has_bias)
            for (int b = 0; b < B; ++b)
                  for (int t = 0; t < T; ++t)
                        for (int e = 0; e < E; ++e)
                              g.db.at(e) += dOut.at(b, t, e);

      return dX;
}

/**
 * @brief LayerNorm backward (Ba et al. formulation).
 *
 * @param dOut  [B, T, C]. Upstream gradient.
 * @param saved Forward activation cache.
 * @param gamma [C]. Scale parameter.
 * @param g     Gradient accumulator. dgamma and dbeta accumulated.
 * @return      [B, T, C]. Input gradient. New allocation.
 */
inline Tensor backward_layernorm(const Tensor &dOut, const SavedLN &saved, const Tensor &gamma,
                                 GradLayerNorm &g)
{
      int B = dOut.shape[0], T = dOut.shape[1], C = dOut.shape[2];
      Tensor dX({B, T, C}, 0.0f);

      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  float inv_std = saved.invstd_vec[b * T + t];

                  for (int c = 0; c < C; ++c)
                  {
                        float xhat_c = saved.xhat.at(b, t, c);
                        g.dgamma.at(c) += dOut.at(b, t, c) * xhat_c;
                        g.dbeta.at(c) += dOut.at(b, t, c);
                  }

                  float sum1 = 0.0f, sum2 = 0.0f;
                  for (int c = 0; c < C; ++c)
                  {
                        float gd = gamma.at(c) * dOut.at(b, t, c);
                        sum1 += gd;
                        sum2 += gd * saved.xhat.at(b, t, c);
                  }

                  for (int c = 0; c < C; ++c)
                  {
                        float xhat_c = saved.xhat.at(b, t, c);
                        dX.at(b, t, c) =
                            inv_std / C *
                            (C * gamma.at(c) * dOut.at(b, t, c) - sum1 - xhat_c * sum2);
                  }
            }
      }
      return dX;
}

/**
 * @brief ReLU backward.
 *
 * @param dOut    Upstream gradient. Any shape.
 * @param pre_relu Pre-activation values. Same shape as dOut.
 * @return         Input gradient. Same shape. New allocation.
 */
inline Tensor backward_relu(const Tensor &dOut, const Tensor &pre_relu)
{
      Tensor dX(dOut.shape);
      for (int i = 0; i < dOut.numel(); ++i)
            dX.data[i] = (pre_relu.data[i] > 0.0f) ? dOut.data[i] : 0.0f;
      return dX;
}

/**
 * @brief Inverted dropout backward.
 *
 * @param dOut Upstream gradient. Any shape.
 * @param mask Binary retention mask (1=kept, 0=dropped). Same shape.
 * @param p    Dropout probability.
 * @return     Input gradient. Same shape. New allocation.
 */
inline Tensor backward_dropout(const Tensor &dOut, const Tensor &mask, float p)
{
      if (p == 0.0f)
            return dOut;

      Tensor dX(dOut.shape);
      float inv_keep = 1.0f / (1.0f - p);
      for (int i = 0; i < dOut.numel(); ++i)
            dX.data[i] = dOut.data[i] * mask.data[i] * inv_keep;
      return dX;
}

/**
 * @brief Batched matrix multiplication backward.
 *
 * @param dOut [B, T, T2]. Upstream gradient.
 * @param a    [B, T, D]. Forward LHS.
 * @param b    [B, D, T2]. Forward RHS.
 * @return     {dA [B, T, D], dB [B, D, T2]}. Both new allocations.
 */
inline std::pair<Tensor, Tensor> backward_bmm(const Tensor &dOut, const Tensor &a, const Tensor &b)
{
      int B = dOut.shape[0], T = dOut.shape[1], T2 = dOut.shape[2];
      int D = a.shape[2];
      Tensor da({B, T, D}, 0.0f);
      Tensor db({B, D, T2}, 0.0f);

      for (int bb = 0; bb < B; ++bb)
      {
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < D; ++d)
                  {
                        float s = 0.0f;
                        for (int t2 = 0; t2 < T2; ++t2)
                              s += dOut.at(bb, t, t2) * b.at(bb, d, t2);
                        da.at(bb, t, d) += s;
                  }

            for (int d = 0; d < D; ++d)
                  for (int t2 = 0; t2 < T2; ++t2)
                  {
                        float s = 0.0f;
                        for (int t = 0; t < T; ++t)
                              s += a.at(bb, t, d) * dOut.at(bb, t, t2);
                        db.at(bb, d, t2) += s;
                  }
      }
      return {da, db};
}

/**
 * @brief 3D softmax backward.
 *
 * @param dwei [B, T, T]. Upstream gradient.
 * @param wei  [B, T, T]. Forward softmax probabilities.
 * @return     [B, T, T]. Pre-softmax gradient. New allocation.
 */
inline Tensor backward_softmax3d(const Tensor &dwei, const Tensor &wei)
{
      int B = wei.shape[0], T1 = wei.shape[1], T2 = wei.shape[2];
      Tensor dpre({B, T1, T2}, 0.0f);

      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T1; ++t)
            {
                  float dot = 0.0f;
                  for (int t2 = 0; t2 < T2; ++t2)
                        dot += wei.at(b, t, t2) * dwei.at(b, t, t2);

                  for (int t2 = 0; t2 < T2; ++t2)
                        dpre.at(b, t, t2) = wei.at(b, t, t2) * (dwei.at(b, t, t2) - dot);
            }
      return dpre;
}

/**
 * @brief Split concatenated multi-head gradient into per-head tensors.
 *
 * @param dConcat    [B, T, sum(hs)]. Upstream gradient.
 * @param head_sizes Per-head dimensions. Sum must equal dConcat.shape[2].
 * @return           Vector of [B, T, hs] tensors. Each new allocation.
 */
inline std::vector<Tensor> backward_cat_last(const Tensor &dConcat,
                                             const std::vector<int> &head_sizes)
{
      int B = dConcat.shape[0], T = dConcat.shape[1];
      std::vector<Tensor> out;
      int offset = 0;

      for (int hs : head_sizes)
      {
            Tensor dh({B, T, hs}, 0.0f);
            for (int b = 0; b < B; ++b)
                  for (int t = 0; t < T; ++t)
                        for (int d = 0; d < hs; ++d)
                              dh.at(b, t, d) = dConcat.at(b, t, offset + d);
            out.push_back(dh);
            offset += hs;
      }
      return out;
}

/**
 * @brief LayerNorm forward with activation caching.
 *
 * @param x     [B, T, C].
 * @param gamma [C]. Scale.
 * @param beta  [C]. Shift.
 * @param saved Output activation cache. Populated by this call.
 * @param eps   Epsilon for numerical stability.
 * @return      [B, T, C]. Normalized output. New allocation.
 */
inline Tensor forward_ln_save(const Tensor &x, const Tensor &gamma, const Tensor &beta,
                              SavedLN &saved, float eps = 1e-5f)
{
      int B = x.shape[0], T = x.shape[1], C = x.shape[2];
      saved.x = x;
      saved.xhat = Tensor({B, T, C});
      saved.mu_vec.resize(B * T);
      saved.invstd_vec.resize(B * T);
      Tensor out({B, T, C});

      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  float mu = 0.0f;
                  for (int c = 0; c < C; ++c)
                        mu += x.at(b, t, c);
                  mu /= C;

                  float var = 0.0f;
                  for (int c = 0; c < C; ++c)
                  {
                        float d = x.at(b, t, c) - mu;
                        var += d * d;
                  }
                  var /= C;

                  float inv = 1.0f / std::sqrt(var + eps);
                  saved.mu_vec[b * T + t] = mu;
                  saved.invstd_vec[b * T + t] = inv;

                  for (int c = 0; c < C; ++c)
                  {
                        float xh = (x.at(b, t, c) - mu) * inv;
                        saved.xhat.at(b, t, c) = xh;
                        out.at(b, t, c) = xh * gamma.at(c) + beta.at(c);
                  }
            }
      }
      return out;
}

/**
 * @brief Single attention head forward with activation caching.
 *
 * @param x         [B, T, n_embd].
 * @param Wk        [n_embd, head_size]. Key weight.
 * @param Wq        [n_embd, head_size]. Query weight.
 * @param Wv        [n_embd, head_size]. Value weight.
 * @param training  Enables dropout when true.
 * @param drop_p    Dropout probability.
 * @param rng       Thread-local MT19937.
 * @param sh        Output activation cache. Populated by this call.
 * @return          [B, T, head_size]. New allocation.
 */
inline Tensor forward_head_save(const Tensor &x, const Tensor &Wk, const Tensor &Wq,
                                const Tensor &Wv, bool training, float drop_p, std::mt19937 &rng,
                                SavedHead &sh)
{
      int B = x.shape[0], T = x.shape[1], hs = Wk.shape[1];
      sh.x = x;
      sh.k = matmul(x, Wk); // [B, T, head_size]
      sh.q = matmul(x, Wq);
      sh.v = matmul(x, Wv);

      float scale = 1.0f / std::sqrt((float)hs);

      Tensor kt = transpose23(sh.k);
      sh.wei_pre = bmm(sh.q, kt);
      for (auto &v : sh.wei_pre.data)
            v *= scale;

      // Causal mask
      for (int b = 0; b < B; ++b)
            for (int i = 0; i < T; ++i)
                  for (int j = i + 1; j < T; ++j)
                        sh.wei_pre.at(b, i, j) = -1e30f;

      sh.wei = softmax3d(sh.wei_pre);

      sh.used_dropout = training && drop_p > 0.0f;
      Tensor wei_drop = sh.wei;
      if (sh.used_dropout)
      {
            sh.dropout_mask = Tensor(sh.wei.shape, 1.0f);
            float inv_keep = 1.0f / (1.0f - drop_p);
            std::bernoulli_distribution bd(1.0f - drop_p);
            for (int i = 0; i < sh.dropout_mask.numel(); ++i)
            {
                  bool kept = bd(rng);
                  sh.dropout_mask.data[i] = kept ? 1.0f : 0.0f;
                  wei_drop.data[i] = kept ? sh.wei.data[i] * inv_keep : 0.0f;
            }
      }

      return bmm(wei_drop, sh.v); // [B, T, head_size]
}

/**
 * @brief Multi-Head Attention forward with activation caching.
 *
 * @param x         [B, T, n_embd].
 * @param Wks       Per-head key weights. Size n_head.
 * @param Wqs       Per-head query weights. Size n_head.
 * @param Wvs       Per-head value weights. Size n_head.
 * @param Wp        [n_head * hs, n_embd]. Output projection weight.
 * @param bp        [n_embd]. Output projection bias.
 * @param n_head    Number of attention heads.
 * @param training  Enables dropout when true.
 * @param drop_p    Dropout probability.
 * @param rng       Thread-local MT19937.
 * @param sm        Output activation cache. Populated by this call.
 * @return          [B, T, n_embd]. New allocation.
 */
inline Tensor forward_mha_save(const Tensor &x, const std::vector<Tensor> &Wks,
                               const std::vector<Tensor> &Wqs, const std::vector<Tensor> &Wvs,
                               const Tensor &Wp, const Tensor &bp, int n_head, bool training,
                               float drop_p, std::mt19937 &rng, SavedMHA &sm)
{
      sm.heads.resize(n_head);
      std::vector<Tensor> head_outs(n_head);
      for (int h = 0; h < n_head; ++h)
            head_outs[h] =
                forward_head_save(x, Wks[h], Wqs[h], Wvs[h], training, drop_p, rng, sm.heads[h]);

      sm.concat = cat_last(head_outs); // [B, T, n_head * head_size]
      sm.proj_out = matmul(sm.concat, Wp);
      sm.proj_out = add_bias(sm.proj_out, bp);

      sm.used_dropout = training && drop_p > 0.0f;
      Tensor out = sm.proj_out;
      if (sm.used_dropout)
      {
            sm.dropout_mask = Tensor(out.shape, 1.0f);
            float inv_keep = 1.0f / (1.0f - drop_p);
            std::bernoulli_distribution bd(1.0f - drop_p);
            for (int i = 0; i < sm.dropout_mask.numel(); ++i)
            {
                  bool kept = bd(rng);
                  sm.dropout_mask.data[i] = kept ? 1.0f : 0.0f;
                  out.data[i] = kept ? out.data[i] * inv_keep : 0.0f;
            }
      }
      return out;
}

/**
 * @brief Feed-Forward Network forward with activation caching.
 *
 * @param x        [B, T, n_embd].
 * @param W1       [n_embd, 4 * n_embd]. FC1 weight.
 * @param b1       [4 * n_embd]. FC1 bias.
 * @param W2       [4 * n_embd, n_embd]. FC2 weight.
 * @param b2       [n_embd]. FC2 bias.
 * @param training Enables dropout when true.
 * @param drop_p   Dropout probability.
 * @param rng      Thread-local MT19937.
 * @param sf       Output activation cache. Populated by this call.
 * @return         [B, T, n_embd]. New allocation.
 */
inline Tensor forward_ffn_save(const Tensor &x, const Tensor &W1, const Tensor &b1,
                               const Tensor &W2, const Tensor &b2, bool training, float drop_p,
                               std::mt19937 &rng, SavedFFN &sf)
{
      sf.x = x;
      sf.h_pre = matmul(x, W1);
      sf.h_pre = add_bias(sf.h_pre, b1);
      sf.h = relu(sf.h_pre);
      sf.out_pre = matmul(sf.h, W2);
      sf.out_pre = add_bias(sf.out_pre, b2);

      sf.used_dropout = training && drop_p > 0.0f;
      Tensor out = sf.out_pre;
      if (sf.used_dropout)
      {
            sf.dropout_mask = Tensor(out.shape, 1.0f);
            float inv_keep = 1.0f / (1.0f - drop_p);
            std::bernoulli_distribution bd(1.0f - drop_p);
            for (int i = 0; i < sf.dropout_mask.numel(); ++i)
            {
                  bool kept = bd(rng);
                  sf.dropout_mask.data[i] = kept ? 1.0f : 0.0f;
                  out.data[i] = kept ? out.data[i] * inv_keep : 0.0f;
            }
      }
      return out;
}

#include "lm.h"

/**
 * @brief Full model forward pass with activation caching.
 *
 * @param model   GPT language model.
 * @param idx     [B * T]. Flat token indices.
 * @param B       Batch size.
 * @param T       Sequence length.
 * @param targets [B * T]. Ground-truth next-token indices.
 * @param training Enables dropout when true.
 * @return        SavedForward activation cache. New allocation.
 */
inline SavedForward forward_save(GPTLanguageModel &model, const std::vector<int> &idx, int B, int T,
                                 const std::vector<int> &targets, bool training)
{
      SavedForward s;
      s.idx = idx;
      s.B = B;
      s.T = T;
      s.targets = targets;
      int C = model.n_embd;
      int V = model.vocab_size;

      // Embeddings
      s.tok_out = model.token_emb.forward(idx, B, T);
      s.pos_out = model.pos_emb.forward_pos(T);
      s.emb_sum = Tensor({B, T, C});
      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < C; ++d)
                        s.emb_sum.at(b, t, d) = s.tok_out.at(b, t, d) + s.pos_out.at(0, t, d);

      // Transformer blocks
      s.blocks.resize(model.n_layer);
      Tensor x = s.emb_sum;
      for (int l = 0; l < model.n_layer; ++l)
      {
            auto &blk = model.blocks[l];
            auto &sb = s.blocks[l];
            sb.x_in = x;

            Tensor x_ln1 = forward_ln_save(x, blk.ln1.gamma, blk.ln1.beta, sb.ln1);

            int n_head = model.n_head;
            std::vector<Tensor> Wks(n_head), Wqs(n_head), Wvs(n_head);
            for (int h = 0; h < n_head; ++h)
            {
                  Wks[h] = blk.sa.heads[h].key.weight;
                  Wqs[h] = blk.sa.heads[h].query.weight;
                  Wvs[h] = blk.sa.heads[h].value.weight;
            }

            Tensor attn =
                forward_mha_save(x_ln1, Wks, Wqs, Wvs, blk.sa.proj.weight, blk.sa.proj.bias, n_head,
                                 training, DROPOUT, model.rng, sb.mha);
            sb.x_after_mha = add(x, attn);

            Tensor x_ln2 = forward_ln_save(sb.x_after_mha, blk.ln2.gamma, blk.ln2.beta, sb.ln2);
            Tensor ffn =
                forward_ffn_save(x_ln2, blk.ffwd.fc1.weight, blk.ffwd.fc1.bias, blk.ffwd.fc2.weight,
                                 blk.ffwd.fc2.bias, training, DROPOUT, model.rng, sb.ffn);
            x = add(sb.x_after_mha, ffn);
      }

      // Final norm + LM head
      s.lm_in = forward_ln_save(x, model.ln_f.gamma, model.ln_f.beta, s.ln_f);
      s.logits3d = matmul(s.lm_in, model.lm_head.weight);
      s.logits3d = add_bias(s.logits3d, model.lm_head.bias);

      s.logits2d = Tensor({B * T, V});
      for (int i = 0; i < B * T; ++i)
            for (int v = 0; v < V; ++v)
                  s.logits2d.at(i, v) = s.logits3d.data[i * V + v];

      return s;
}

/**
 * @brief Full model backward pass.
 *
 * Traverses layers in reverse topological order. Accumulates exact gradients
 * into the returned Grads structure.
 *
 * @param model GPT language model.
 * @param s     SavedForward activation cache from forward_save().
 * @return      Grads container with accumulated parameter gradients. New allocation.
 */
inline Grads backward(GPTLanguageModel &model, const SavedForward &s)
{
      int B = s.B, T = s.T;
      int C = model.n_embd, V = model.vocab_size;
      int n_head = model.n_head, hs = C / n_head;

      Grads g(V, C, n_head, model.n_layer, model.block_size);

      // Loss gradient w.r.t. logits
      Tensor dlogits2d = backward_cross_entropy(s.logits2d, s.targets);

      Tensor dlogits3d({B, T, V});
      for (int i = 0; i < B * T; ++i)
            for (int v = 0; v < V; ++v)
                  dlogits3d.data[i * V + v] = dlogits2d.at(i, v);

      // Output head backprop
      Tensor dx = backward_linear(dlogits3d, s.lm_in, model.lm_head.weight, g.lm_head);
      dx = backward_layernorm(dx, s.ln_f, model.ln_f.gamma, g.ln_f);

      // Reverse block backprop
      for (int l = model.n_layer - 1; l >= 0; --l)
      {
            auto &blk = model.blocks[l];
            auto &sb = s.blocks[l];
            auto &gb = g.blocks[l];

            // FFN branch
            Tensor dffn_out = dx;
            Tensor dffn = dffn_out;
            if (sb.ffn.used_dropout)
                  dffn = backward_dropout(dffn, sb.ffn.dropout_mask, DROPOUT);

            Tensor dh_relu = backward_linear(dffn, sb.ffn.h, blk.ffwd.fc2.weight, gb.ffwd.dfc2);
            Tensor dh_pre = backward_relu(dh_relu, sb.ffn.h_pre);
            Tensor dx_ln2 = backward_linear(dh_pre, sb.ffn.x, blk.ffwd.fc1.weight, gb.ffwd.dfc1);

            Tensor dx_after_mha_ffn = backward_layernorm(dx_ln2, sb.ln2, blk.ln2.gamma, gb.ln2);
            Tensor dx_after_mha = add(dx, dx_after_mha_ffn);

            // MHA branch
            Tensor dattn_out = dx_after_mha;
            Tensor dmha = dattn_out;
            if (sb.mha.used_dropout)
                  dmha = backward_dropout(dmha, sb.mha.dropout_mask, DROPOUT);

            Tensor dconcat = backward_linear(dmha, sb.mha.concat, blk.sa.proj.weight, gb.sa.proj);

            std::vector<int> head_sizes(n_head, hs);
            auto dhead_outs = backward_cat_last(dconcat, head_sizes);

            Tensor dx_ln1({B, T, C}, 0.0f);

            for (int h = 0; h < n_head; ++h)
            {
                  auto &sh = sb.mha.heads[h];
                  auto &gh = gb.sa.heads[h];
                  Tensor &dout_h = dhead_outs[h];

                  Tensor wei_used = sh.used_dropout
                                        ? Tensor(
                                              [&]()
                                              {
                                                    Tensor tmp = sh.wei;
                                                    float inv_keep = 1.0f / (1.0f - DROPOUT);
                                                    for (int i = 0; i < tmp.numel(); ++i)
                                                          tmp.data[i] = sh.dropout_mask.data[i] *
                                                                        sh.wei.data[i] * inv_keep;
                                                    return tmp;
                                              }())
                                        : sh.wei;

                  Tensor vT = transpose23(sh.v);
                  Tensor d_wei_drop = bmm(dout_h, vT);
                  Tensor dv = bmm(transpose23(sh.wei), dout_h);

                  Tensor d_wei = d_wei_drop;
                  if (sh.used_dropout)
                  {
                        float inv_keep = 1.0f / (1.0f - DROPOUT);
                        for (int i = 0; i < d_wei.numel(); ++i)
                              d_wei.data[i] *= sh.dropout_mask.data[i] * inv_keep;
                  }

                  // Zero causal upper-triangle
                  for (int b = 0; b < B; ++b)
                        for (int i = 0; i < T; ++i)
                              for (int j = i + 1; j < T; ++j)
                                    d_wei.at(b, i, j) = 0.0f;

                  Tensor d_wei_pre = backward_softmax3d(d_wei, sh.wei);

                  float scale = 1.0f / std::sqrt((float)hs);
                  for (auto &v : d_wei_pre.data)
                        v *= scale;

                  Tensor dq = bmm(d_wei_pre, sh.k);
                  Tensor dk = bmm(transpose23(d_wei_pre), sh.q);

                  Tensor dx_k = backward_linear(dk, sh.x, blk.sa.heads[h].key.weight, gh.dkey);
                  Tensor dx_q = backward_linear(dq, sh.x, blk.sa.heads[h].query.weight, gh.dquery);
                  Tensor dx_v = backward_linear(dv, sh.x, blk.sa.heads[h].value.weight, gh.dvalue);

                  for (int i = 0; i < dx_ln1.numel(); ++i)
                  {
                        dx_ln1.data[i] += dx_k.data[i] + dx_q.data[i] + dx_v.data[i];
                  }
            }

            Tensor dx_in_mha = backward_layernorm(dx_ln1, sb.ln1, blk.ln1.gamma, gb.ln1);
            dx = add(dx_after_mha, dx_in_mha);
      }

      // Embedding backprop
      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < C; ++d)
                        g.pos_emb.dW.at(t, d) += dx.at(b, t, d);

      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
            {
                  int tok = s.idx[b * T + t];
                  for (int d = 0; d < C; ++d)
                        g.tok_emb.dW.at(tok, d) += dx.at(b, t, d);
            }

      return g;
}

/**
 * @brief AdamW optimizer state (first and second moment buffers).
 */
struct AdamWState
{
      int step{0};
      float lr, beta1, beta2, eps;

      struct ParamState
      {
            std::vector<float> *param;
            std::vector<float> m, v;
      };
      std::vector<ParamState> states;

      /**
       * @brief Construct optimizer with hyperparameters.
       *
       * @param lr_  Learning rate.
       * @param b1   First moment decay.
       * @param b2   Second moment decay.
       * @param e    Epsilon for numerical stability.
       */
      AdamWState(float lr_ = 3e-4f, float b1 = 0.9f, float b2 = 0.999f, float e = 1e-8f)
          : lr(lr_), beta1(b1), beta2(b2), eps(e)
      {
      }

      /**
       * @brief Register a parameter vector for moment tracking.
       *
       * @param p Parameter vector. Pointer stored; caller owns memory.
       */
      void register_param(std::vector<float> &p)
      {
            states.push_back(
                {&p, std::vector<float>(p.size(), 0.0f), std::vector<float>(p.size(), 0.0f)});
      }

      /**
       * @brief Update a single registered parameter using its gradient.
       *
       * @param idx  Index into states vector.
       * @param grad Same-shape gradient tensor.
       */
      void update_one(int idx, const Tensor &grad)
      {
            auto &ps = states[idx];
            for (int i = 0; i < (int)ps.param->size(); ++i)
            {
                  float g = grad.data[i];
                  ps.m[i] = beta1 * ps.m[i] + (1.0f - beta1) * g;
                  ps.v[i] = beta2 * ps.v[i] + (1.0f - beta2) * g * g;

                  float mh = ps.m[i] / (1.0f - std::pow(beta1, step));
                  float vh = ps.v[i] / (1.0f - std::pow(beta2, step));

                  (*ps.param)[i] -= lr * mh / (std::sqrt(vh) + eps);
            }
      }
};

/**
 * @brief Apply accumulated gradients via AdamW update.
 *
 * @param model GPT language model.
 * @param g     Grads container with accumulated gradients.
 * @param opt   AdamWState. step incremented by this call.
 */
inline void apply_grads(GPTLanguageModel &model, const Grads &g, AdamWState &opt)
{
      opt.step++;
      int pi = 0;

      auto upd = [&](std::vector<float> &param, const Tensor &grad)
      {
            auto &ps = opt.states[pi++];
            assert(ps.param == &param);
            for (int i = 0; i < (int)param.size(); ++i)
            {
                  float gv = grad.data[i];
                  ps.m[i] = opt.beta1 * ps.m[i] + (1.0f - opt.beta1) * gv;
                  ps.v[i] = opt.beta2 * ps.v[i] + (1.0f - opt.beta2) * gv * gv;

                  float mh = ps.m[i] / (1.0f - std::pow(opt.beta1, opt.step));
                  float vh = ps.v[i] / (1.0f - std::pow(opt.beta2, opt.step));

                  param[i] -= opt.lr * mh / (std::sqrt(vh) + opt.eps);
            }
      };

      // Embeddings
      upd(model.token_emb.weight.data, g.tok_emb.dW);
      upd(model.pos_emb.weight.data, g.pos_emb.dW);

      // Blocks
      for (int l = 0; l < model.n_layer; ++l)
      {
            auto &blk = model.blocks[l];
            auto &gb = g.blocks[l];

            for (int h = 0; h < model.n_head; ++h)
            {
                  upd(blk.sa.heads[h].key.weight.data, gb.sa.heads[h].dkey.dW);
                  upd(blk.sa.heads[h].query.weight.data, gb.sa.heads[h].dquery.dW);
                  upd(blk.sa.heads[h].value.weight.data, gb.sa.heads[h].dvalue.dW);
            }
            upd(blk.sa.proj.weight.data, gb.sa.proj.dW);
            upd(blk.sa.proj.bias.data, gb.sa.proj.db);

            upd(blk.ffwd.fc1.weight.data, gb.ffwd.dfc1.dW);
            upd(blk.ffwd.fc1.bias.data, gb.ffwd.dfc1.db);
            upd(blk.ffwd.fc2.weight.data, gb.ffwd.dfc2.dW);
            upd(blk.ffwd.fc2.bias.data, gb.ffwd.dfc2.db);

            upd(blk.ln1.gamma.data, gb.ln1.dgamma);
            upd(blk.ln1.beta.data, gb.ln1.dbeta);
            upd(blk.ln2.gamma.data, gb.ln2.dgamma);
            upd(blk.ln2.beta.data, gb.ln2.dbeta);
      }

      // Final norm + LM head
      upd(model.ln_f.gamma.data, g.ln_f.dgamma);
      upd(model.ln_f.beta.data, g.ln_f.dbeta);
      upd(model.lm_head.weight.data, g.lm_head.dW);
      upd(model.lm_head.bias.data, g.lm_head.db);
}

/**
 * @brief Build AdamWState and register all model parameters.
 *
 * Must be called once during model initialization before training begins.
 *
 * @param model GPT language model.
 * @param lr    Learning rate.
 * @return      Configured AdamWState with all parameters registered.
 */
inline AdamWState build_optimizer(GPTLanguageModel &model, float lr)
{
      AdamWState opt(lr);

      opt.register_param(model.token_emb.weight.data);
      opt.register_param(model.pos_emb.weight.data);

      for (auto &blk : model.blocks)
      {
            for (auto &h : blk.sa.heads)
            {
                  opt.register_param(h.key.weight.data);
                  opt.register_param(h.query.weight.data);
                  opt.register_param(h.value.weight.data);
            }
            opt.register_param(blk.sa.proj.weight.data);
            opt.register_param(blk.sa.proj.bias.data);

            opt.register_param(blk.ffwd.fc1.weight.data);
            opt.register_param(blk.ffwd.fc1.bias.data);
            opt.register_param(blk.ffwd.fc2.weight.data);
            opt.register_param(blk.ffwd.fc2.bias.data);

            opt.register_param(blk.ln1.gamma.data);
            opt.register_param(blk.ln1.beta.data);
            opt.register_param(blk.ln2.gamma.data);
            opt.register_param(blk.ln2.beta.data);
      }

      opt.register_param(model.ln_f.gamma.data);
      opt.register_param(model.ln_f.beta.data);
      opt.register_param(model.lm_head.weight.data);
      opt.register_param(model.lm_head.bias.data);

      return opt;
}