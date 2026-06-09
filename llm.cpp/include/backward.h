#pragma once
// ============================================================
//  include/backward.h  –  Full analytical backpropagation
//  Each backward_* function:
//    - receives dOut  (gradient of loss w.r.t. this layer's output)
//    - receives saved activations from the forward pass
//    - fills gradients into Grad structs (mirroring weight layout)
//    - returns dX     (gradient of loss w.r.t. this layer's input)
// ============================================================

#include "config/config.h"
#include "tensor.h"

#include <cassert>
#include <cmath>
#include <vector>

// =============================================================
// Gradient stores (one per learnable weight tensour)
// ============================================================

struct GradLinear
{
      Tensor dW; // same shape as weight [in, out]
      Tensor db; // same shape as bias   [out]  (may be empty)
      bool has_bias;

      GradLinear() : has_bias(false) {}
      GradLinear(int in_f, int out_f, bool bias)
            : dW({in_f, out_f}, 0.0f), db(bias ? Tensor({out_f}, 0.0f) : Tensor()), has_bias(bias)
      {
      }

      void zero()
      {
            dW.fill(0.0f);
            if (has_bias)
                  db.fill(0.0f);
      }
};

struct GradEmbedding
{
      Tensor dW; // [num_embeddings, embedding_dim]
      GradEmbedding() = default;
      GradEmbedding(int n, int d) : dW({n, d}, 0.0f) {}
      void zero()
      {
            dW.fill(0.0f);
      }
};

struct GradLayerNorm
{
      Tensor dgamma; // [C]
      Tensor dbeta;  // [C]
      GradLayerNorm() = default;
      GradLayerNorm(int C) : dgamma({C}, 0.0f), dbeta({C}, 0.0f) {}
      void zero()
      {
            dgamma.fill(0.0f);
            dbeta.fill(0.0f);
      }
};

struct GradHead
{
      GradLinear dkey, dquery, dvalue;
      GradHead() = default;
      GradHead(int n_embd, int hs)
            : dkey(n_embd, hs, false), dquery(n_embd, hs, false), dvalue(n_embd, hs, false)
      {
      }
      void zero()
      {
            dkey.zero();
            dquery.zero();
            dvalue.zero();
      }
};

struct GradMHA
{
      std::vector<GradHead> heads;
      GradLinear proj;
      GradMHA() = default;
      GradMHA(int n_embd, int n_head, int hs) : proj(n_head * hs, n_embd, true)
      {
            for (int i = 0; i < n_head; ++i)
                  heads.emplace_back(n_embd, hs);
      }
      void zero()
      {
            for (auto &h : heads)
                  h.zero();
            proj.zero();
      }
};

struct GradFFN
{
      GradLinear dfc1, dfc2;
      GradFFN() = default;
      GradFFN(int n_embd) : dfc1(n_embd, 4 * n_embd, true), dfc2(4 * n_embd, n_embd, true) {}
      void zero()
      {
            dfc1.zero();
            dfc2.zero();
      }
};

struct GradBlock
{
      GradMHA sa;
      GradFFN ffwd;
      GradLayerNorm ln1, ln2;
      GradBlock() = default;
      GradBlock(int n_embd, int n_head, int hs)
            : sa(n_embd, n_head, hs), ffwd(n_embd), ln1(n_embd), ln2(n_embd)
      {
      }
      void zero()
      {
            sa.zero();
            ffwd.zero();
            ln1.zero();
            ln2.zero();
      }
};

struct Grads
{
      GradEmbedding tok_emb, pos_emb;
      std::vector<GradBlock> blocks;
      GradLayerNorm ln_f;
      GradLinear lm_head;

      Grads() = default;
      Grads(int vocab_size, int n_embd, int n_head, int n_layer, int block_size)
            : tok_emb(vocab_size, n_embd), pos_emb(block_size, n_embd), ln_f(n_embd),
              lm_head(n_embd, vocab_size, true)
      {
            int hs = n_embd / n_head;
            for (int i = 0; i < n_layer; ++i)
                  blocks.emplace_back(n_embd, n_head, hs);
      }

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

// ============================================================
// Saved activations from the forward pass
// (we need these to compute gradients)
// ============================================================

struct SavedHead
{
      Tensor x;            // input  [B, T, n_embd]
      Tensor k, q, v;      // key/query/value projections [B,T,hs]
      Tensor wei_pre;      // pre-softmax scores [B, T, T]
      Tensor wei;          // post-softmax weights [B, T, T]
      Tensor dropout_mask; // 1=kept, 0=zeroed  [B, T, T]  (or empty)
      bool used_dropout;
};

struct SavedMHA
{
      std::vector<SavedHead> heads;
      Tensor concat;       // [B, T, n_head*hs]  (after cat, before proj)
      Tensor proj_out;     // [B, T, n_embd]     (after proj, before dropout)
      Tensor dropout_mask; // proj dropout mask
      bool used_dropout;
};

struct SavedFFN
{
      Tensor x;       // input to fc1              [B, T, n_embd]
      Tensor h_pre;   // fc1 output before relu    [B, T, 4*n_embd]
      Tensor h;       // after relu                [B, T, 4*n_embd]
      Tensor out_pre; // fc2 output before dropout [B, T, n_embd]
      Tensor dropout_mask;
      bool used_dropout;
};

struct SavedLN
{
      Tensor x;                              // input  [B, T, C]
      Tensor xhat;                           // normalized, before gamma/beta [B,T,C]
      Tensor inv_std;                        // 1/sqrt(var+eps) per row  stored as [B, T, 1] → flat
      std::vector<float> mu_vec, invstd_vec; // [B*T] each
};

struct SavedBlock
{
      SavedLN ln1, ln2;
      SavedMHA mha;
      SavedFFN ffn;
      Tensor x_in;        // input to block          [B, T, C]
      Tensor x_after_mha; // after residual add      [B, T, C]
};

struct SavedForward
{
      // Embeddings
      std::vector<int> idx; // flat token indices B*T
      int B, T;
      Tensor tok_out; // [B, T, C]
      Tensor pos_out; // [1, T, C]
      Tensor emb_sum; // tok+pos  [B, T, C]

      // Blocks
      std::vector<SavedBlock> blocks;

      // Final LN + lm_head
      SavedLN ln_f;
      Tensor lm_in;    // input to lm_head  [B, T, C]
      Tensor logits3d; // [B, T, V]
      Tensor logits2d; // [B*T, V]
      std::vector<int> targets;
};

// ============================================================
// Primitive backward ops
// ============================================================

// ---- cross-entropy backward ---------------------------------
// dlogits [B*T, V]:  softmax(logits) - one_hot(targets)  / BT
inline Tensor backward_cross_entropy(const Tensor &logits2d, const std::vector<int> &targets)
{
      int BT = logits2d.shape[0], V = logits2d.shape[1];
      Tensor dlogits({BT, V}, 0.0f);
      for (int i = 0; i < BT; ++i)
      {
            // softmax
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
            // subtract one-hot
            dlogits.at(i, targets[i]) -= 1.0f;
            // scale by 1/BT
            for (int v = 0; v < V; ++v)
                  dlogits.at(i, v) /= (float)BT;
      }
      return dlogits;
}

// ---- linear backward  [B,T,out] → [B,T,in] -----------------
// dOut [B,T,E],  x [B,T,D],  W [D,E]  (bias grad trivially = sum over B,T)
// Returns dX [B,T,D];  accumulates into gW [D,E] and gb [E]
inline Tensor backward_linear(const Tensor &dOut, // [B, T, E]
                              const Tensor &x,    // [B, T, D]
                              const Tensor &W,    // [D, E]
                              GradLinear &g)
{
      int B = dOut.shape[0], T = dOut.shape[1], E = dOut.shape[2];
      int D = W.shape[0];
      assert(E == W.shape[1]);

      // dX = dOut @ W^T   [B, T, D]
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

      // dW += x^T @ dOut  accumulated [D, E]
      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < D; ++d)
                        for (int e = 0; e < E; ++e)
                              g.dW.at(d, e) += x.at(b, t, d) * dOut.at(b, t, e);

      // db += sum over B, T
      if (g.has_bias)
            for (int b = 0; b < B; ++b)
                  for (int t = 0; t < T; ++t)
                        for (int e = 0; e < E; ++e)
                              g.db.at(e) += dOut.at(b, t, e);

      return dX;
}

// ---- layer-norm backward  [B,T,C] → [B,T,C] ----------------
// Gradient of LayerNorm as derived in the original Ba et al. paper.
inline Tensor backward_layernorm(const Tensor &dOut, // [B, T, C]
                                 const SavedLN &saved,
                                 const Tensor &gamma, // [C]
                                 GradLayerNorm &g)
{
      int B = dOut.shape[0], T = dOut.shape[1], C = dOut.shape[2];
      Tensor dX({B, T, C}, 0.0f);

      for (int b = 0; b < B; ++b)
      {
            for (int t = 0; t < T; ++t)
            {
                  float inv_std = saved.invstd_vec[b * T + t];

                  // dgamma += dOut * xhat   (accumulated)
                  // dbeta  += dOut           (accumulated)
                  for (int c = 0; c < C; ++c)
                  {
                        float xhat_c = saved.xhat.at(b, t, c);
                        g.dgamma.at(c) += dOut.at(b, t, c) * xhat_c;
                        g.dbeta.at(c) += dOut.at(b, t, c);
                  }

                  // dX = (1/C) * inv_std * (
                  //       C * gamma * dOut
                  //     - sum(gamma * dOut)
                  //     - xhat * sum(gamma * dOut * xhat) )
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

// ---- ReLU backward ------------------------------------------
inline Tensor backward_relu(const Tensor &dOut, const Tensor &pre_relu)
{
      Tensor dX(dOut.shape);
      for (int i = 0; i < dOut.numel(); ++i)
            dX.data[i] = (pre_relu.data[i] > 0.0f) ? dOut.data[i] : 0.0f;
      return dX;
}

// ---- dropout backward (use same mask from forward) ----------
inline Tensor backward_dropout(const Tensor &dOut,
                               const Tensor &mask, // 1=kept, 0=zeroed
                               float p)
{
      if (p == 0.0f)
            return dOut;
      Tensor dX(dOut.shape);
      float inv_keep = 1.0f / (1.0f - p);
      for (int i = 0; i < dOut.numel(); ++i)
            dX.data[i] = dOut.data[i] * mask.data[i] * inv_keep;
      return dX;
}

// ---- batched matmul backward  [B,T,D] x [B,D,T2] → [B,T,T2] --
// da = dOut @ b^T,   db = a^T @ dOut   (both accumulated)
inline std::pair<Tensor, Tensor> backward_bmm(const Tensor &dOut, // [B, T, T2]
                                              const Tensor &a,    // [B, T,  D]
                                              const Tensor &b)
{ // [B, D, T2]
      int B = dOut.shape[0], T = dOut.shape[1], T2 = dOut.shape[2];
      int D = a.shape[2];
      Tensor da({B, T, D}, 0.0f);
      Tensor db({B, D, T2}, 0.0f);

      for (int bb = 0; bb < B; ++bb)
      {
            // da = dOut @ b^T:  da[b,t,d] += sum_{t2} dOut[b,t,t2]*b[b,d,t2]
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < D; ++d)
                  {
                        float s = 0.0f;
                        for (int t2 = 0; t2 < T2; ++t2)
                              s += dOut.at(bb, t, t2) * b.at(bb, d, t2);
                        da.at(bb, t, d) += s;
                  }
            // db = a^T @ dOut:  db[b,d,t2] += sum_{t} a[b,t,d]*dOut[b,t,t2]
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

// ---- softmax backward  [B,T,T] (attention weights) ----------
// d_pre = softmax_bwd(dwei, wei)
// For each row:  d_pre_i = s_i * (d_i - sum_j(s_j * d_j))
inline Tensor backward_softmax3d(const Tensor &dwei, // [B, T, T]
                                 const Tensor &wei)
{ // [B, T, T]
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

// ---- cat_last backward  [B,T,D_total] → slice per head ------
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

// ============================================================
// Forward pass WITH activation saving
// ============================================================

// ---- saved layernorm forward --------------------------------
inline Tensor forward_ln_save(const Tensor &x,
                              const Tensor &gamma,
                              const Tensor &beta,
                              SavedLN &saved,
                              float eps = 1e-5f)
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

// ---- saved single attention head forward --------------------
inline Tensor forward_head_save(const Tensor &x, // [B,T,n_embd]
                                const Tensor &Wk,
                                const Tensor &Wq,
                                const Tensor &Wv,
                                bool training,
                                float drop_p,
                                std::mt19937 &rng,
                                SavedHead &sh)
{
      int B = x.shape[0], T = x.shape[1], hs = Wk.shape[1];
      sh.x = x;
      sh.k = matmul(x, Wk); // [B,T,hs]
      sh.q = matmul(x, Wq);
      sh.v = matmul(x, Wv);

      float scale = 1.0f / std::sqrt((float)hs);

      // wei_pre = q @ k^T * scale  [B, T, T]
      Tensor kt = transpose23(sh.k);
      sh.wei_pre = bmm(sh.q, kt);
      for (auto &v : sh.wei_pre.data)
            v *= scale;

      // causal mask
      for (int b = 0; b < B; ++b)
            for (int i = 0; i < T; ++i)
                  for (int j = i + 1; j < T; ++j)
                        sh.wei_pre.at(b, i, j) = -1e30f;

      // softmax
      sh.wei = softmax3d(sh.wei_pre);

      // dropout on attention weights
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

      return bmm(wei_drop, sh.v); // [B,T,hs]
}

// ---- saved MHA forward --------------------------------------
inline Tensor forward_mha_save(const Tensor &x,
                               const std::vector<Tensor> &Wks,
                               const std::vector<Tensor> &Wqs,
                               const std::vector<Tensor> &Wvs,
                               const Tensor &Wp, // proj weight [n_head*hs, n_embd]
                               const Tensor &bp, // proj bias   [n_embd]
                               int n_head,
                               bool training,
                               float drop_p,
                               std::mt19937 &rng,
                               SavedMHA &sm)
{
      sm.heads.resize(n_head);
      std::vector<Tensor> head_outs(n_head);
      for (int h = 0; h < n_head; ++h)
            head_outs[h] =
                  forward_head_save(x, Wks[h], Wqs[h], Wvs[h], training, drop_p, rng, sm.heads[h]);

      sm.concat = cat_last(head_outs); // [B,T, n_head*hs]
      sm.proj_out = matmul(sm.concat, Wp);
      sm.proj_out = add_bias(sm.proj_out, bp);

      // projection dropout
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

// ---- saved FFN forward --------------------------------------
inline Tensor forward_ffn_save(const Tensor &x,
                               const Tensor &W1,
                               const Tensor &b1,
                               const Tensor &W2,
                               const Tensor &b2,
                               bool training,
                               float drop_p,
                               std::mt19937 &rng,
                               SavedFFN &sf)
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

// ============================================================
// Full model forward with all activations saved
// ============================================================
#include "quadtrix.h" // for GPTLanguageModel layout

inline SavedForward forward_save(GPTLanguageModel &model,
                                 const std::vector<int> &idx,
                                 int B,
                                 int T,
                                 const std::vector<int> &targets,
                                 bool training)
{
      SavedForward s;
      s.idx = idx;
      s.B = B;
      s.T = T;
      s.targets = targets;
      int C = model.n_embd;
      int V = model.vocab_size;

      // ── Embeddings ───────────────────────────────────────────
      s.tok_out = model.token_emb.forward(idx, B, T);
      s.pos_out = model.pos_emb.forward_pos(T);
      s.emb_sum = Tensor({B, T, C});
      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < C; ++d)
                        s.emb_sum.at(b, t, d) = s.tok_out.at(b, t, d) + s.pos_out.at(0, t, d);

      // ── Transformer blocks ───────────────────────────────────
      s.blocks.resize(model.n_layer);
      Tensor x = s.emb_sum;
      for (int l = 0; l < model.n_layer; ++l)
      {
            auto &blk = model.blocks[l];
            auto &sb = s.blocks[l];
            sb.x_in = x;

            // LN1 + MHA + residual
            Tensor x_ln1 = forward_ln_save(x, blk.ln1.gamma, blk.ln1.beta, sb.ln1);

            // Build weight vector views for each head
            int n_head = model.n_head;

            std::vector<Tensor> Wks(n_head), Wqs(n_head), Wvs(n_head);
            for (int h = 0; h < n_head; ++h)
            {
                  Wks[h] = blk.sa.heads[h].key.weight;
                  Wqs[h] = blk.sa.heads[h].query.weight;
                  Wvs[h] = blk.sa.heads[h].value.weight;
            }

            Tensor attn = forward_mha_save(x_ln1,
                                           Wks,
                                           Wqs,
                                           Wvs,
                                           blk.sa.proj.weight,
                                           blk.sa.proj.bias,
                                           n_head,
                                           training,
                                           DROPOUT,
                                           model.rng,
                                           sb.mha);
            sb.x_after_mha = add(x, attn); // residual

            // LN2 + FFN + residual
            Tensor x_ln2 = forward_ln_save(sb.x_after_mha, blk.ln2.gamma, blk.ln2.beta, sb.ln2);
            Tensor ffn = forward_ffn_save(x_ln2,
                                          blk.ffwd.fc1.weight,
                                          blk.ffwd.fc1.bias,
                                          blk.ffwd.fc2.weight,
                                          blk.ffwd.fc2.bias,
                                          training,
                                          DROPOUT,
                                          model.rng,
                                          sb.ffn);
            x = add(sb.x_after_mha, ffn);
      }

      // ── Final LN + lm_head ───────────────────────────────────
      s.lm_in = forward_ln_save(x, model.ln_f.gamma, model.ln_f.beta, s.ln_f);
      s.logits3d = matmul(s.lm_in, model.lm_head.weight);
      s.logits3d = add_bias(s.logits3d, model.lm_head.bias);

      // reshape [B,T,V] → [B*T, V]
      s.logits2d = Tensor({B * T, V});
      for (int i = 0; i < B * T; ++i)
            for (int v = 0; v < V; ++v)
                  s.logits2d.at(i, v) = s.logits3d.data[i * V + v];

      return s;
}

// ============================================================
// Full backward pass
// ============================================================

inline Grads backward(GPTLanguageModel &model, const SavedForward &s)
{
      int B = s.B, T = s.T;
      int C = model.n_embd, V = model.vocab_size;
      int n_head = model.n_head, hs = C / n_head;

      Grads g(V, C, n_head, model.n_layer, model.block_size);

      // ── dLoss / dLogits  [B*T, V] ────────────────────────────
      Tensor dlogits2d = backward_cross_entropy(s.logits2d, s.targets);

      // reshape to [B, T, V]
      Tensor dlogits3d({B, T, V});
      for (int i = 0; i < B * T; ++i)
            for (int v = 0; v < V; ++v)
                  dlogits3d.data[i * V + v] = dlogits2d.at(i, v);

      // ── lm_head backward  [B,T,V] → [B,T,C] ──────────────────
      Tensor dx = backward_linear(dlogits3d, s.lm_in, model.lm_head.weight, g.lm_head);

      // ── final layernorm backward ──────────────────────────────
      dx = backward_layernorm(dx, s.ln_f, model.ln_f.gamma, g.ln_f);

      // ── transformer blocks (reverse order) ───────────────────
      for (int l = model.n_layer - 1; l >= 0; --l)
      {
            auto &blk = model.blocks[l];
            auto &sb = s.blocks[l];
            auto &gb = g.blocks[l];

            // ---- FFN residual: dx is d(x_after_mha + ffn_out) ──
            // dffn = dx  (residual, pass-through)
            // dx  += dx  (will be added after LN2 backward below)
            Tensor dffn_out = dx; // gradient to the ffn output branch
            // residual adds straight through:
            // d(x_after_mha) gets dx directly (accumulated below)

            // ---- LN2 + FFN backward ─────────────────────────────
            // dffn_out → (dropout bwd) → fc2 bwd → relu bwd → fc1 bwd → dx_ln2
            Tensor dffn = dffn_out;
            if (sb.ffn.used_dropout)
                  dffn = backward_dropout(dffn, sb.ffn.dropout_mask, DROPOUT);

            // fc2 backward
            Tensor dh_relu = backward_linear(dffn, sb.ffn.h, blk.ffwd.fc2.weight, gb.ffwd.dfc2);
            // relu backward
            Tensor dh_pre = backward_relu(dh_relu, sb.ffn.h_pre);
            // fc1 backward
            Tensor dx_ln2 = backward_linear(dh_pre, sb.ffn.x, blk.ffwd.fc1.weight, gb.ffwd.dfc1);

            // LN2 backward → d(x_after_mha) from FFN branch
            Tensor dx_after_mha_ffn = backward_layernorm(dx_ln2, sb.ln2, blk.ln2.gamma, gb.ln2);
            // total d(x_after_mha) = residual-pass + LN2 branch
            Tensor dx_after_mha = add(dx, dx_after_mha_ffn);

            // ---- MHA residual: x_after_mha = x_in + attn_out ───
            Tensor dattn_out = dx_after_mha; // gradient to attn branch
            // d(x_in) from this residual = dx_after_mha (passed through)

            // ---- MHA backward ────────────────────────────────────
            Tensor dmha = dattn_out;
            if (sb.mha.used_dropout)
                  dmha = backward_dropout(dmha, sb.mha.dropout_mask, DROPOUT);

            // proj backward  [B,T,n_embd] → [B,T,n_head*hs]
            Tensor dconcat = backward_linear(dmha, sb.mha.concat, blk.sa.proj.weight, gb.sa.proj);

            // split concat grad back to each head
            std::vector<int> head_sizes(n_head, hs);
            auto dhead_outs = backward_cat_last(dconcat, head_sizes);

            // input grad accumulator
            Tensor dx_ln1({B, T, C}, 0.0f);

            for (int h = 0; h < n_head; ++h)
            {
                  auto &sh = sb.mha.heads[h];
                  auto &gh = gb.sa.heads[h];

                  // dhead_outs[h]:  [B, T, hs]  = d(wei_drop @ v)
                  Tensor &dout_h = dhead_outs[h];

                  // attention dropout on wei
                  Tensor dwei_drop = dout_h; // placeholder — we need d(wei_drop @ v)
                  // Actually: out_h = wei_drop @ v
                  // d(wei_drop) = dout_h @ v^T   [B,T,T]
                  // d(v)        = wei_drop^T @ dout_h  [B,T,hs]
                  Tensor wei_used =
                        sh.used_dropout ? Tensor(
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

                  // d(out_h) = dhead_outs[h]
                  // out_h = wei_used @ v   →  backward of bmm
                  //   da = dOut @ b^T  →  d_wei_used = dout_h @ v^T  [B,T,T]
                  //   db = a^T @ dOut  →  dv         = wei_used^T @ dout_h [B,T,hs]
                  Tensor vT = transpose23(sh.v); // [B, hs, T]
                  // d_wei_drop [B,T,T] = dout_h @ vT  (but vT is [B,hs,T], need [B,T,hs]@... )
                  // Use bmm with v transposed
                  //   dout_h [B,T,hs], v [B,T,hs]  →  vT [B,hs,T]
                  //   d_wei_drop = bmm(dout_h, vT)  = [B,T,hs]@[B,hs,T] = [B,T,T]
                  Tensor d_wei_drop = bmm(dout_h, vT); // [B,T,T]

                  // dv = wei_used^T @ dout_h  = [B,T,T]^T @ [B,T,hs] = [B,T,hs]
                  Tensor wei_usedT = transpose23(wei_used); // [B, T, T] transposed → [B, T, T]
                  // bmm needs [B,T,T] x [B,T,hs]:  wei_usedT [B,T,T] x dout_h [B,T,hs]
                  // but bmm signature is [B,T,D]x[B,D,T2]
                  // wei_usedT as [B,T,T] x dout_h [B,T,hs]: first transpose wei_used to get [B,T,T]
                  // correct: dv = bmm(wei_used^T, dout_h) where wei_used^T = [B,T,T] transposed
                  // back
                  Tensor dv = bmm(transpose23(sh.wei), dout_h); // [B,T,hs]

                  // attention dropout backward on d_wei_drop → d_wei
                  Tensor d_wei = d_wei_drop;
                  if (sh.used_dropout)
                  {
                        float inv_keep = 1.0f / (1.0f - DROPOUT);
                        for (int i = 0; i < d_wei.numel(); ++i)
                              d_wei.data[i] *= sh.dropout_mask.data[i] * inv_keep;
                  }

                  // causal mask backward: zero out upper-triangle grads (they were -inf, grad=0)
                  for (int b = 0; b < B; ++b)
                        for (int i = 0; i < T; ++i)
                              for (int j = i + 1; j < T; ++j)
                                    d_wei.at(b, i, j) = 0.0f;

                  // softmax backward
                  Tensor d_wei_pre = backward_softmax3d(d_wei, sh.wei);

                  // scale backward (multiply by same 1/sqrt(hs))
                  float scale = 1.0f / std::sqrt((float)hs);
                  for (auto &v : d_wei_pre.data)
                        v *= scale;

                  // d_wei_pre = dq @ kT + q @ dkT (product rule of q@k^T)
                  // d_wei_pre [B,T,T], q [B,T,hs], k [B,T,hs]
                  // dq  = d_wei_pre @ k           [B,T,T]@[B,T,hs]... need k [B,hs,T]^T = k
                  // actual: d_wei_pre[b,i,j] = sum over... it's q[b,i,:] · k[b,j,:]
                  // dq[b,i,d] = sum_j d_wei_pre[b,i,j] * k[b,j,d]
                  //   = bmm(d_wei_pre, k)  where k is [B,T,hs] → need [B,T,hs] directly
                  //   = bmm with b=[B,hs,T]? No: bmm([B,T,T], [B,T,hs]) needs second arg [B,T,T]
                  // Use: dq = d_wei_pre @ k  with k as [B, hs, T]... no.
                  // dq[b,i,d] = sum_j d_pre[b,i,j] * k[b,j,d]  → this IS bmm(d_pre, k) if
                  // k were [B,T,hs]... but bmm expects [B,D,T2].  So treat as matmul style:
                  // bmm(d_pre [B,T,T], k [B,T,hs]) — but bmm signature needs [B,D,T2]:
                  // interpret as B batches, each [T,T] @ [T,hs] = [T,hs]:  b's D=T, T2=hs → valid!
                  Tensor dq = bmm(d_wei_pre, sh.k); // [B,T,hs]

                  // dk[b,j,d] = sum_i d_pre[b,i,j] * q[b,i,d]
                  //   = (d_pre^T @ q)[b,j,d] = bmm(transpose23(d_pre), q)  [B,T,T]@[B,T,hs]
                  Tensor dk = bmm(transpose23(d_wei_pre), sh.q); // [B,T,hs]

                  // Now project back through key/query/value linear layers (no bias)
                  // dX_from_k = dk @ Wk^T  etc.  Use backward_linear with dummy GradLinear
                  // but we need separate grads:
                  // dWk += xln1^T @ dk  (accumulated)
                  // similar for Wq, Wv
                  // and dx_ln1 += dk @ Wk^T + dq @ Wq^T + dv @ Wv^T

                  // Key
                  Tensor dx_k = backward_linear(dk, sh.x, blk.sa.heads[h].key.weight, gh.dkey);
                  Tensor dx_q = backward_linear(dq, sh.x, blk.sa.heads[h].query.weight, gh.dquery);
                  Tensor dx_v = backward_linear(dv, sh.x, blk.sa.heads[h].value.weight, gh.dvalue);

                  // accumulate into dx_ln1
                  for (int i = 0; i < dx_ln1.numel(); ++i)
                  {
                        dx_ln1.data[i] += dx_k.data[i] + dx_q.data[i] + dx_v.data[i];
                  }
            }

            // LN1 backward → d(x_in)
            Tensor dx_in_mha = backward_layernorm(dx_ln1, sb.ln1, blk.ln1.gamma, gb.ln1);

            // total dx for this block: residual pass-through + LN1/MHA branch + LN2/FFN branch
            // d(x_in) = d(x_after_mha) [residual from MHA] + d(x_in) [from MHA path]
            // d(x_in) = dx_after_mha (pass-through of FFN residual) + dx_in_mha
            dx = add(dx_after_mha, dx_in_mha);
      }

      // ── Embedding backward ───────────────────────────────────
      // dx is now d(emb_sum) = d(tok_emb + pos_emb)
      // pos_emb grad: sum over batch
      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < C; ++d)
                        g.pos_emb.dW.at(t, d) += dx.at(b, t, d);

      // tok_emb grad: scatter into vocab rows
      for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
            {
                  int tok = s.idx[b * T + t];
                  for (int d = 0; d < C; ++d)
                        g.tok_emb.dW.at(tok, d) += dx.at(b, t, d);
            }

      return g;
}

// ============================================================
// Apply Grads → model weights  (AdamW update)
// ============================================================

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

      AdamWState(float lr_ = 3e-4f, float b1 = 0.9f, float b2 = 0.999f, float e = 1e-8f)
            : lr(lr_), beta1(b1), beta2(b2), eps(e)
      {
      }

      void register_param(std::vector<float> &p)
      {
            states.push_back(
                  {&p, std::vector<float>(p.size(), 0.0f), std::vector<float>(p.size(), 0.0f)});
      }

      // Update one param tensor from its gradient tensor
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

      upd(model.token_emb.weight.data, g.tok_emb.dW);
      upd(model.pos_emb.weight.data, g.pos_emb.dW);

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
      upd(model.ln_f.gamma.data, g.ln_f.dgamma);
      upd(model.ln_f.beta.data, g.ln_f.dbeta);
      upd(model.lm_head.weight.data, g.lm_head.dW);
      upd(model.lm_head.bias.data, g.lm_head.db);
}

// Build AdamWState from model params (call once before training)
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