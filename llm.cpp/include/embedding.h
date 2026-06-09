#pragma once
// ============================================================
//  include/embedding.h  –  Lookup-table embedding
//  Mirrors: nn.Embedding(num_embeddings, embedding_dim)
// ============================================================

#include "tensor.h"
#include <fstream>
#include <vector>

struct Embedding
{
      int num_embeddings;
      int embedding_dim;
      Tensor weight; // [num_embeddings, embedding_dim]

      Embedding() = default;

      Embedding(int num_emb, int emb_dim, std::mt19937 &rng)
          : num_embeddings(num_emb), embedding_dim(emb_dim)
      {
            weight = Tensor::randn({num_emb, emb_dim}, 0.0f, 0.02f, rng);
      }

      // idx: flat vector of token indices, length B*T
      // returns Tensor [B, T, emb_dim]
      Tensor forward(const std::vector<int> &idx, int B, int T) const
      {
            Tensor out({B, T, embedding_dim});
            for (int b = 0; b < B; ++b)
                  for (int t = 0; t < T; ++t)
                  {
                        int token = idx[b * T + t];
                        for (int d = 0; d < embedding_dim; ++d)
                              out.at(b, t, d) = weight.at(token, d);
                  }
            return out;
      }

      // position embedding: idx = 0,1,...,T-1  to  [1, T, emb_dim]
      Tensor forward_pos(int T) const
      {
            Tensor out({1, T, embedding_dim});
            for (int t = 0; t < T; ++t)
                  for (int d = 0; d < embedding_dim; ++d)
                        out.at(0, t, d) = weight.at(t, d);
            return out;
      }

      int num_params() const { return weight.numel(); }

      void save(std::ofstream &f) const
      {
            f.write(reinterpret_cast<const char *>(weight.data.data()),
                    weight.numel() * sizeof(float));
      }
      void load(std::ifstream &f)
      {
            f.read(reinterpret_cast<char *>(weight.data.data()),
                   weight.numel() * sizeof(float));
      }
};