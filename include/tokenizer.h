// @Eamon2009
#pragma once
#include "config/config.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct DataLoader
{
      std::vector<std::string> vocab;
      std::unordered_map<std::string, int> token_to_id;
      std::vector<std::pair<int, int>> merges;

      // Packed uint64_t key: ((uint64_t)left << 32) | (uint32_t)right
      std::unordered_map<uint64_t, int> merge_rank;

      int base_vocab_size{0};
      int vocab_size{0};

      std::vector<int> train_data;
      std::vector<int> val_data;

      // 16-byte aligned struct for optimal L1/L2 cache line packing
      struct alignas(16) BPEIndex
      {
            int id;
            int prev;
            int next;
            int active; // 1 = valid, 0 = merged/removed
      };

      static inline uint64_t make_pair_key(int left, int right)
      {
            return ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
      }

      void load(const std::string &path,
                int target_vocab = BPE_VOCAB_SIZE,
                double train_split = TRAIN_SPLIT)
      {
            std::ifstream f(path);
            if (!f.is_open())
                  throw std::runtime_error("[DataLoader] Cannot open file: " + path);

            std::ostringstream ss;
            ss << f.rdbuf();
            std::string text = ss.str();
            if (text.empty())
                  throw std::runtime_error("[DataLoader] File is empty: " + path);

            std::cout << "[BPE]   Text length: " << text.size() << " characters\n";
            std::cout << "[BPE]   Target vocab size: " << target_vocab << "\n";
            std::cout.flush();

            std::vector<int> data = train_bpe(text, target_vocab);

            int n = (int)(train_split * (double)data.size());
            train_data = std::vector<int>(data.begin(), data.begin() + n);
            val_data = std::vector<int>(data.begin() + n, data.end());

            if ((int)train_data.size() <= BLOCK_SIZE || (int)val_data.size() <= BLOCK_SIZE)
                  throw std::runtime_error("[DataLoader] Dataset too small for BLOCK_SIZE=" +
                                           std::to_string(BLOCK_SIZE));

            std::cout << "[DATA]  Total tokens : " << data.size() << "\n";
            std::cout << "[DATA]  Train tokens : " << train_data.size() << "\n";
            std::cout << "[DATA]  Val tokens   : " << val_data.size() << "\n";
      }

      std::vector<int> encode(const std::string &text) const
      {
            return apply_merges(base_encode(text));
      }

      std::string decode(const std::vector<int> &ids) const
      {
            std::string out;
            out.reserve(ids.size() * 2);
            for (int id : ids)
                  if (id >= 0 && id < (int)vocab.size())
                        out += vocab[id];
            return out;
      }

      std::pair<std::vector<int>, std::vector<int>>
      get_batch(const std::string &split, int batch_size, int block_size, std::mt19937 &rng) const
      {
            const std::vector<int> &d = (split == "train") ? train_data : val_data;
            std::uniform_int_distribution<int> dist(0, (int)d.size() - block_size - 1);

            std::vector<int> x(batch_size * block_size);
            std::vector<int> y(batch_size * block_size);

            // Generate deterministic seeds for parallel threads
            std::vector<uint32_t> seeds(batch_size);
            for (int b = 0; b < batch_size; ++b)
                  seeds[b] = rng();

#pragma omp parallel for schedule(static)
            for (int b = 0; b < batch_size; ++b)
            {
                  std::mt19937 thread_rng(seeds[b]);
                  int start = dist(thread_rng);
                  for (int t = 0; t < block_size; ++t)
                  {
                        x[b * block_size + t] = d[start + t];
                        y[b * block_size + t] = d[start + t + 1];
                  }
            }
            return {x, y};
      }

    private:
      std::vector<int> base_encode(const std::string &text) const
      {
            std::vector<int> ids(text.size());

            // Parallel character mapping for large text buffers
            if (text.size() > 10000)
            {
#pragma omp parallel for schedule(static)
                  for (size_t i = 0; i < text.size(); ++i)
                  {
                        auto it = token_to_id.find(std::string(1, text[i]));
                        ids[i] = (it != token_to_id.end()) ? it->second : -1;
                  }
                  ids.erase(std::remove(ids.begin(), ids.end(), -1), ids.end());
            }
            else
            {
                  ids.clear();
                  ids.reserve(text.size());
                  for (char c : text)
                  {
                        auto it = token_to_id.find(std::string(1, c));
                        if (it != token_to_id.end())
                              ids.push_back(it->second);
                  }
            }
            return ids;
      }

      std::vector<int> apply_merges(std::vector<int> ids) const
      {
            if (ids.size() < 2 || merges.empty())
                  return ids;

            int n = (int)ids.size();
            std::vector<BPEIndex> nodes(n);
            for (int i = 0; i < n; ++i)
                  nodes[i] = {ids[i], i - 1, i + 1, 1};
            nodes[n - 1].next = -1;

            // Priority queue to merge pairs in lowest rank order first (O(N log N))
            using PQItem = std::pair<int, int>; // {rank, left_index}
            std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;

            auto try_enqueue = [&](int left_idx)
            {
                  if (left_idx == -1 || !nodes[left_idx].active)
                        return;
                  int right_idx = nodes[left_idx].next;
                  if (right_idx == -1 || !nodes[right_idx].active)
                        return;

                  uint64_t key = make_pair_key(nodes[left_idx].id, nodes[right_idx].id);
                  auto it = merge_rank.find(key);
                  if (it != merge_rank.end())
                        pq.push({it->second, left_idx});
            };

            for (int i = 0; i < n - 1; ++i)
                  try_enqueue(i);

            while (!pq.empty())
            {
                  auto [rank, left_idx] = pq.top();
                  pq.pop();

                  if (!nodes[left_idx].active)
                        continue;
                  int right_idx = nodes[left_idx].next;
                  if (right_idx == -1 || !nodes[right_idx].active)
                        continue;

                  uint64_t key = make_pair_key(nodes[left_idx].id, nodes[right_idx].id);
                  auto it = merge_rank.find(key);
                  if (it == merge_rank.end() || it->second != rank)
                        continue;

                  int new_id = base_vocab_size + rank;
                  int prev_idx = nodes[left_idx].prev;
                  int next_next_idx = nodes[right_idx].next;

                  nodes[left_idx].id = new_id;
                  nodes[left_idx].next = next_next_idx;
                  nodes[right_idx].active = 0;

                  if (next_next_idx != -1)
                        nodes[next_next_idx].prev = left_idx;

                  try_enqueue(prev_idx);
                  try_enqueue(left_idx);
            }

            std::vector<int> out;
            out.reserve(n);
            int curr = 0;
            while (curr != -1)
            {
                  if (nodes[curr].active)
                        out.push_back(nodes[curr].id);
                  curr = nodes[curr].next;
            }
            return out;
      }

      std::vector<int> train_bpe(const std::string &text, int target_vocab)
      {
            std::set<char> chars(text.begin(), text.end());
            for (char c : chars)
            {
                  std::string s(1, c);
                  token_to_id[s] = (int)vocab.size();
                  vocab.push_back(s);
            }
            base_vocab_size = (int)vocab.size();
            if (target_vocab <= base_vocab_size)
            {
                  vocab_size = base_vocab_size;
                  return base_encode(text);
            }

            std::vector<int> initial_ids = base_encode(text);
            int n = (int)initial_ids.size();

            std::vector<BPEIndex> list(n);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
                  list[i] = {initial_ids[i], i - 1, i + 1, 1};
            if (n > 0)
                  list[n - 1].next = -1;

            // Inverted index mapping pair key -> list of indices where it occurs
            std::unordered_map<uint64_t, std::vector<int>> pair_pos;
            pair_pos.reserve(n);

            for (int i = 0; i < n - 1; ++i)
            {
                  uint64_t key = make_pair_key(list[i].id, list[i + 1].id);
                  pair_pos[key].push_back(i);
            }

            int num_merges = target_vocab - base_vocab_size;
            merges.reserve(num_merges);
            std::cout << "[BPE]   Running " << num_merges << " merges...\n";
            std::cout.flush();

            for (int step = 0; step < num_merges; ++step)
            {
                  uint64_t best_key = 0;
                  size_t max_count = 0;

                  // Scan unique key pairs to find max frequency
                  for (auto const &[key, pos_vec] : pair_pos)
                  {
                        if (pos_vec.size() > max_count)
                        {
                              max_count = pos_vec.size();
                              best_key = key;
                        }
                  }

                  if (max_count == 0)
                        break;

                  int left = (int)(best_key >> 32);
                  int right = (int)(best_key & 0xFFFFFFFF);

                  int new_id = (int)vocab.size();
                  vocab.push_back(vocab[left] + vocab[right]);
                  token_to_id[vocab.back()] = new_id;
                  merges.push_back({left, right});
                  merge_rank[best_key] = step;

                  auto occs = std::move(pair_pos[best_key]);
                  pair_pos.erase(best_key);

                  // Update only actual locations where the pair appears
                  for (int head : occs)
                  {
                        if (!list[head].active || list[head].id != left)
                              continue;
                        int next_node = list[head].next;
                        if (next_node == -1 || !list[next_node].active ||
                            list[next_node].id != right)
                              continue;

                        int prev_node = list[head].prev;
                        int next_next_node = list[next_node].next;

                        list[next_node].active = 0;
                        list[head].id = new_id;
                        list[head].next = next_next_node;

                        if (next_next_node != -1)
                              list[next_next_node].prev = head;

                        if (prev_node != -1 && list[prev_node].active)
                        {
                              uint64_t new_left_key = make_pair_key(list[prev_node].id, new_id);
                              pair_pos[new_left_key].push_back(prev_node);
                        }
                        if (next_next_node != -1 && list[next_next_node].active)
                        {
                              uint64_t new_right_key =
                                    make_pair_key(new_id, list[next_next_node].id);
                              pair_pos[new_right_key].push_back(head);
                        }
                  }

                  if ((step + 1) % 100 == 0 || step + 1 == num_merges)
                  {
                        std::cout << "[BPE]   step " << (step + 1) << "/" << num_merges
                                  << "   vocab=" << (int)vocab.size() << "\n";
                        std::cout.flush();
                  }
            }

            std::vector<int> final_ids;
            final_ids.reserve(n);
            int curr = 0;
            while (curr != -1)
            {
                  if (list[curr].active)
                        final_ids.push_back(list[curr].id);
                  curr = list[curr].next;
            }

            vocab_size = (int)vocab.size();
            std::cout << "[BPE]   Done. Final vocab size: " << vocab_size << "\n";
            return final_ids;
      }
};