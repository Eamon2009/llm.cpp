// @Eamon2009
#pragma once
#include "config/config.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct DataLoader
{
      std::vector<std::string> vocab;
      std::map<std::string, int> token_to_id;
      std::vector<std::pair<int, int>> merges;
      std::map<std::pair<int, int>, int> merge_rank;
      int base_vocab_size{0};
      int vocab_size{0};

      std::vector<int> train_data;
      std::vector<int> val_data;

      struct BPEIndex
      {
            int id;
            int prev;
            int next;
      };

      struct PairFreq
      {
            long long key;
            int count;
            bool operator<(const PairFreq &o) const
            {
                  return count < o.count;
            }
      };

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
            for (int b = 0; b < batch_size; ++b)
            {
                  int start = dist(rng);
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
            std::vector<int> ids;
            ids.reserve(text.size());
            for (char c : text)
            {
                  auto it = token_to_id.find(std::string(1, c));
                  if (it != token_to_id.end())
                        ids.push_back(it->second);
            }
            return ids;
      }

      std::vector<int> apply_merges(std::vector<int> ids) const
      {
            for (int rank = 0; rank < (int)merges.size(); ++rank)
            {
                  int left = merges[rank].first;
                  int right = merges[rank].second;
                  int new_id = base_vocab_size + rank;
                  std::vector<int> out;
                  out.reserve(ids.size());
                  int i = 0;
                  while (i < (int)ids.size())
                  {
                        if (i + 1 < (int)ids.size() && ids[i] == left && ids[i + 1] == right)
                        {
                              out.push_back(new_id);
                              i += 2;
                        }
                        else
                        {
                              out.push_back(ids[i]);
                              ++i;
                        }
                  }
                  ids = std::move(out);
            }
            return ids;
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
            int n = initial_ids.size();

            std::vector<BPEIndex> list(n);
            for (int i = 0; i < n; ++i)
                  list[i] = {initial_ids[i], i - 1, i + 1};
            list[n - 1].next = -1;

            // Direct mapping tracking instead of nested maps
            // Uses flat vectors to keep data compact in RAM for older CPUs
            std::map<long long, int> pair_counts;
            auto get_key = [](int left, int right) -> long long
            { return ((long long)left << 32) | (unsigned int)right; };

            for (int i = 0; i < n - 1; ++i)
            {
                  pair_counts[get_key(list[i].id, list[i + 1].id)]++;
            }

            int num_merges = target_vocab - base_vocab_size;
            merges.reserve(num_merges);
            std::cout << "[BPE]   Running " << num_merges << " merges...\n";
            std::cout.flush();

            for (int step = 0; step < num_merges; ++step)
            {
                  long long best_key = 0;
                  int max_count = 0;

                  // Old CPUs read flat contiguous structures incredibly fast
                  for (auto const &[key, count] : pair_counts)
                  {
                        if (count > max_count)
                        {
                              max_count = count;
                              best_key = key;
                        }
                  }

                  if (max_count == 0)
                        break;

                  int left = (int)(best_key >> 32);
                  int right = (int)(best_key & 0xFFFFFFFFLL);

                  int new_id = (int)vocab.size();
                  vocab.push_back(vocab[left] + vocab[right]);
                  token_to_id[vocab.back()] = new_id;
                  merges.push_back({left, right});
                  merge_rank[{left, right}] = step;

                  // Clear out targeted count map entry
                  pair_counts.erase(best_key);

                  // Single pass linked list pointer update
                  int curr = 0;
                  while (curr != -1 && list[curr].next != -1)
                  {
                        int next_node = list[curr].next;
                        if (list[curr].id == left && list[next_node].id == right)
                        {
                              int prev_node = list[curr].prev;
                              int next_next_node = list[next_node].next;

                              // Decrement counts for pairs broken by this merge
                              if (prev_node != -1)
                                    pair_counts[get_key(list[prev_node].id, list[curr].id)]--;
                              if (next_next_node != -1)
                                    pair_counts[get_key(list[next_node].id,
                                                        list[next_next_node].id)]--;

                              // Mutate list nodes
                              list[curr].id = new_id;
                              list[curr].next = next_next_node;
                              if (next_next_node != -1)
                                    list[next_next_node].prev = curr;

                              // Increment counts for newly created pairs
                              if (prev_node != -1)
                                    pair_counts[get_key(list[prev_node].id, list[curr].id)]++;
                              if (next_next_node != -1)
                                    pair_counts[get_key(list[curr].id, list[next_next_node].id)]++;

                              curr = next_next_node;
                        }
                        else
                        {
                              curr = next_node;
                        }
                  }

                  // Clean out zero records to prevent lookups from bloating memory
                  for (auto it = pair_counts.begin(); it != pair_counts.end();)
                  {
                        if (it->second <= 0)
                              it = pair_counts.erase(it);
                        else
                              ++it;
                  }

                  if ((step + 1) % 100 == 0 || step + 1 == num_merges)
                  {
                        std::cout << "[BPE]   step " << (step + 1) << "/" << num_merges
                                  << "   vocab=" << (int)vocab.size() << "\n";
                        std::cout.flush();
                  }
            }

            std::vector<int> final_ids;
            int curr = 0;
            while (curr != -1)
            {
                  final_ids.push_back(list[curr].id);
                  curr = list[curr].next;
            }

            vocab_size = (int)vocab.size();
            std::cout << "[BPE]   Done. Final vocab size: " << vocab_size << "\n";
            return final_ids;
      }
};