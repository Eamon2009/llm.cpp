#pragma once
// ============================================================
//  include/dataloader.h  –  Character-level tokeniser & batching
//  Mirrors the data-loading section of the Python script
// ============================================================

#include "tensor.h"
#include "config/config.h"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

struct DataLoader
{
      std::string text;
      std::vector<char> chars;  // sorted vocabulary
      std::map<char, int> stoi; // char → index
      std::map<int, char> itos; // index → char
      int vocab_size{0};

      std::vector<int> train_data;
      std::vector<int> val_data;

      // ---- load and prepare ----------------------------------------
      void load(const std::string &path, double train_split = TRAIN_SPLIT)
      {
            std::ifstream f(path);
            if (!f.is_open())
                  throw std::runtime_error("[DataLoader] Cannot open file: " + path);

            std::ostringstream ss;
            ss << f.rdbuf();
            text = ss.str();
            if (text.empty())
                  throw std::runtime_error("[DataLoader] File is empty: " + path);

            // build vocabulary
            std::set<char> charset(text.begin(), text.end());
            chars = std::vector<char>(charset.begin(), charset.end());
            std::sort(chars.begin(), chars.end());
            vocab_size = (int)chars.size();

            stoi.clear();
            itos.clear();
            for (int i = 0; i < vocab_size; ++i)
            {
                  stoi[chars[i]] = i;
                  itos[i] = chars[i];
            }

            // encode
            std::vector<int> data;
            data.reserve(text.size());
            for (char c : text)
                  data.push_back(stoi.at(c));

            int n = (int)(train_split * data.size());
            train_data = std::vector<int>(data.begin(), data.begin() + n);
            val_data = std::vector<int>(data.begin() + n, data.end());

            if (train_data.size() <= (size_t)BLOCK_SIZE ||
                val_data.size() <= (size_t)BLOCK_SIZE)
                  throw std::runtime_error("[DataLoader] Dataset is too small for the configured BLOCK_SIZE=" +
                                           std::to_string(BLOCK_SIZE) + ". Need more than " +
                                           std::to_string(BLOCK_SIZE + 1) + " tokens in both train and validation splits.");

            std::cout << "[DATA]  Total characters : " << text.size() << "\n";
            std::cout << "[DATA]  Vocabulary size  : " << vocab_size << "\n";
            std::cout << "[DATA]  Train tokens     : " << train_data.size() << "\n";
            std::cout << "[DATA]  Val   tokens     : " << val_data.size() << "\n";
      }

      // ---- encode / decode -----------------------------------------
      std::vector<int> encode(const std::string &s) const
      {
            std::vector<int> out;
            out.reserve(s.size());
            for (char c : s)
            {
                  auto it = stoi.find(c);
                  if (it != stoi.end())
                        out.push_back(it->second);
            }
            return out;
      }

      std::string decode(const std::vector<int> &ids) const
      {
            std::string out;
            out.reserve(ids.size());
            for (int id : ids)
            {
                  auto it = itos.find(id);
                  if (it != itos.end())
                        out += it->second;
            }
            return out;
      }

      // ---- batch sampler -------------------------------------------
      //  Returns (x, y) as flat integer vectors of size B*T
      std::pair<std::vector<int>, std::vector<int>>
      get_batch(const std::string &split, int batch_size, int block_size,
                std::mt19937 &rng) const
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
};
