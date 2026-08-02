// @Eamon2009
/*// -------------------------------------------------------
//  overview and some NOTES
//-----------------------------------------------------------
//
// This header file has a dual-mode Byte-Pair Encoding
// (BPE) tokenizer and dataset loader (DataLoader)  for repo llm.cpp
// train pipeline. It supports raw text corpus training and
// memory-mapped (mmap) binary shards also training.

// 1. text Mode: Reads raw text files, trains a BPE tokenizer from scratch up to a
//    target vocabulary size, caches vocabularies to disk (.tokenizer.bin), and
//    splits tokens into contiguous train/validation memory vectors.
// 2. SHARDED Mode: Bypasses heavy RAM overhead by using  memory-mapping
//    (POSIX mmap / Win32 File Mapping) to stream pre-tokenized binary shards (.bin).
//    Shards utilize a fixed 1KB header followed by a flat stream of uint16_t
//    token IDs, capping vocabulary support at 65,536 entries.
// notes:
// * Platform-Specific Memory Mapping: look `MMapShard::open()` for platform
//    branches. Windows uses CreateFileMappingA/MapViewOfFile, while POSIX systems
//    rely on open(), mmap(), and madvise(MADV_RANDOM) to optimize non-sequential
//    batch sampling hints for kerenels
// * Move-Only Semantics: `MMapShard` implements explicit move-only semantics
//    (`= delete` on copy constructors/operators) to safely manage file descriptors
//    and resource handles without double-free errors
// * Cache-Aligned Index Structures: `BPEIndex` is explicitly alignas(16) to ensure
//    optimal L1/L2 cache line utilization during fast doubly-linked list traversal
//    in BPE merging routine
// * Packed Hash Keys: Pair keys for merge ranks are packed into a single uint64_t
//    using bit shifts (`((uint64_t)left << 32) | (uint32_t)right`) to optimize
//    hash map lookups (`std::unordered_map`)
// * Parallelized Batch Sampling: OpenMP (`#pragma omp parallel for`) is leveraged
//    across batch sampling (`get_batch_text` and `get_batch_sharded`) and base encoding
//    to parallelize random number generation and index lookups safely using thread-local
//    random engines (std::mt19937)

Shard Capacity Limit: Because shards serialize tokens as `uint16_t`,
vocabularies exceeding 65,536 entries will cause a hard runtime exception
(`write_shards` check). Do not alter token datatype widths without redesigning
the header structure (`SHARD_HEADER_INTS`).

Boundary Conditions in Sharded Sampling: `get_batch_sharded` contains both a
fast path (contiguous memory inside a single shard) and a rare path (blocks
crossing shard boundaries via prefix-sum lookups using `std::upper_bound`).
Changes to indexing logic must maintain safety against off-by-one segment faults.

Mutability and Thread Safety: While data streaming via mmap is read-only and
inherently thread-safe across multiple worker threads, ensure that external
callers do not concurrently mutate vocabulary mappings (`token_to_id`, `vocab`)
while inference/encoding or training loops are executing.
// ----------------------------------------==--------------------------------------*/

#pragma once
#include "config/config.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Conditionally include OpenMP so it compiles on compilers without native OpenMP support
#ifdef _OPENMP
#include <omp.h>
#endif

// Directory scanning / mkdir via std::filesystem -- portable across
// Linux/macOS/Windows, no platform #ifdef needed for this part.
#include <filesystem>

// Memory-mapping is platform-specific: POSIX mmap vs Win32 file mappings.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX // Prevent Windows macros from breaking std::min / std::max
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// Shard binary format : fixed 1KB header followed by a flat
// stream of uint16_t token ids. uint16_t caps vocab at 65536 entries, which
// covers essentially every BPE config people run (32k/50k/64k) at half the
// footprint of int32 -- shards are twice as fast to page in and to scan.
static constexpr int32_t SHARD_MAGIC = 20240801;
static constexpr int32_t SHARD_VERSION = 1;
static constexpr size_t SHARD_HEADER_INTS = 256; // 1024 bytes, room to grow
static constexpr size_t SHARD_HEADER_BYTES = SHARD_HEADER_INTS * sizeof(int32_t);
#ifndef SHARD_SIZE_TOKENS
#define SHARD_SIZE_TOKENS (100ull * 1000ull * 1000ull) // 100M tokens/shard (~200MB)
#endif

struct DataLoader
{
      enum class DataMode
      {
            TEXT,   // raw .txt -> BPE-trained, fully in RAM (small/medium datasets)
            SHARDED // directory of pre-tokenized .bin shards, mmap-streamed
      };

      std::vector<std::string> vocab;
      std::unordered_map<std::string, int> token_to_id;
      std::vector<std::pair<int, int>> merges;

      // Packed uint64_t key: ((uint64_t)left << 32) | (uint32_t)right
      std::unordered_map<uint64_t, int> merge_rank;

      int base_vocab_size{0};
      int vocab_size{0};

      DataMode mode{DataMode::TEXT};

      std::vector<int> train_data; // populated only in TEXT mode
      std::vector<int> val_data;   // populated only in TEXT mode

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
      // mmap'd shard: one .bin file, read-only, paged in lazily by the OS.
      // Move-only (owns an fd + mapping).

      struct MMapShard
      {
#ifdef _WIN32
            HANDLE hFile{INVALID_HANDLE_VALUE};
            HANDLE hMap{nullptr};
#else
            int fd{-1};
#endif
            void *map_base{nullptr};
            size_t map_size{0};
            const uint16_t *data{nullptr};
            uint64_t num_tokens{0};
            std::string path;

            MMapShard() = default;
            MMapShard(const MMapShard &) = delete;
            MMapShard &operator=(const MMapShard &) = delete;

            MMapShard(MMapShard &&o) noexcept { *this = std::move(o); }
            MMapShard &operator=(MMapShard &&o) noexcept
            {
                  if (this != &o)
                  {
                        release();
#ifdef _WIN32
                        hFile = o.hFile;
                        hMap = o.hMap;
                        o.hFile = INVALID_HANDLE_VALUE;
                        o.hMap = nullptr;
#else
                        fd = o.fd;
                        o.fd = -1;
#endif
                        map_base = o.map_base;
                        map_size = o.map_size;
                        data = o.data;
                        num_tokens = o.num_tokens;
                        path = std::move(o.path);
                        o.map_base = nullptr;
                        o.map_size = 0;
                        o.data = nullptr;
                        o.num_tokens = 0;
                  }
                  return *this;
            }

            ~MMapShard() { release(); }

            void release()
            {
#ifdef _WIN32
                  if (map_base)
                        UnmapViewOfFile(map_base);
                  if (hMap)
                        CloseHandle(hMap);
                  if (hFile != INVALID_HANDLE_VALUE)
                        CloseHandle(hFile);
                  hMap = nullptr;
                  hFile = INVALID_HANDLE_VALUE;
#else
                  if (map_base && map_base != MAP_FAILED)
                        munmap(map_base, map_size);
                  if (fd >= 0)
                        ::close(fd);
                  fd = -1;
#endif
                  map_base = nullptr;
            }

            void open(const std::string &p)
            {
                  path = p;

#ifdef _WIN32
                  hFile = CreateFileA(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                  if (hFile == INVALID_HANDLE_VALUE)
                        throw std::runtime_error("[Shard] Cannot open: " + p);

                  LARGE_INTEGER size{};
                  if (!GetFileSizeEx(hFile, &size))
                        throw std::runtime_error("[Shard] GetFileSizeEx failed: " + p);
                  map_size = (size_t)size.QuadPart;
                  if (map_size <= SHARD_HEADER_BYTES)
                        throw std::runtime_error("[Shard] File too small to be a valid shard: " +
                                                 p);

                  hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
                  if (!hMap)
                        throw std::runtime_error("[Shard] CreateFileMapping failed: " + p);

                  map_base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
                  if (!map_base)
                        throw std::runtime_error("[Shard] MapViewOfFile failed: " + p);
#else
                  fd = ::open(p.c_str(), O_RDONLY);
                  if (fd < 0)
                        throw std::runtime_error("[Shard] Cannot open: " + p);

                  struct stat st{};
                  if (fstat(fd, &st) != 0)
                        throw std::runtime_error("[Shard] fstat failed: " + p);
                  map_size = (size_t)st.st_size;
                  if (map_size <= SHARD_HEADER_BYTES)
                        throw std::runtime_error("[Shard] File too small to be a valid shard: " +
                                                 p);

                  map_base = mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
                  if (map_base == MAP_FAILED)
                        throw std::runtime_error("[Shard] mmap failed: " + p);

                  madvise(map_base, map_size, MADV_RANDOM);
#endif

                  const int32_t *header = reinterpret_cast<const int32_t *>(map_base);
                  if (header[0] != SHARD_MAGIC)
                        throw std::runtime_error("[Shard] Bad magic (not a shard file): " + p);
                  if (header[1] != SHARD_VERSION)
                        throw std::runtime_error("[Shard] Unsupported shard version: " + p);

                  uint64_t lo = (uint32_t)header[3];
                  uint64_t hi = (uint32_t)header[4];
                  num_tokens = (hi << 32) | lo;

                  data = reinterpret_cast<const uint16_t *>(
                      reinterpret_cast<const char *>(map_base) + SHARD_HEADER_BYTES);

                  uint64_t expected_bytes = SHARD_HEADER_BYTES + num_tokens * sizeof(uint16_t);
                  if (expected_bytes != map_size)
                        throw std::runtime_error("[Shard] Size/header mismatch in: " + p);
            }
      };

      struct ShardedSplit
      {
            std::vector<MMapShard> shards;
            std::vector<uint64_t> prefix;
            uint64_t total_tokens{0};

            void add(const std::string &path)
            {
                  MMapShard s;
                  s.open(path);
                  total_tokens += s.num_tokens;
                  shards.push_back(std::move(s));
            }

            void build_prefix()
            {
                  prefix.assign(shards.size() + 1, 0);
                  for (size_t i = 0; i < shards.size(); ++i)
                        prefix[i + 1] = prefix[i] + shards[i].num_tokens;
            }

            inline size_t locate(uint64_t global_idx) const
            {
                  auto it = std::upper_bound(prefix.begin(), prefix.end(), global_idx);
                  return (size_t)(it - prefix.begin()) - 1;
            }

            inline int token_at(uint64_t global_idx) const
            {
                  size_t s = locate(global_idx);
                  uint64_t local = global_idx - prefix[s];
                  return (int)shards[s].data[local];
            }
      };

      ShardedSplit shard_train;
      ShardedSplit shard_val;

      void load(const std::string &path, int target_vocab = BPE_VOCAB_SIZE,
                double train_split = TRAIN_SPLIT)
      {
            if (is_directory(path))
            {
                  load_shards(path);
            }
            else
            {
                  load_text(path, target_vocab, train_split);
            }
      }

      void load_text(const std::string &path, int target_vocab = BPE_VOCAB_SIZE,
                     double train_split = TRAIN_SPLIT)
      {
            mode = DataMode::TEXT;

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

            std::vector<int> data;
            std::string cache_path = path + ".tokenizer.bin";
            if (load_vocab_if_matches(cache_path, target_vocab))
            {
                  std::cout << "[BPE]   Loaded cached vocab from " << cache_path
                            << " (skipping BPE training)\n";
                  std::cout.flush();
                  data = apply_merges(base_encode(text));
            }
            else
            {
                  data = train_bpe(text, target_vocab);
                  save_vocab(cache_path);
                  std::cout << "[BPE]   Cached vocab to " << cache_path << "\n";
            }

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

      void load_shards(const std::string &dir)
      {
            mode = DataMode::SHARDED;

            std::vector<std::string> train_files = list_files_matching(dir, "train_", ".bin");
            std::vector<std::string> val_files = list_files_matching(dir, "val_", ".bin");

            if (train_files.empty())
                  throw std::runtime_error("[DataLoader] No train_*.bin shards found in: " + dir);
            if (val_files.empty())
                  throw std::runtime_error("[DataLoader] No val_*.bin shards found in: " + dir);

            std::sort(train_files.begin(), train_files.end());
            std::sort(val_files.begin(), val_files.end());

            for (auto &p : train_files)
                  shard_train.add(p);
            for (auto &p : val_files)
                  shard_val.add(p);

            shard_train.build_prefix();
            shard_val.build_prefix();

            std::string vocab_path = (fs::path(dir) / "tokenizer.bin").string();
            std::ifstream check(vocab_path, std::ios::binary);
            if (check.good())
            {
                  check.close();
                  load_vocab(vocab_path);
                  std::cout << "[SHARD] Loaded tokenizer vocab from " << vocab_path << "\n";
            }
            else
            {
                  std::cout << "[SHARD] Warning: no tokenizer.bin found in " << dir
                            << " -- encode()/decode() unavailable, training ids only.\n";
            }

            std::cout << "[SHARD] Train shards : " << shard_train.shards.size() << "   ("
                      << shard_train.total_tokens << " tokens)\n";
            std::cout << "[SHARD] Val shards   : " << shard_val.shards.size() << "   ("
                      << shard_val.total_tokens << " tokens)\n";
            std::cout.flush();

            if (shard_train.total_tokens <= (uint64_t)BLOCK_SIZE ||
                shard_val.total_tokens <= (uint64_t)BLOCK_SIZE)
                  throw std::runtime_error(
                      "[DataLoader] Sharded dataset too small for BLOCK_SIZE=" +
                      std::to_string(BLOCK_SIZE));
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
            if (mode == DataMode::SHARDED)
                  return get_batch_sharded(split, batch_size, block_size, rng);
            return get_batch_text(split, batch_size, block_size, rng);
      }

      void write_shards(const std::string &out_dir,
                        uint64_t shard_size_tokens = SHARD_SIZE_TOKENS) const
      {
            if (vocab.empty())
                  throw std::runtime_error(
                      "[SHARD] No vocab trained/loaded -- call load_text() first");
            if (vocab.size() > 65536)
                  throw std::runtime_error("[SHARD] vocab_size " + std::to_string(vocab.size()) +
                                           " exceeds uint16_t shard capacity (65536)");
            if (train_data.empty() || val_data.empty())
                  throw std::runtime_error(
                      "[SHARD] No train/val data in memory -- call load_text() first");

            make_directory(out_dir);
            write_split_shards(out_dir, "train", train_data, shard_size_tokens);
            write_split_shards(out_dir, "val", val_data, shard_size_tokens);
            save_vocab((fs::path(out_dir) / "tokenizer.bin").string());

            std::cout << "[SHARD] Wrote shards + tokenizer.bin to " << out_dir << "\n";
      }

      void save_vocab(const std::string &path) const
      {
            std::ofstream f(path, std::ios::binary);
            if (!f.is_open())
                  throw std::runtime_error("[Vocab] Cannot write: " + path);

            uint32_t magic = 0x544B5631; // 'TKV1'
            write_u32(f, magic);
            write_u32(f, (uint32_t)base_vocab_size);
            write_u32(f, (uint32_t)vocab_size);
            write_u32(f, (uint32_t)vocab.size());
            for (const auto &tok : vocab)
            {
                  write_u32(f, (uint32_t)tok.size());
                  f.write(tok.data(), (std::streamsize)tok.size());
            }
            write_u32(f, (uint32_t)merges.size());
            for (const auto &m : merges)
            {
                  write_u32(f, (uint32_t)m.first);
                  write_u32(f, (uint32_t)m.second);
            }
      }

      void load_vocab(const std::string &path)
      {
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open())
                  throw std::runtime_error("[Vocab] Cannot read: " + path);

            uint32_t magic = read_u32(f);
            if (magic != 0x544B5631)
                  throw std::runtime_error("[Vocab] Bad magic (not a tokenizer.bin): " + path);

            base_vocab_size = (int)read_u32(f);
            vocab_size = (int)read_u32(f);

            uint32_t n_vocab = read_u32(f);
            vocab.clear();
            token_to_id.clear();
            vocab.reserve(n_vocab);
            for (uint32_t i = 0; i < n_vocab; ++i)
            {
                  uint32_t len = read_u32(f);
                  std::string tok(len, '\0');
                  f.read(&tok[0], (std::streamsize)len);
                  token_to_id[tok] = (int)vocab.size();
                  vocab.push_back(std::move(tok));
            }

            uint32_t n_merges = read_u32(f);
            merges.clear();
            merge_rank.clear();
            merges.reserve(n_merges);
            for (uint32_t i = 0; i < n_merges; ++i)
            {
                  int left = (int)read_u32(f);
                  int right = (int)read_u32(f);
                  merges.push_back({left, right});
                  merge_rank[make_pair_key(left, right)] = (int)i;
            }
      }

    private:
      static void write_u32(std::ofstream &f, uint32_t v)
      {
            f.write(reinterpret_cast<const char *>(&v), sizeof(v));
      }
      static uint32_t read_u32(std::ifstream &f)
      {
            uint32_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            return v;
      }

      bool load_vocab_if_matches(const std::string &cache_path, int target_vocab)
      {
            std::ifstream check(cache_path, std::ios::binary);
            if (!check.good())
                  return false;
            check.close();
            try
            {
                  load_vocab(cache_path);
            }
            catch (const std::exception &)
            {
                  return false;
            }
            return vocab_size == target_vocab;
      }

      static bool is_directory(const std::string &path)
      {
            std::error_code ec;
            return fs::is_directory(path, ec);
      }

      static void make_directory(const std::string &path)
      {
            std::error_code ec;
            fs::create_directories(path, ec);
            if (ec && !fs::is_directory(path))
                  throw std::runtime_error("[SHARD] Cannot create directory: " + path + " (" +
                                           ec.message() + ")");
      }

      static std::vector<std::string> list_files_matching(const std::string &dir,
                                                          const std::string &prefix,
                                                          const std::string &suffix)
      {
            std::vector<std::string> out;
            std::error_code ec;
            fs::directory_iterator it(dir, ec);
            if (ec)
                  throw std::runtime_error("[DataLoader] Cannot open directory: " + dir);

            for (const auto &entry : it)
            {
                  if (!entry.is_regular_file())
                        continue;
                  std::string name = entry.path().filename().string();
                  bool has_prefix = name.compare(0, prefix.size(), prefix) == 0;
                  bool has_suffix =
                      name.size() >= suffix.size() &&
                      name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
                  if (has_prefix && has_suffix)
                        out.push_back(entry.path().string());
            }
            return out;
      }

      void write_split_shards(const std::string &out_dir, const std::string &split_name,
                              const std::vector<int> &ids, uint64_t shard_size_tokens) const
      {
            uint64_t total = ids.size();
            uint64_t shard_idx = 0;
            for (uint64_t offset = 0; offset < total; offset += shard_size_tokens, ++shard_idx)
            {
                  uint64_t count =
                      (std::min)(shard_size_tokens,
                                 std::min<uint64_t>(shard_size_tokens, total - offset));
                  char name_buf[64];
                  std::snprintf(name_buf, sizeof(name_buf), "%s_%06llu.bin", split_name.c_str(),
                                (unsigned long long)shard_idx);

                  std::string path = (fs::path(out_dir) / name_buf).string();

                  std::ofstream f(path, std::ios::binary);
                  if (!f.is_open())
                        throw std::runtime_error("[SHARD] Cannot write: " + path);

                  int32_t header[SHARD_HEADER_INTS] = {0};
                  header[0] = SHARD_MAGIC;
                  header[1] = SHARD_VERSION;
                  header[2] = 0;
                  header[3] = (int32_t)(uint32_t)(count & 0xFFFFFFFFull);
                  header[4] = (int32_t)(uint32_t)(count >> 32);
                  f.write(reinterpret_cast<const char *>(header), sizeof(header));

                  std::vector<uint16_t> buf(count);
                  for (uint64_t i = 0; i < count; ++i)
                        buf[i] = (uint16_t)ids[offset + i];
                  f.write(reinterpret_cast<const char *>(buf.data()),
                          (std::streamsize)(count * sizeof(uint16_t)));
            }
      }

      std::pair<std::vector<int>, std::vector<int>> get_batch_text(const std::string &split,
                                                                   int batch_size, int block_size,
                                                                   std::mt19937 &rng) const
      {
            const std::vector<int> &d = (split == "train") ? train_data : val_data;
            std::uniform_int_distribution<int> dist(0, (int)d.size() - block_size - 1);

            std::vector<int> x(batch_size * block_size);
            std::vector<int> y(batch_size * block_size);

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

      std::pair<std::vector<int>, std::vector<int>> get_batch_sharded(const std::string &split,
                                                                      int batch_size,
                                                                      int block_size,
                                                                      std::mt19937 &rng) const
      {
            const ShardedSplit &s = (split == "train") ? shard_train : shard_val;
            std::uniform_int_distribution<uint64_t> dist(0, s.total_tokens - block_size - 1);

            std::vector<int> x(batch_size * block_size);
            std::vector<int> y(batch_size * block_size);

            std::vector<uint32_t> seeds(batch_size);
            for (int b = 0; b < batch_size; ++b)
                  seeds[b] = rng();

#pragma omp parallel for schedule(static)
            for (int b = 0; b < batch_size; ++b)
            {
                  std::mt19937 thread_rng(seeds[b]);
                  uint64_t start = dist(thread_rng);

                  size_t shard_i = s.locate(start);
                  uint64_t local = start - s.prefix[shard_i];
                  const MMapShard &shard = s.shards[shard_i];

                  if (local + (uint64_t)block_size + 1 <= shard.num_tokens)
                  {
                        for (int t = 0; t < block_size; ++t)
                        {
                              x[b * block_size + t] = (int)shard.data[local + t];
                              y[b * block_size + t] = (int)shard.data[local + t + 1];
                        }
                  }
                  else
                  {
                        for (int t = 0; t < block_size; ++t)
                        {
                              x[b * block_size + t] = s.token_at(start + t);
                              y[b * block_size + t] = s.token_at(start + t + 1);
                        }
                  }
            }
            return {x, y};
      }

      std::vector<int> base_encode(const std::string &text) const
      {
            std::vector<int> ids(text.size());

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

            using PQItem = std::pair<int, int>;
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

                  for (const auto &[key, pos_vec] : pair_pos)
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