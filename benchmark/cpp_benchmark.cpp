// Real C++ benchmark suite for Quadtrix.cpp.
//
// Build:
//   g++ -std=c++17 -O3 -DNDEBUG -I. benchmark/cpp_benchmark.cpp -o benchmark/quadtrix_cpp_bench
//
// Run:
//   benchmark/quadtrix_cpp_bench --quick
//   benchmark/quadtrix_cpp_bench --data data/input.txt --model best_model.bin

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "../config/config.h"
#include "../include/backward.h"
#include "../include/dataloader.h"
#include "../include/gpt.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#endif

struct Options
{
      std::string data_path = DEFAULT_CLEANED_PATH;
      std::string model_path = BEST_MODEL_PATH;
      std::string out_dir = "benchmark/results";
      int runs = 10;
      int warmup = 3;
      int batch_size = BATCH_SIZE;
      int train_steps = 3;
      int generate_tokens = 32;
      int max_data_chars = 0;
      bool quick = false;
      bool random_weights = false;
};

struct Stats
{
      double avg_ms{0.0}, median_ms{0.0}, min_ms{0.0}, max_ms{0.0};
      double p90_ms{0.0}, p95_ms{0.0}, std_ms{0.0};
};

struct BenchRow
{
      std::string suite;
      std::string name;
      int batch_size{0};
      int sequence_length{0};
      int tokens{0};
      Stats stats;
      double tokens_per_sec{0.0};
      int samples{0};
      double loss{0.0};
      bool has_loss{false};
      double memory_mb{0.0};
      std::string notes;
};

static double wall_ms()
{
      using namespace std::chrono;
      return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static bool file_exists(const std::string &path)
{
      std::ifstream f(path.c_str(), std::ios::binary);
      return f.good();
}

static std::string now_iso()
{
      std::time_t t = std::time(nullptr);
      char buf[40];
      std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
      return buf;
}

static std::string json_escape(const std::string &s)
{
      std::ostringstream out;
      for (char ch : s)
      {
            switch (ch)
            {
            case '\\':
                  out << "\\\\";
                  break;
            case '"':
                  out << "\\\"";
                  break;
            case '\n':
                  out << "\\n";
                  break;
            case '\r':
                  out << "\\r";
                  break;
            case '\t':
                  out << "\\t";
                  break;
            default:
                  out << ch;
            }
      }
      return out.str();
}

static double percentile(std::vector<double> values, double pct)
{
      if (values.empty())
            return 0.0;
      std::sort(values.begin(), values.end());
      double pos = (values.size() - 1) * pct;
      int lo = (int)std::floor(pos);
      int hi = (int)std::ceil(pos);
      if (lo == hi)
            return values[lo];
      return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}

static Stats summarize(const std::vector<double> &samples)
{
      Stats s;
      std::vector<double> sorted = samples;
      std::sort(sorted.begin(), sorted.end());
      s.avg_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
      s.median_ms = sorted[sorted.size() / 2];
      if (sorted.size() % 2 == 0)
            s.median_ms = 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
      s.min_ms = sorted.front();
      s.max_ms = sorted.back();
      s.p90_ms = percentile(samples, 0.90);
      s.p95_ms = percentile(samples, 0.95);
      double sq = 0.0;
      for (double x : samples)
            sq += (x - s.avg_ms) * (x - s.avg_ms);
      s.std_ms = std::sqrt(sq / samples.size());
      return s;
}

template <typename Fn>
static std::pair<Stats, double> run_timed(int runs, int warmup, Fn fn)
{
      double sink = 0.0;
      for (int i = 0; i < warmup; ++i)
            sink += fn();
      std::vector<double> samples;
      samples.reserve(runs);
      for (int i = 0; i < runs; ++i)
      {
            double t0 = wall_ms();
            sink += fn();
            samples.push_back(wall_ms() - t0);
      }
      volatile double keep = sink;
      (void)keep;
      return {summarize(samples), sink};
}

static double estimated_model_mb(const GPTLanguageModel &model)
{
      return (double)model.num_params() * sizeof(float) / (1024.0 * 1024.0);
}

static void push_row(std::vector<BenchRow> &rows, const BenchRow &row)
{
      rows.push_back(row);
      const BenchRow &r = rows.back();
      std::cout << std::left << std::setw(14) << r.suite
                << std::setw(24) << r.name
                << "avg=" << std::right << std::setw(9) << std::fixed << std::setprecision(3)
                << r.stats.avg_ms << " ms  p95=" << std::setw(9)
                << r.stats.p95_ms << " ms  tok/s=" << std::setw(10)
                << std::setprecision(1) << r.tokens_per_sec << "\n";
}

static void prepare_dataloader_from_text(DataLoader &dl, const std::string &text, double train_split)
{
      if (text.empty())
            throw std::runtime_error("[Benchmark] Input text is empty.");

      dl.text = text;
      std::set<char> charset(text.begin(), text.end());
      dl.chars = std::vector<char>(charset.begin(), charset.end());
      std::sort(dl.chars.begin(), dl.chars.end());
      dl.vocab_size = (int)dl.chars.size();
      dl.stoi.clear();
      dl.itos.clear();
      for (int i = 0; i < dl.vocab_size; ++i)
      {
            dl.stoi[dl.chars[i]] = i;
            dl.itos[i] = dl.chars[i];
      }

      std::vector<int> data;
      data.reserve(text.size());
      for (char c : text)
            data.push_back(dl.stoi.at(c));

      int n = (int)(train_split * data.size());
      dl.train_data = std::vector<int>(data.begin(), data.begin() + n);
      dl.val_data = std::vector<int>(data.begin() + n, data.end());
      if (dl.train_data.size() <= (size_t)BLOCK_SIZE || dl.val_data.size() <= (size_t)BLOCK_SIZE)
            throw std::runtime_error("[Benchmark] Capped dataset is too small for BLOCK_SIZE.");
}

static void load_data(const Options &opt, DataLoader &dl)
{
      if (opt.max_data_chars <= 0)
      {
            dl.load(opt.data_path);
            return;
      }

      std::ifstream f(opt.data_path.c_str(), std::ios::binary);
      if (!f.is_open())
            throw std::runtime_error("[Benchmark] Cannot open file: " + opt.data_path);

      std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      if ((int)text.size() > opt.max_data_chars)
            text.resize(opt.max_data_chars);
      prepare_dataloader_from_text(dl, text, TRAIN_SPLIT);
      std::cout << "[DATA]  Total characters : " << dl.text.size() << " (capped)\n";
      std::cout << "[DATA]  Vocabulary size  : " << dl.vocab_size << "\n";
      std::cout << "[DATA]  Train tokens     : " << dl.train_data.size() << "\n";
      std::cout << "[DATA]  Val   tokens     : " << dl.val_data.size() << "\n";
}

static void bench_data(const Options &opt, DataLoader &dl, std::vector<BenchRow> &rows)
{
      std::string text = dl.text;
      auto encode_fn = [&]() -> double
      {
            std::vector<int> ids = dl.encode(text);
            return (double)ids.size();
      };
      auto timed = run_timed(opt.runs, opt.warmup, encode_fn);
      BenchRow row;
      row.suite = "data";
      row.name = "char_encode";
      row.tokens = (int)text.size();
      row.stats = timed.first;
      row.tokens_per_sec = row.tokens / (row.stats.avg_ms / 1000.0);
      row.samples = opt.runs;
      row.memory_mb = 0.0;
      push_row(rows, row);

      std::mt19937 rng(SEED);
      int seq_len = BLOCK_SIZE;
      int batch_size = opt.batch_size;
      auto batch_fn = [&]() -> double
      {
            auto batch = dl.get_batch("train", batch_size, seq_len, rng);
            return (double)(batch.first[0] + batch.second[0]);
      };
      timed = run_timed(opt.runs, opt.warmup, batch_fn);
      row = BenchRow();
      row.suite = "data";
      row.name = "batch_sample";
      row.batch_size = batch_size;
      row.sequence_length = seq_len;
      row.tokens = batch_size * seq_len;
      row.stats = timed.first;
      row.tokens_per_sec = row.tokens / (row.stats.avg_ms / 1000.0);
      row.samples = opt.runs;
      push_row(rows, row);
}

static void bench_primitives(const Options &opt, std::vector<BenchRow> &rows)
{
      std::mt19937 rng(SEED);
      struct Case
      {
            std::string name;
            int B, T, D, E;
      };
      std::vector<Case> cases = {
          {"matmul_3d", 1, 16, N_EMBD, N_EMBD},
          {"matmul_3d", opt.batch_size, BLOCK_SIZE, N_EMBD, N_EMBD},
          {"softmax3d", opt.batch_size, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE},
          {"layer_norm", opt.batch_size, BLOCK_SIZE, N_EMBD, N_EMBD},
      };

      for (const auto &c : cases)
      {
            Tensor x = Tensor::randn({c.B, c.T, c.D}, 0.0f, 1.0f, rng);
            Tensor w = Tensor::randn({c.D, c.E}, 0.0f, 0.02f, rng);
            Tensor gamma = Tensor::ones({c.D});
            Tensor beta = Tensor::zeros({c.D});
            auto fn = [&]() -> double
            {
                  Tensor out;
                  if (c.name == "softmax3d")
                        out = softmax3d(x);
                  else if (c.name == "layer_norm")
                        out = layer_norm(x, gamma, beta);
                  else
                        out = matmul(x, w);
                  return out.data.empty() ? 0.0 : out.data[0];
            };
            auto timed = run_timed(opt.runs, opt.warmup, fn);
            BenchRow row;
            row.suite = "primitive";
            row.name = c.name + "_" + std::to_string(c.B) + "x" + std::to_string(c.T);
            row.batch_size = c.B;
            row.sequence_length = c.T;
            row.tokens = c.B * c.T;
            row.stats = timed.first;
            row.tokens_per_sec = row.tokens / (row.stats.avg_ms / 1000.0);
            row.samples = opt.runs;
            push_row(rows, row);
      }
}

static void bench_forward(const Options &opt, GPTLanguageModel &model, DataLoader &dl, std::vector<BenchRow> &rows)
{
      std::mt19937 rng(SEED);
      std::vector<std::pair<int, int>> cases = {{1, 8}, {1, BLOCK_SIZE}, {opt.batch_size, BLOCK_SIZE}};
      for (auto c : cases)
      {
            int B = c.first;
            int T = c.second;
            auto batch = dl.get_batch("train", B, T, rng);
            float last_loss = 0.0f;
            auto fn = [&]() -> double
            {
                  auto out = model.forward(batch.first, B, T, batch.second, false);
                  last_loss = out.second;
                  return out.first.data.empty() ? 0.0 : out.first.data[0];
            };
            auto timed = run_timed(opt.runs, opt.warmup, fn);
            BenchRow row;
            row.suite = "forward";
            row.name = "batch" + std::to_string(B) + "_seq" + std::to_string(T);
            row.batch_size = B;
            row.sequence_length = T;
            row.tokens = B * T;
            row.stats = timed.first;
            row.tokens_per_sec = row.tokens / (row.stats.avg_ms / 1000.0);
            row.samples = opt.runs;
            row.loss = last_loss;
            row.has_loss = true;
            row.memory_mb = estimated_model_mb(model);
            push_row(rows, row);
      }
}

static void bench_training_step(const Options &opt, GPTLanguageModel &model, DataLoader &dl, std::vector<BenchRow> &rows)
{
      std::mt19937 rng(SEED);
      AdamWState optimizer = build_optimizer(model, LEARNING_RATE);
      auto batch = dl.get_batch("train", opt.batch_size, BLOCK_SIZE, rng);
      float last_loss = 0.0f;

      auto fn = [&]() -> double
      {
            SavedForward saved = forward_save(model, batch.first, opt.batch_size, BLOCK_SIZE, batch.second, true);
            last_loss = cross_entropy(saved.logits2d, batch.second);
            Grads grads = backward(model, saved);
            apply_grads(model, grads, optimizer);
            return last_loss;
      };
      auto timed = run_timed(opt.train_steps, opt.warmup, fn);
      BenchRow row;
      row.suite = "training";
      row.name = "adamw_step";
      row.batch_size = opt.batch_size;
      row.sequence_length = BLOCK_SIZE;
      row.tokens = opt.batch_size * BLOCK_SIZE;
      row.stats = timed.first;
      row.tokens_per_sec = row.tokens / (row.stats.avg_ms / 1000.0);
      row.samples = opt.train_steps;
      row.loss = last_loss;
      row.has_loss = true;
      row.memory_mb = estimated_model_mb(model);
      push_row(rows, row);
}

static void bench_generation(const Options &opt, GPTLanguageModel &model, DataLoader &dl, std::vector<BenchRow> &rows)
{
      std::vector<std::pair<std::string, std::string>> prompts = {
          {"empty", ""},
          {"short", "The future of local AI is"},
          {"long", "Quadtrix is a compact transformer benchmark that measures "}};
      for (const auto &p : prompts)
      {
            std::vector<int> ctx = dl.encode(p.second);
            if (ctx.empty())
                  ctx = {0};
            if ((int)ctx.size() > BLOCK_SIZE)
                  ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
            int prompt_len = (int)ctx.size();
            auto fn = [&]() -> double
            {
                  std::vector<int> out = model.generate(ctx, opt.generate_tokens);
                  return (double)out.back();
            };
            auto timed = run_timed(opt.runs, opt.warmup, fn);
            BenchRow row;
            row.suite = "generation";
            row.name = p.first;
            row.batch_size = 1;
            row.sequence_length = prompt_len;
            row.tokens = opt.generate_tokens;
            row.stats = timed.first;
            row.tokens_per_sec = row.tokens / (row.stats.avg_ms / 1000.0);
            row.samples = opt.runs;
            row.memory_mb = estimated_model_mb(model);
            push_row(rows, row);
      }
}

static void ensure_dir(const std::string &dir)
{
#if __has_include(<filesystem>)
      fs::create_directories(dir);
#else
      (void)dir;
#endif
}

static void save_results(const Options &opt, const GPTLanguageModel &model, const DataLoader &dl, const std::vector<BenchRow> &rows)
{
      ensure_dir(opt.out_dir);
      std::string json_path = opt.out_dir + "/cpp_benchmark.json";
      std::string csv_path = opt.out_dir + "/cpp_benchmark.csv";

      std::ofstream j(json_path.c_str());
      j << std::fixed << std::setprecision(6);
      j << "{\n";
      j << "  \"schema_version\": 1,\n";
      j << "  \"timestamp\": \"" << json_escape(now_iso()) << "\",\n";
      j << "  \"backend\": \"cpp\",\n";
      j << "  \"system\": {\n";
      j << "    \"compiler\": \"" << json_escape(
#ifdef __VERSION__
        __VERSION__
#else
        "unknown"
#endif
        ) << "\",\n";
      j << "    \"standard\": \"C++17\"\n";
      j << "  },\n";
      j << "  \"model\": {\n";
      j << "    \"vocab_size\": " << dl.vocab_size << ",\n";
      j << "    \"block_size\": " << BLOCK_SIZE << ",\n";
      j << "    \"n_embd\": " << N_EMBD << ",\n";
      j << "    \"n_head\": " << N_HEAD << ",\n";
      j << "    \"n_layer\": " << N_LAYER << ",\n";
      j << "    \"dropout\": " << DROPOUT << ",\n";
      j << "    \"parameters\": " << model.num_params() << ",\n";
      j << "    \"parameter_mb_fp32\": " << estimated_model_mb(model) << "\n";
      j << "  },\n";
      j << "  \"config\": {\n";
      j << "    \"data\": \"" << json_escape(opt.data_path) << "\",\n";
      j << "    \"model\": \"" << json_escape(opt.model_path) << "\",\n";
      j << "    \"runs\": " << opt.runs << ",\n";
      j << "    \"warmup\": " << opt.warmup << ",\n";
      j << "    \"batch_size\": " << opt.batch_size << ",\n";
      j << "    \"train_steps\": " << opt.train_steps << ",\n";
      j << "    \"generate_tokens\": " << opt.generate_tokens << ",\n";
      j << "    \"max_data_chars\": " << opt.max_data_chars << ",\n";
      j << "    \"random_weights\": " << (opt.random_weights ? "true" : "false") << "\n";
      j << "  },\n";
      j << "  \"results\": [\n";
      for (size_t i = 0; i < rows.size(); ++i)
      {
            const BenchRow &r = rows[i];
            j << "    {\n";
            j << "      \"suite\": \"" << json_escape(r.suite) << "\",\n";
            j << "      \"name\": \"" << json_escape(r.name) << "\",\n";
            j << "      \"backend\": \"cpp\",\n";
            j << "      \"batch_size\": " << r.batch_size << ",\n";
            j << "      \"sequence_length\": " << r.sequence_length << ",\n";
            j << "      \"tokens\": " << r.tokens << ",\n";
            j << "      \"avg_ms\": " << r.stats.avg_ms << ",\n";
            j << "      \"median_ms\": " << r.stats.median_ms << ",\n";
            j << "      \"min_ms\": " << r.stats.min_ms << ",\n";
            j << "      \"max_ms\": " << r.stats.max_ms << ",\n";
            j << "      \"p90_ms\": " << r.stats.p90_ms << ",\n";
            j << "      \"p95_ms\": " << r.stats.p95_ms << ",\n";
            j << "      \"std_ms\": " << r.stats.std_ms << ",\n";
            j << "      \"tokens_per_sec\": " << r.tokens_per_sec << ",\n";
            j << "      \"samples\": " << r.samples << ",\n";
            if (r.has_loss)
                  j << "      \"loss\": " << r.loss << ",\n";
            else
                  j << "      \"loss\": null,\n";
            j << "      \"memory_mb\": " << r.memory_mb << ",\n";
            j << "      \"notes\": \"" << json_escape(r.notes) << "\"\n";
            j << "    }" << (i + 1 < rows.size() ? "," : "") << "\n";
      }
      j << "  ]\n";
      j << "}\n";

      std::ofstream c(csv_path.c_str());
      c << "suite,name,backend,batch_size,sequence_length,tokens,avg_ms,median_ms,min_ms,max_ms,p90_ms,p95_ms,std_ms,tokens_per_sec,samples,loss,memory_mb,notes\n";
      c << std::fixed << std::setprecision(6);
      for (const auto &r : rows)
      {
            c << r.suite << "," << r.name << ",cpp,"
              << r.batch_size << "," << r.sequence_length << "," << r.tokens << ","
              << r.stats.avg_ms << "," << r.stats.median_ms << "," << r.stats.min_ms << ","
              << r.stats.max_ms << "," << r.stats.p90_ms << "," << r.stats.p95_ms << ","
              << r.stats.std_ms << "," << r.tokens_per_sec << "," << r.samples << ",";
            if (r.has_loss)
                  c << r.loss;
            c << "," << r.memory_mb << ",\"" << json_escape(r.notes) << "\"\n";
      }

      std::cout << "Saved " << json_path << "\n";
      std::cout << "Saved " << csv_path << "\n";
}

static Options parse_args(int argc, char **argv)
{
      Options opt;
      for (int i = 1; i < argc; ++i)
      {
            std::string a = argv[i];
            if (a == "--data" && i + 1 < argc)
                  opt.data_path = argv[++i];
            else if (a == "--model" && i + 1 < argc)
                  opt.model_path = argv[++i];
            else if (a == "--out" && i + 1 < argc)
                  opt.out_dir = argv[++i];
            else if (a == "--runs" && i + 1 < argc)
                  opt.runs = std::atoi(argv[++i]);
            else if (a == "--warmup" && i + 1 < argc)
                  opt.warmup = std::atoi(argv[++i]);
            else if (a == "--batch-size" && i + 1 < argc)
                  opt.batch_size = std::atoi(argv[++i]);
            else if (a == "--train-steps" && i + 1 < argc)
                  opt.train_steps = std::atoi(argv[++i]);
            else if (a == "--generate-tokens" && i + 1 < argc)
                  opt.generate_tokens = std::atoi(argv[++i]);
            else if (a == "--max-data-chars" && i + 1 < argc)
                  opt.max_data_chars = std::atoi(argv[++i]);
            else if (a == "--random-weights")
                  opt.random_weights = true;
            else if (a == "--quick")
                  opt.quick = true;
      }
      if (opt.quick)
      {
            opt.runs = 2;
            opt.warmup = 1;
            opt.train_steps = 1;
            opt.generate_tokens = 4;
            opt.max_data_chars = opt.max_data_chars > 0 ? std::min(opt.max_data_chars, 50000) : 50000;
      }
      opt.runs = std::max(1, opt.runs);
      opt.warmup = std::max(0, opt.warmup);
      opt.batch_size = std::max(1, opt.batch_size);
      opt.train_steps = std::max(1, opt.train_steps);
      opt.generate_tokens = std::max(1, opt.generate_tokens);
      opt.max_data_chars = std::max(0, opt.max_data_chars);
      return opt;
}

int main(int argc, char **argv)
{
      Options opt = parse_args(argc, argv);
      std::cout << "Quadtrix C++ Benchmark\n";
      std::cout << "Runs: " << opt.runs << ", warmup: " << opt.warmup << "\n";

      DataLoader dl;
      try
      {
            load_data(opt, dl);
      }
      catch (const std::exception &e)
      {
            std::cerr << e.what() << "\n";
            return 1;
      }

      GPTLanguageModel model(dl.vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, SEED);
      if (!opt.random_weights && file_exists(opt.model_path))
            model.load(opt.model_path);
      else
            std::cout << "[INFO] Using random weights for model benchmarks.\n";

      std::cout << "Parameters: " << model.num_params() << "\n";
      std::vector<BenchRow> rows;
      bench_data(opt, dl, rows);
      bench_primitives(opt, rows);
      bench_forward(opt, model, dl, rows);
      bench_training_step(opt, model, dl, rows);
      bench_generation(opt, model, dl, rows);
      save_results(opt, model, dl, rows);
      return 0;
}
