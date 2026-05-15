// Run:
//   .\quadtrix_bench.exe data\input.txt
//   .\quadtrix_bench.exe data\input.txt --tokens 100 --runs 10 --warmup 3
//
// Flags (all optional):
//   --tokens  N   tokens to generate per run        (default: 50)
//   --runs    N   how many timed runs per prompt     (default: 5)
//   --warmup  N   un-timed warmup runs per prompt    (default: 2)

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include "config/config.h"
#include "include/dataloader.h"
#include "include/gpt.h"

static bool file_exists(const std::string &p)
{
      std::ifstream f(p.c_str(), std::ios::binary);
      return f.good();
}

static double now_ms()
{
      using namespace std::chrono;
      return duration<double, std::milli>(
                 steady_clock::now().time_since_epoch())
          .count();
}

static double mean(const std::vector<double> &v)
{
      return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double stdev(const std::vector<double> &v, double m)
{
      double sq = 0.0;
      for (double x : v)
            sq += (x - m) * (x - m);
      return std::sqrt(sq / v.size());
}

static double timed_run(GPTLanguageModel &model,
                        DataLoader &dl,
                        const std::vector<int> &prompt_ctx,
                        int n_tokens)
{
      std::vector<int> ctx = prompt_ctx;

      double t0 = now_ms();
      for (int i = 0; i < n_tokens; ++i)
      {
            ctx = model.generate(ctx, 1);
            if ((int)ctx.size() > BLOCK_SIZE)
                  ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
      }
      return now_ms() - t0;
}

//

static void section(const std::string &title)
{
      ;
      std::cout << "  " << title << "\n";
}

struct PromptResult
{
      std::string label;
      int prompt_tokens;
      int gen_tokens;
      double avg_ms;
      double min_ms;
      double max_ms;
      double std_ms;
      double avg_tps; // tokens per second
};

static PromptResult bench_prompt(GPTLanguageModel &model,
                                 DataLoader &dl,
                                 const std::string &prompt,
                                 int n_tokens,
                                 int n_runs,
                                 int n_warmup)
{
      // encode
      std::vector<int> ctx = dl.encode(prompt);
      if (ctx.empty())
            ctx = {0};
      if ((int)ctx.size() > BLOCK_SIZE)
            ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());

      int prompt_len = (int)ctx.size();

      // warmup (un-timed)
      for (int i = 0; i < n_warmup; ++i)
            timed_run(model, dl, ctx, n_tokens);

      // timed runs
      std::vector<double> times;
      times.reserve(n_runs);
      for (int i = 0; i < n_runs; ++i)
            times.push_back(timed_run(model, dl, ctx, n_tokens));

      double m = mean(times);
      double sd = stdev(times, m);
      double mn = *std::min_element(times.begin(), times.end());
      double mx = *std::max_element(times.begin(), times.end());
      double tps = n_tokens / (m / 1000.0);

      // truncate prompt for display
      std::string label = prompt.size() > 30
                              ? prompt.substr(0, 27) + "..."
                              : prompt;

      return PromptResult{label, prompt_len, n_tokens, m, mn, mx, sd, tps};
}

static void print_table(const std::vector<PromptResult> &results)
{
      section("RESULTS");

      // header
      std::cout << std::left
                << std::setw(34) << "Prompt"
                << std::right
                << std::setw(8) << "P.Tok"
                << std::setw(8) << "G.Tok"
                << std::setw(10) << "Avg ms"
                << std::setw(10) << "Min ms"
                << std::setw(10) << "Max ms"
                << std::setw(9) << "Std ms"
                << std::setw(10) << "tok/s"
                << "\n";
      std::cout << std::string(99, '-') << "\n";

      std::cout << std::fixed;
      for (const auto &r : results)
      {
            std::cout << std::left
                      << std::setw(34) << r.label
                      << std::right
                      << std::setw(8) << r.prompt_tokens
                      << std::setw(8) << r.gen_tokens
                      << std::setw(10) << std::setprecision(1) << r.avg_ms
                      << std::setw(10) << std::setprecision(1) << r.min_ms
                      << std::setw(10) << std::setprecision(1) << r.max_ms
                      << std::setw(9) << std::setprecision(1) << r.std_ms
                      << std::setw(10) << std::setprecision(2) << r.avg_tps
                      << "\n";
      }

      double total_avg_tps = 0.0;
      double best_tps = 0.0;
      for (const auto &r : results)
      {
            total_avg_tps += r.avg_tps;
            best_tps = std::max(best_tps, r.avg_tps);
      }
      double overall_tps = total_avg_tps / results.size();

      std::cout << "\n  Overall avg throughput : "
                << std::setprecision(2) << overall_tps << " tok/s\n";
      std::cout << "  Peak throughput        : "
                << std::setprecision(2) << best_tps << " tok/s\n";
      std::cout << "  ms per token (avg)     : "
                << std::setprecision(2) << 1000.0 / overall_tps << " ms\n";
}

static void save_csv(const std::vector<PromptResult> &results,
                     const std::string &path)
{
      std::ofstream f(path);
      if (!f)
      {
            std::cerr << "[WARN] Could not write CSV to " << path << "\n";
            return;
      }
      f << "prompt,prompt_tokens,gen_tokens,avg_ms,min_ms,max_ms,std_ms,tok_per_sec\n";
      for (const auto &r : results)
      {
            f << "\"" << r.label << "\","
              << r.prompt_tokens << ","
              << r.gen_tokens << ","
              << r.avg_ms << ","
              << r.min_ms << ","
              << r.max_ms << ","
              << r.std_ms << ","
              << r.avg_tps << "\n";
      }
      std::cout << "\n  CSV saved to: " << path << "\n";
}

int main(int argc, char *argv[])
{

      std::string data_path = DEFAULT_CLEANED_PATH;
      std::string model_path = BEST_MODEL_PATH;
      int n_tokens = 50;
      int n_runs = 5;
      int n_warmup = 2;

      for (int i = 1; i < argc; ++i)
      {
            std::string a = argv[i];
            if (a == "--tokens" && i + 1 < argc)
                  n_tokens = std::atoi(argv[++i]);
            else if (a == "--runs" && i + 1 < argc)
                  n_runs = std::atoi(argv[++i]);
            else if (a == "--warmup" && i + 1 < argc)
                  n_warmup = std::atoi(argv[++i]);
            else
                  data_path = a;
      }

      std::cout << "  Quadtrix Inference Benchmark\n";
      std::cout << "  data   : " << data_path << "\n";
      std::cout << "  model  : " << model_path << "\n";
      std::cout << "  tokens : " << n_tokens << " per run\n";
      std::cout << "  runs   : " << n_runs << " timed  +  "
                << n_warmup << " warmup\n";

      DataLoader dl;
      try
      {
            dl.load(data_path);
      }
      catch (const std::exception &e)
      {
            std::cerr << "[ERROR] " << e.what() << "\n";
            return 1;
      }

      if (!file_exists(model_path))
      {
            std::cerr << "[ERROR] Weights not found at " << model_path << "\n";
            std::cerr << "[HINT]  Train first, or set " << MODEL_PATH_ENV_VAR << "\n";
            return 1;
      }

      GPTLanguageModel model(dl.vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, SEED);
      model.load(model_path);

      std::cout << "\n[OK] Model loaded  (" << model.num_params() / 1.0e6f
                << " M params)\n";

      std::vector<std::string> prompts = {
          "",
          "The",                       // 1-token prompt
          "Once upon a time",          // short prompt
          "The quick brown fox jumps", // medium prompt
          std::string(1, 'a'),         // long prompt (stress-tests context window)
      };

      section("RUNNING");
      std::vector<PromptResult> results;
      results.reserve(prompts.size());

      for (size_t i = 0; i < prompts.size(); ++i)
      {
            std::string display = prompts[i].empty()
                                      ? "(empty / BOS)"
                                      : (prompts[i].size() > 30
                                             ? prompts[i].substr(0, 27) + "..."
                                             : prompts[i]);

            std::cout << "  [" << (i + 1) << "/" << prompts.size() << "] \""
                      << display << "\"  ...  " << std::flush;

            PromptResult r = bench_prompt(model, dl,
                                          prompts[i],
                                          n_tokens, n_runs, n_warmup);
            results.push_back(r);

            std::cout << std::fixed << std::setprecision(2)
                      << r.avg_tps << " tok/s\n";
      }

      print_table(results);
      save_csv(results, "benchmark_results.csv");

      std::cout << "\n";

      std::cout << "  Done.\n";
      return 0;
}