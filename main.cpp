/*'''
 * llm.cpp - LMGNU Organization
 * Copyright (C) 2026 Eamon
 * https://github.com/LMGNU/llm.cpp
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "config/config.h"
#include "include/backward.h"
#include "include/llm-cpp.hpp"
#include "include/lm.h"
#include "include/sampler.h"
#include "include/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <psapi.h> // K32GetProcessMemoryInfo ships in kernel32.dll
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <unistd.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

// ANSI color escape codes for terminal formatting
#define ANSI_CYAN "\033[96m"
#define ANSI_BOLD "\033[1m"
#define ANSI_RESET "\033[0m"

static volatile bool g_interrupted = false;
static void sig_handler(int)
{
      g_interrupted = true;
}

// Return the current wall-clock time as a formatted string.
static std::string now_str()
{
      std::time_t t = std::time(nullptr);
      char buf[32];
      std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
      return buf;
}

// Return elapsed seconds since an arbitrary epoch using a monotonic clock.
static double wall_secs()
{
      using namespace std::chrono;
      return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Helper function to fetch Hardware Specs safely across platforms
static std::string get_cpu_info()
{
#if defined(__linux__)
      std::ifstream cpuinfo("/proc/cpuinfo");
      std::string line;
      while (std::getline(cpuinfo, line))
      {
            if (line.find("model name") != std::string::npos)
            {
                  size_t colon = line.find(':');
                  if (colon != std::string::npos)
                  {
                        std::string name = line.substr(colon + 2);
                        if (name.length() > 34)
                              return name.substr(0, 31) + "...";
                        return name;
                  }
            }
      }
      return "Generic Linux CPU";
#elif defined(__APPLE__)
      char buffer[128];
      size_t bufferlen = sizeof(buffer);
      if (sysctlbyname("machdep.cpu.brand_string", &buffer, &bufferlen, NULL, 0) == 0)
      {
            std::string name(buffer);
            if (name.length() > 34)
                  return name.substr(0, 31) + "...";
            return name;
      }
      return "Apple Silicon / Mac CPU";
#elif defined(_WIN32)
      HKEY hKey;
      if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS)
      {
            char name[128];
            DWORD size = sizeof(name);
            if (RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL, (LPBYTE)name, &size) ==
                ERROR_SUCCESS)
            {
                  RegCloseKey(hKey);
                  std::string cpu_name(name);
                  if (cpu_name.length() > 34)
                        return cpu_name.substr(0, 31) + "...";
                  return cpu_name;
            }
            RegCloseKey(hKey);
      }
      return "Generic Windows CPU";
#else
      return "Generic CPU Host";
#endif
}

// Return the total physical system RAM in megabytes.
// Works on Windows, Linux, and macOS with no extra linker flags required.
// Returns 0.0 if the platform-specific query fails.
static double get_total_ram_mb()
{
#if defined(_WIN32)
      MEMORYSTATUSEX status;
      status.dwLength = sizeof(status);
      if (GlobalMemoryStatusEx(&status))
            return (double)status.ullTotalPhys / (1024.0 * 1024.0);
      return 0.0;
#elif defined(__APPLE__)
      int64_t mem = 0;
      size_t len = sizeof(mem);
      if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) == 0)
            return (double)mem / (1024.0 * 1024.0);
      return 0.0;
#elif defined(__linux__)
      std::ifstream meminfo("/proc/meminfo");
      std::string line;
      while (std::getline(meminfo, line))
      {
            if (line.compare(0, 9, "MemTotal:") == 0)
            {
                  std::istringstream iss(line.substr(9));
                  double kb = 0.0;
                  iss >> kb;
                  return kb / 1024.0;
            }
      }
      return 0.0;
#else
      return 0.0;
#endif
}

// Return the current process's resident memory usage in megabytes.
// Works on Windows, Linux, and macOS with no extra linker flags required.
// Returns 0.0 if the platform-specific query fails.
static double get_ram_usage_mb()
{
#if defined(_WIN32)
      PROCESS_MEMORY_COUNTERS pmc;
      if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
            return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
      return 0.0;
#elif defined(__APPLE__)
      mach_task_basic_info_data_t info;
      mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
      if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) ==
          KERN_SUCCESS)
            return (double)info.resident_size / (1024.0 * 1024.0);
      return 0.0;
#elif defined(__linux__)
      std::ifstream status("/proc/self/status");
      std::string line;
      while (std::getline(status, line))
      {
            if (line.compare(0, 6, "VmRSS:") == 0)
            {
                  std::istringstream iss(line.substr(6));
                  double kb = 0.0;
                  iss >> kb;
                  return kb / 1024.0;
            }
      }
      return 0.0;
#else
      return 0.0;
#endif
}

// Return true if the file at path exists and can be opened.
static bool file_exists(const std::string &path)
{
      std::ifstream f(path.c_str(), std::ios::binary);
      return f.good();
}

// Return the directory portion of a file path, or "." for bare filenames.
static std::string dir_name(const std::string &path)
{
      std::string::size_type pos = path.find_last_of("/\\");
      if (pos == std::string::npos)
            return ".";
      if (pos == 0)
            return path.substr(0, 1);
      return path.substr(0, pos);
}

// Return true if path starts with a drive letter or a slash.
static bool is_absolute_path(const std::string &path)
{
      if (path.empty())
            return false;
      if (path.size() > 1 && path[1] == ':')
            return true;
      return path[0] == '/' || path[0] == '\\';
}

// Join base directory and child path with a separator.
static std::string join_path(const std::string &base, const std::string &child)
{
      if (base.empty() || base == ".")
            return child;
      char last = base[base.size() - 1];
      if (last == '/' || last == '\\')
            return base + child;
      return base + "/" + child;
}

// Try the requested path first, then look next to the executable.
// Return whichever path exists, or the requested path if neither does.
static std::string choose_existing_path(const std::string &requested_path, const std::string &argv0)
{
      if (requested_path.empty())
            return requested_path;
      if (file_exists(requested_path))
            return requested_path;
      if (is_absolute_path(requested_path))
            return requested_path;

      std::vector<std::string> candidates;
      candidates.push_back(join_path(dir_name(argv0), requested_path));
      candidates.push_back(join_path(".", requested_path));

      for (size_t i = 0; i < candidates.size(); ++i)
      {
            if (file_exists(candidates[i]))
                  return candidates[i];
      }
      return requested_path;
}

// Choose a writable output path, preferring one next to the executable
// only when the current directory copy does not already exist.
static std::string choose_output_path(const std::string &requested_path, const std::string &argv0)
{
      if (requested_path.empty() || is_absolute_path(requested_path))
            return requested_path;

      std::string exe_relative = join_path(dir_name(argv0), requested_path);
      if (file_exists(requested_path) || !file_exists(exe_relative))
            return requested_path;
      return exe_relative;
}

// Print a one-line usage summary listing all supported flags.
static void print_usage(const char *argv0)
{
      std::cout << "Usage: " << argv0 << " [options] [data_file]\n"
                << "\n"
                << "  --generate              run inference only (needs saved weights)\n"
                << "  --chat                  start interactive chat mode\n"
                << "  --chat-tokens N         max tokens per chat reply (default "
                << DEFAULT_CHAT_TOKENS << ")\n"
                << "  --system TEXT           system prompt prepended to every chat turn\n"
                << "  --rep-penalty F         repetition penalty, 1.0 means off (default "
                << DEFAULT_REP_PENALTY << ")\n"
                << "  --rep-window N          recent token window for penalty (default "
                << DEFAULT_REP_WINDOW << ")\n"
                << "  --help                  show this message\n";
}

// Sample n_tokens from the model using the given sampler params and print them.
static void
sample_tokens(GPTLanguageModel &model, DataLoader &dl, int n_tokens, const SamplerParams &params)
{
      std::vector<int> ctx = {0};
      for (int i = 0; i < n_tokens; ++i)
      {
            ctx = model.generate(ctx, 1, params);
            std::cout << dl.decode({ctx.back()}) << std::flush;
            if ((int)ctx.size() > BLOCK_SIZE)
                  ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
      }
      std::cout << "\n";
}

// Estimate average cross-entropy loss over EVAL_ITERS random batches.
// No gradients are computed and training mode is disabled.
static float
estimate_loss(GPTLanguageModel &model, DataLoader &dl, const std::string &split, std::mt19937 &rng)
{
      float total = 0.0f;
      for (int k = 0; k < EVAL_ITERS; ++k)
      {
            std::pair<std::vector<int>, std::vector<int>> batch =
                  dl.get_batch(split, BATCH_SIZE, BLOCK_SIZE, rng);
            std::pair<Tensor, float> result =
                  model.forward(batch.first, BATCH_SIZE, BLOCK_SIZE, batch.second, false);
            total += result.second;
      }
      return total / EVAL_ITERS;
}

// Build the initial context for a chat turn.
// Places system tokens first (if any), then appends user tokens.
// Crops the combined context to BLOCK_SIZE from the right.
static std::vector<int> build_turn_context(const std::vector<int> &sys_tokens,
                                           const std::vector<int> &user_tokens)
{
      std::vector<int> ctx;
      ctx.reserve(sys_tokens.size() + user_tokens.size());
      ctx.insert(ctx.end(), sys_tokens.begin(), sys_tokens.end());
      ctx.insert(ctx.end(), user_tokens.begin(), user_tokens.end());

      if ((int)ctx.size() > BLOCK_SIZE)
            ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());

      return ctx;
}

// Run an interactive chat loop.
// The system prompt is encoded once and prepended to the context on every turn.
// Repetition penalty is applied during generation using params.
static void
run_chat(GPTLanguageModel &model, DataLoader &dl, int max_new_tokens, const SamplerParams &params)
{
      std::vector<int> sys_tokens;
      if (!params.system_prompt.empty())
      {
            sys_tokens = dl.encode(params.system_prompt);
            if (sys_tokens.empty())
            {
                  std::cerr << "[WARN]  System prompt produced zero tokens. "
                               "All characters may be outside the training vocabulary.\n";
            }
            else
            {
                  std::cout << "[CHAT]  System prompt active (" << sys_tokens.size() << " tokens, "
                            << BLOCK_SIZE - (int)sys_tokens.size()
                            << " tokens left for user input)\n";
            }
      }

      std::cout << "\n" << std::string(60, '=') << "\n";
      std::cout << "  CHAT MODE\n";
      std::cout << "  Type your prompt and press Enter.\n";
      std::cout << "  Type quit or exit to leave.\n";
      std::cout << std::string(60, '=') << "\n\n";

      while (!g_interrupted)
      {
            std::cout << "\033[1;32mroot>\033[0m ";
            std::cout.flush();

            std::string prompt;
            if (!std::getline(std::cin, prompt))
                  break;

            size_t s = prompt.find_first_not_of(" \t\r\n");
            size_t e = prompt.find_last_not_of(" \t\r\n");
            if (s == std::string::npos)
                  continue;
            prompt = prompt.substr(s, e - s + 1);

            if (prompt == "quit" || prompt == "exit")
            {
                  std::cout << "[Chat] Bye!\n";
                  break;
            }

            std::vector<int> user_tokens = dl.encode(prompt);
            if (user_tokens.empty())
                  user_tokens = {0};

            std::vector<int> ctx = build_turn_context(sys_tokens, user_tokens);

            std::cout << "\033[1;36mllm>\033[0m ";
            std::cout.flush();

            for (int tok = 0; tok < max_new_tokens && !g_interrupted; ++tok)
            {
                  ctx = model.generate(ctx, 1, params);
                  std::cout << dl.decode({ctx.back()}) << std::flush;

                  if ((int)ctx.size() > BLOCK_SIZE)
                        ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
            }
            std::cout << "\n\n";
      }
}

// Entry point: parse flags, load data, build model, then train or run inference.
int main(int argc, char *argv[])
{
      std::signal(SIGINT, sig_handler);

      print_banner();

      std::string data_path = DEFAULT_CLEANED_PATH;
      std::string model_path = BEST_MODEL_PATH;

      const char *env_data = std::getenv(DATA_PATH_ENV_VAR.c_str());
      const char *env_model = std::getenv(MODEL_PATH_ENV_VAR.c_str());
      if (env_data != nullptr && env_data[0] != '\0')
            data_path = env_data;
      if (env_model != nullptr && env_model[0] != '\0')
            model_path = env_model;

      bool gen_mode = false;
      bool chat_mode = false;
      int chat_tokens = DEFAULT_CHAT_TOKENS;
      float rep_penalty = DEFAULT_REP_PENALTY;
      int rep_window = DEFAULT_REP_WINDOW;
      std::string system_prompt;

      for (int i = 1; i < argc; ++i)
      {
            std::string a = argv[i];

            if (a == "--help")
            {
                  print_usage(argv[0]);
                  return 0;
            }
            else if (a == "--generate")
            {
                  gen_mode = true;
            }
            else if (a == "--chat")
            {
                  chat_mode = true;
            }
            else if (a == "--chat-tokens" && i + 1 < argc)
            {
                  chat_tokens = std::atoi(argv[++i]);
            }
            else if (a == "--system" && i + 1 < argc)
            {
                  system_prompt = argv[++i];
            }
            else if (a == "--rep-penalty" && i + 1 < argc)
            {
                  rep_penalty = (float)std::atof(argv[++i]);
            }
            else if (a == "--rep-window" && i + 1 < argc)
            {
                  rep_window = std::atoi(argv[++i]);
            }
            else
            {
                  data_path = a;
            }
      }

      data_path = choose_existing_path(data_path, argv[0]);
      model_path = choose_output_path(model_path, argv[0]);

      SamplerParams sampler;
      sampler.rep_penalty = rep_penalty;
      sampler.rep_window = rep_window;
      sampler.system_prompt = system_prompt;

      DataLoader dl;
      try
      {
            dl.load(data_path);
      }
      catch (const std::exception &e)
      {
            std::cerr << e.what() << "\n";
            std::cerr << "[HINT]  Put your text at " << DEFAULT_CLEANED_PATH
                      << ", pass a file path as the first argument, or set " << DATA_PATH_ENV_VAR
                      << ".\n";
            return 1;
      }

      GPTLanguageModel model(dl.vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, SEED);

      long n_params = model.num_params();
      std::string cpu_spec = get_cpu_info();

      // Formatted nvidia-smi style system output with CPU Specs
      std::cout
            << "+-----------------------------------------------------------------------------+\n";
      std::cout
            << "| LLM.cpp                                                                     |\n";
      std::cout
            << "|======================================+======================================|\n";
      std::cout
            << "| Parameter / Spec                     | Value                                |\n";
      std::cout
            << "|--------------------------------------+--------------------------------------|\n";
      std::cout << "| Host CPU Device                      | " << std::left << std::setw(36)
                << cpu_spec << " |\n";
      std::cout << "| Host RAM (Total)                     | " << std::left << std::setw(36)
                << (std::to_string((long long)get_total_ram_mb()) + " MB") << " |\n";
      std::cout << "| Max Sequence Length                  | " << std::left << std::setw(36)
                << BLOCK_SIZE << " |\n";
      std::cout << "| Vocab Size (BPE Merges)              | " << std::left << std::setw(36)
                << dl.vocab_size << " |\n";
      std::cout << "| Number of Layers                     | " << std::left << std::setw(36)
                << N_LAYER << " |\n";
      std::cout << "| Number of Heads                      | " << std::left << std::setw(36)
                << N_HEAD << " |\n";
      std::cout << "| Channels (Embeddings)                | " << std::left << std::setw(36)
                << N_EMBD << " |\n";
      std::cout << "| Number of Parameters                 | " << std::left << std::setw(36)
                << n_params << " |\n";
      std::cout << "| Repetition Penalty                   | " << std::left << std::setw(36)
                << rep_penalty << " |\n";
      std::cout << "| Repetition Window                    | " << std::left << std::setw(36)
                << rep_window << " |\n";
      std::cout
            << "+--------------------------------------+--------------------------------------+\n";
      std::cout << std::right; // Reset formatting stream state

      if (chat_mode)
      {
            if (!file_exists(model_path))
            {
                  std::cerr << "[ERROR] Cannot start chat because model weights were not found at "
                            << model_path << "\n";
                  std::cerr << "[HINT]  Train first, or set " << MODEL_PATH_ENV_VAR
                            << " to an existing weights file.\n";
                  return 1;
            }

            model.load(model_path);
            std::cout << "weights:    " << model_path << "\n";
            std::cout << "max_tokens: " << chat_tokens << "\n";

            if (!sampler.system_prompt.empty())
                  std::cout << "system:     " << sampler.system_prompt << "\n";

            run_chat(model, dl, chat_tokens, sampler);
            return 0;
      }

      if (gen_mode)
      {
            if (!file_exists(model_path))
            {
                  std::cerr << "[ERROR] Cannot generate because model weights were not found at "
                            << model_path << "\n";
                  std::cerr << "[HINT]  Train first, or set " << MODEL_PATH_ENV_VAR
                            << " to an existing weights file.\n";
                  return 1;
            }

            model.load(model_path);
            std::cout << "\ngenerating:\n";
            std::vector<int> ctx = {0};
            while (!g_interrupted)
            {
                  ctx = model.generate(ctx, 1, sampler);
                  std::cout << dl.decode({ctx.back()}) << std::flush;
                  if ((int)ctx.size() > BLOCK_SIZE)
                        ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
            }
            std::cout << "\n";
            return 0;
      }

      // Training mode: build optimizer, then iterate.
      AdamWState opt = build_optimizer(model, LEARNING_RATE);
      std::mt19937 rng(SEED);

      float best_val_loss = 1e30f;
      float last_val_loss = 0.0f;

      {
            std::mt19937 init_rng(SEED);
            last_val_loss = estimate_loss(model, dl, "val", init_rng);
      }

      for (int iter = 1; iter <= MAX_ITERS && !g_interrupted; ++iter)
      {
            double step_start = wall_secs();

            std::pair<std::vector<int>, std::vector<int>> batch =
                  dl.get_batch("train", BATCH_SIZE, BLOCK_SIZE, rng);

            SavedForward saved =
                  forward_save(model, batch.first, BATCH_SIZE, BLOCK_SIZE, batch.second, true);

            float batch_loss =
                  model.forward(batch.first, BATCH_SIZE, BLOCK_SIZE, batch.second, false).second;

            Grads grads = backward(model, saved);
            apply_grads(model, grads, opt);

            double step_ms = (wall_secs() - step_start) * 1000.0;
            int tok_per_sec =
                  (step_ms > 0.0) ? (int)((long)BATCH_SIZE * BLOCK_SIZE / (step_ms / 1000.0)) : 0;

            bool better = false;
            bool val_updated = false;
            if (iter % EVAL_INTERVAL == 0 || iter == MAX_ITERS)
            {
                  last_val_loss = estimate_loss(model, dl, "val", rng);
                  val_updated = true;
                  if (last_val_loss < best_val_loss)
                  {
                        best_val_loss = last_val_loss;
                        model.save(model_path);
                        better = true;
                  }
            }

            double ram_mb = get_ram_usage_mb();

            // train loss is always the fresh loss from this step's forward pass.
            // val loss is only recomputed every EVAL_INTERVAL steps (recomputing it
            // every step would be wasteful) but the value shown is always the most
            // recently and correctly measured one — never stale/fake data. A "*"
            // marks the steps where it was just refreshed.
            std::cout << "step " << std::right << std::setw(6) << iter << "/" << MAX_ITERS
                      << " | train loss " << std::fixed << std::setprecision(6) << batch_loss
                      << " | val loss " << std::fixed << std::setprecision(6) << last_val_loss
                      << (val_updated ? "*" : " ") << " | lr " << std::scientific
                      << std::setprecision(2) << (float)LEARNING_RATE << " | " << std::fixed
                      << std::setprecision(2) << std::setw(8) << step_ms << " ms"
                      << " | " << std::setw(6) << tok_per_sec << " tok/s"
                      << " | ram " << std::setprecision(1) << ram_mb << " MB"
                      << (better ? "  best" : "") << "\n";
            std::cout.flush();

            if (iter % EVAL_INTERVAL == 0 || iter == MAX_ITERS)
            {
                  std::cout << "generating:\n";
                  sample_tokens(model, dl, iter == MAX_ITERS ? 10000 : 150, sampler);
            }
      }

      return 0;
}