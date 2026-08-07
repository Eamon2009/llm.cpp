
/**
 * @file   llm.cu
 * @author Eamon Sippy 
 * @Copyright (c) 2026 Eamon Sippy . All rights reserved.
 */

#include "config/config.h"
#include "include/backward.h"
#include "include/bpe.h"
#include "include/lm.h"
#include "include/sampler.h"
#include "include/cuda_kernels.cuh"
#include "include/cuda_layers.cuh"
#include "include/cuda_tensor.cuh"
#include "include/cuda_utils.cuh"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Global state
// ------------------------------------------------------------------

/**
 * @brief Global cuBLAS context handle.
 *
 * Created at startup, destroyed at exit. All GPU matmul ops use this handle.
 */
cublasHandle_t g_cublas_handle = nullptr;

/**
 * @brief Global interrupt flag. Set by SIGINT handler.
 *
 * Checked in training loop and generation loops to enable graceful shutdown.
 */
static volatile bool g_interrupted = false;

/**
 * @brief SIGINT handler. Sets g_interrupted to true.
 */
static void sig_handler(int) { g_interrupted = true; }

// ------------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------------

/**
 * @brief Current wall-clock time as formatted string.
 * @return "YYYY-MM-DD HH:MM:SS".
 */
static std::string now_str()
{
      std::time_t t = std::time(nullptr);
      char buf[32];
      std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
      return buf;
}

/**
 * @brief Monotonic wall-clock time in seconds.
 * @return Seconds since an arbitrary epoch.
 */
static double wall_secs()
{
      using namespace std::chrono;
      return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/**
 * @brief Check if file exists and is readable.
 * @param path File path.
 * @return True if file can be opened.
 */
static bool file_exists(const std::string &path)
{
      std::ifstream f(path.c_str(), std::ios::binary);
      return f.good();
}

/**
 * @brief Extract directory portion of a path.
 * @param path File path.
 * @return Directory string, or "." for bare filenames.
 */
static std::string dir_name(const std::string &path)
{
      std::string::size_type pos = path.find_last_of("/\");
      if (pos == std::string::npos)
            return ".";
      if (pos == 0)
            return path.substr(0, 1);
      return path.substr(0, pos);
}

/**
 * @brief Check if path is absolute.
 * @param path File path.
 * @return True if starts with drive letter or slash.
 */
static bool is_absolute_path(const std::string &path)
{
      if (path.empty())
            return false;
      if (path.size() > 1 && path[1] == ':')
            return true;
      return path[0] == '/' || path[0] == '\';
}

/**
 * @brief Join base directory and child path.
 * @param base  Base directory.
 * @param child Child path.
 * @return      Joined path with separator.
 */
static std::string join_path(const std::string &base, const std::string &child)
{
      if (base.empty() || base == ".")
            return child;
      char last = base[base.size() - 1];
      if (last == '/' || last == '\')
            return base + child;
      return base + "/" + child;
}

/**
 * @brief Resolve data path, falling back to executable-relative locations.
 *
 * @param requested_path User-specified or default path.
 * @param argv0          Executable path (argv[0]).
 * @return               Existing file path or original if not found.
 */
static std::string choose_existing_path(const std::string &requested_path,
                                         const std::string &argv0)
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

/**
 * @brief Resolve output path, preferring executable-relative when writable.
 *
 * @param requested_path User-specified or default path.
 * @param argv0          Executable path (argv[0]).
 * @return               Resolved output path.
 */
static std::string choose_output_path(const std::string &requested_path,
                                       const std::string &argv0)
{
      if (requested_path.empty() || is_absolute_path(requested_path))
            return requested_path;

      std::string exe_relative = join_path(dir_name(argv0), requested_path);
      if (file_exists(requested_path) || !file_exists(exe_relative))
            return requested_path;
      return exe_relative;
}

/**
 * @brief Print CLI usage summary.
 * @param argv0 Program name.
 */
static void print_usage(const char *argv0)
{
      std::cout << "Usage: " << argv0 << " [options] [data_file]
"
                << "
"
                << "  --generate               run inference only (needs saved weights)
"
                << "  --chat                   start interactive chat mode
"
                << "  --chat-tokens N          max tokens per chat reply (default "
                << DEFAULT_CHAT_TOKENS << ")
"
                << "  --system TEXT            system prompt prepended to every chat turn
"
                << "  --rep-penalty F          repetition penalty, 1.0 means off (default "
                << DEFAULT_REP_PENALTY << ")
"
                << "  --rep-window N           recent token window for penalty (default "
                << DEFAULT_REP_WINDOW << ")
"
                << "  --help                   show this message
";
}

/**
 * @brief Autoregressively sample and print tokens.
 *
 * @param model     GPTLanguageModel instance.
 * @param dl        DataLoader for decode.
 * @param n_tokens  Number of tokens to generate.
 * @param params    Sampler parameters.
 */
static void
sample_tokens(GPTLanguageModel &model, DataLoader &dl, int n_tokens, const SamplerParams &params)
{
      std::vector<int> ctx = {0};
      for (int i = 0; i < n_tokens && !g_interrupted; ++i)
      {
            ctx = model.generate(ctx, 1, params);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cout << dl.decode({ctx.back()}) << std::flush;
            if ((int)ctx.size() > BLOCK_SIZE)
                  ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
      }
      std::cout << "
";
}

/**
 * @brief Estimate average validation loss over EVAL_ITERS batches.
 *
 * @param model  GPTLanguageModel instance.
 * @param dl     DataLoader.
 * @param split  "train" or "val".
 * @param rng    MT19937 for batch sampling.
 * @return       Mean cross-entropy loss.
 */
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
            CUDA_CHECK(cudaDeviceSynchronize());
            total += result.second;
      }
      return total / EVAL_ITERS;
}

/**
 * @brief Cosine learning-rate schedule with linear warmup.
 *
 * Warmup: linear ramp from 0 to max_lr over first 10% of iterations.
 * Decay: cosine annealing from max_lr to 0.1*max_lr over remaining 90%.
 *
 * @param it         Current iteration (1-based).
 * @param max_lr     Peak learning rate.
 * @param max_iters  Total training iterations.
 * @return           Scheduled learning rate for this step.
 */
static float get_lr(int it, float max_lr, int max_iters)
{
      int warmup_iters = max_iters / 10;
      if (warmup_iters == 0)
            warmup_iters = 1;

      float min_lr = max_lr * 0.1f;

      if (it <= warmup_iters)
      {
            return max_lr * (float)it / (float)warmup_iters;
      }
      if (it > max_iters)
      {
            return min_lr;
      }

      float decay_ratio = (float)(it - warmup_iters) / (float)(max_iters - warmup_iters);
      float coeff = 0.5f * (1.0f + std::cos(3.14159265358979323846f * decay_ratio));

      return min_lr + coeff * (max_lr - min_lr);
}

/**
 * @brief Build chat turn context from system and user tokens.
 *
 * Concatenates system tokens (if any) with user tokens.
 * Crops to BLOCK_SIZE if exceeded.
 *
 * @param sys_tokens  System prompt token IDs.
 * @param user_tokens User input token IDs.
 * @return            Combined context vector.
 */
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

/**
 * @brief Run interactive chat loop.
 *
 * @param model          GPTLanguageModel instance.
 * @param dl             DataLoader for encode/decode.
 * @param max_new_tokens Max tokens per reply.
 * @param params         Sampler parameters.
 */
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
                               "All characters may be outside the training vocabulary.
";
            }
            else
            {
                  std::cout << "[CHAT]  System prompt active (" << sys_tokens.size()
                            << " tokens, " << BLOCK_SIZE - (int)sys_tokens.size()
                            << " tokens left for user input)
";
            }
      }

      std::cout << "
" << std::string(60, '=') << "
";
      std::cout << "  CHAT MODE (CUDA Accelerated)
";
      std::cout << "  Type your prompt and press Enter.
";
      std::cout << "  Type quit or exit to leave.
";
      std::cout << std::string(60, '=') << "

";

      while (!g_interrupted)
      {
            std::cout << "\033[1;32mYou>\033[0m ";
            std::cout.flush();

            std::string prompt;
            if (!std::getline(std::cin, prompt))
                  break;

            size_t s = prompt.find_first_not_of(" \t\r
");
            size_t e = prompt.find_last_not_of(" \t\r
");
            if (s == std::string::npos)
                  continue;
            prompt = prompt.substr(s, e - s + 1);

            if (prompt == "quit" || prompt == "exit")
            {
                  std::cout << "[Chat] Bye!
";
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
                  CUDA_CHECK(cudaDeviceSynchronize());
                  std::cout << dl.decode({ctx.back()}) << std::flush;

                  if ((int)ctx.size() > BLOCK_SIZE)
                        ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
            }
            std::cout << "

";
      }
}

/**
 * @brief Program entry point.
 *
 * Initializes CUDA/cuBLAS, parses CLI flags, loads data, builds model,
 * then trains, generates, or chats.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return     Exit code (0 on success, 1 on error).
 */
int main(int argc, char *argv[])
{
      std::signal(SIGINT, sig_handler);

      std::cout << " [llm.cu - CUDA Engine]
";

      // ------------------------------------------------------------------
      // Initialize GPU Driver and cuBLAS Context
      // ------------------------------------------------------------------
      int device_count = 0;
      CUDA_CHECK(cudaGetDeviceCount(&device_count));
      if (device_count == 0)
      {
            std::cerr << "[ERROR] No CUDA-capable GPU detected!" << std::endl;
            return 1;
      }

      cudaDeviceProp prop;
      CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
      std::cout << "[GPU] Active Device: " << prop.name
                << " | VRAM: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;

      CUBLAS_CHECK(cublasCreate(&g_cublas_handle));

      // ------------------------------------------------------------------
      // Resolve paths
      // ------------------------------------------------------------------
      std::string data_path = DEFAULT_CLEANED_PATH;
      std::string model_path = BEST_MODEL_PATH;

      const char *env_data = std::getenv(DATA_PATH_ENV_VAR.c_str());
      const char *env_model = std::getenv(MODEL_PATH_ENV_VAR.c_str());
      if (env_data != nullptr && env_data[0] != '\0')
            data_path = env_data;
      if (env_model != nullptr && env_model[0] != '\0')
            model_path = env_model;

      // ------------------------------------------------------------------
      // Parse CLI flags
      // ------------------------------------------------------------------
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
                  CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
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

      // ------------------------------------------------------------------
      // Load data
      // ------------------------------------------------------------------
      DataLoader dl;
      try
      {
            dl.load(data_path);
      }
      catch (const std::exception &e)
      {
            std::cerr << e.what() << "
";
            std::cerr << "[HINT]  Put your text at " << DEFAULT_CLEANED_PATH
                      << ", pass a file path as the first argument, or set "
                      << DATA_PATH_ENV_VAR << ".
";
            CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
            return 1;
      }

      // ------------------------------------------------------------------
      // Build model
      // ------------------------------------------------------------------
      GPTLanguageModel model(dl.vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, SEED);

      long n_params = model.num_params();

      // ***Architecture Table***
      std::cout << "
";
      std::cout << "  "
                   "+------------------------------------------+-----------------------------------"
                   "-------+
";
      std::cout << "  | " << std::left << std::setw(83) << "LLM Architecture" << " |
";
      std::cout << "  "
                   "+------------------------------------------+-----------------------------------"
                   "-------+
";
      std::cout << "  | Max Context Length   : " << std::left << std::setw(17) << BLOCK_SIZE
                << " | Vocab Size (BPE)     : " << std::left << std::setw(17) << dl.vocab_size
                << " |
";
      std::cout << "  | Number of Layers     : " << std::left << std::setw(17) << N_LAYER
                << " | Attention Heads      : " << std::left << std::setw(17) << N_HEAD
                << " |
";
      std::cout << "  | Embedding Channels   : " << std::left << std::setw(17) << N_EMBD
                << " | Total Parameters     : " << std::left << std::setw(17) << n_params
                << " |
";
      std::cout << "  | Repetition Penalty   : " << std::left << std::setw(17) << rep_penalty
                << " | Repetition Window    : " << std::left << std::setw(17) << rep_window
                << " |
";
      std::cout << "  "
                   "+------------------------------------------+-----------------------------------"
                   "-------+

";

      // ***GPU Hardware Table***
      std::cout << "  "
                   "+------------------------------------------------------------------------------"
                   "-------+
";
      std::cout << "  | " << std::left << std::setw(83) << "GPU Hardware Specs" << " |
";
      std::cout << "  "
                   "+------------------------------------------------------------------------------"
                   "-------+
";
      std::cout << "  | GPU Device           : " << std::left << std::setw(60) << prop.name
                << " |
";
      std::cout << "  | GPU VRAM (Total)     : " << std::left << std::setw(60)
                << (std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB")
                << " |
";
      std::cout << "  "
                   "+------------------------------------------------------------------------------"
                   "-------+

";

      std::cout << std::right;

      std::cout << "max_seq_len:    " << BLOCK_SIZE << "
";
      std::cout << "vocab_size:     " << dl.vocab_size << "
";
      std::cout << "num_layers:     " << N_LAYER << "
";
      std::cout << "num_heads:      " << N_HEAD << "
";
      std::cout << "channels:       " << N_EMBD << "
";
      std::cout << "num_parameters: " << n_params << "
";
      std::cout << "rep_penalty:    " << rep_penalty << "
";
      std::cout << "rep_window:     " << rep_window << "
";

      // ------------------------------------------------------------------
      // Chat mode
      // ------------------------------------------------------------------
      if (chat_mode)
      {
            if (!file_exists(model_path))
            {
                  std::cerr << "[ERROR] Cannot start chat because model weights were not found at "
                            << model_path << "
";
                  std::cerr << "[HINT]  Train first, or set " << MODEL_PATH_ENV_VAR
                            << " to an existing weights file.
";
                  CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
                  return 1;
            }

            model.load(model_path);
            std::cout << "weights:    " << model_path << "
";
            std::cout << "max_tokens: " << chat_tokens << "
";

            if (!sampler.system_prompt.empty())
                  std::cout << "system:     " << sampler.system_prompt << "
";

            run_chat(model, dl, chat_tokens, sampler);
            CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
            return 0;
      }

      // ------------------------------------------------------------------
      // Generate mode
      // ------------------------------------------------------------------
      if (gen_mode)
      {
            if (!file_exists(model_path))
            {
                  std::cerr
                        << "[ERROR] Cannot generate because model weights were not found at "
                        << model_path << "
";
                  std::cerr << "[HINT]  Train first, or set " << MODEL_PATH_ENV_VAR
                            << " to an existing weights file.
";
                  CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
                  return 1;
            }

            model.load(model_path);
            std::cout << "
generating:
";
            std::vector<int> ctx = {0};
            while (!g_interrupted)
            {
                  ctx = model.generate(ctx, 1, sampler);
                  CUDA_CHECK(cudaDeviceSynchronize());
                  std::cout << dl.decode({ctx.back()}) << std::flush;
                  if ((int)ctx.size() > BLOCK_SIZE)
                        ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
            }
            std::cout << "
";
            CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
            return 0;
      }

      // ------------------------------------------------------------------
      // Training mode
      // ------------------------------------------------------------------
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

            // Cosine LR schedule with warmup (matches main.cpp)
            float current_lr = get_lr(iter, LEARNING_RATE, MAX_ITERS);
            opt.lr = current_lr;

            std::pair<std::vector<int>, std::vector<int>> batch =
                  dl.get_batch("train", BATCH_SIZE, BLOCK_SIZE, rng);

            SavedForward saved =
                  forward_save(model, batch.first, BATCH_SIZE, BLOCK_SIZE, batch.second, true);

            float batch_loss =
                  model.forward(batch.first, BATCH_SIZE, BLOCK_SIZE, batch.second, false).second;

            Grads grads = backward(model, saved);
            apply_grads(model, grads, opt);

            CUDA_CHECK(cudaDeviceSynchronize());

            double step_ms = (wall_secs() - step_start) * 1000.0;
            int tok_per_sec =
                  (step_ms > 0.0) ? (int)((long)BATCH_SIZE * BLOCK_SIZE / (step_ms / 1000.0)) : 0;

            bool val_updated = false;
            bool better = false;

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

            double percent_done = ((double)iter / MAX_ITERS) * 100.0;

            // Table-formatted step print with LR, matching main.cpp style
            std::cout << "step " << iter << "/" << MAX_ITERS << "(" << std::fixed
                      << std::setprecision(2) << percent_done << "%)"
                      << " | train loss " << std::fixed << std::setprecision(6) << batch_loss
                      << " | val loss " << std::fixed << std::setprecision(6) << last_val_loss
                      << (val_updated ? "*" : " ")
                      << " | lr " << std::scientific << std::setprecision(2) << current_lr
                      << " | " << std::fixed << std::setprecision(2) << std::setw(8) << step_ms
                      << " ms"
                      << " | " << std::setw(6) << tok_per_sec << " tok/s"
                      << (better ? "  best" : "") << "
";
            std::cout.flush();

            if (iter % EVAL_INTERVAL == 0 || iter == MAX_ITERS)
            {
                  std::cout << "generating:
";
                  sample_tokens(model, dl, iter == MAX_ITERS ? 10000 : 150, sampler);
            }
      }

      CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
      return 0;
}
