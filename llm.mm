#import <Foundation/Foundation.h>

#include "config/config.h"
#include "include/backward.h"
#include "include/dataloader.h"
#include "include/gpt.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static volatile bool g_interrupted = false;
static void sig_handler(int)
{
      g_interrupted = true;
}

static std::string now_str()
{
      std::time_t t = std::time(nullptr);
      char buf[32];
      std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
      return buf;
}

static double wall_secs()
{
      using namespace std::chrono;
      return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static NSString* choose_existing_path(NSString *requested_path, NSString *argv0)
{
      if (requested_path.length == 0) return requested_path;
      
      NSFileManager *fm = [NSFileManager defaultManager];
      if ([fm fileExistsAtPath:requested_path]) return requested_path;
      if ([requested_path isAbsolutePath]) return requested_path;

      NSString *exe_dir = [argv0 stringByDeletingLastPathComponent];
      
      NSArray *candidates = @[
          [exe_dir stringByAppendingPathComponent:requested_path],
          [@"." stringByAppendingPathComponent:requested_path]
      ];

      for (NSString *candidate in candidates)
      {
          if ([fm fileExistsAtPath:candidate]) return candidate;
      }
      return requested_path;
}

static NSString* choose_output_path(NSString *requested_path, NSString *argv0)
{
      if (requested_path.length == 0 || [requested_path isAbsolutePath])
          return requested_path;

      NSFileManager *fm = [NSFileManager defaultManager];
      NSString *exe_relative = [[argv0 stringByDeletingLastPathComponent] stringByAppendingPathComponent:requested_path];
      
      if ([fm fileExistsAtPath:requested_path] || ![fm fileExistsAtPath:exe_relative])
          return requested_path;
          
      return exe_relative;
}

static void sample_tokens(GPTLanguageModel &model, DataLoader &dl, int n_tokens)
{
      std::vector<int> ctx = {0};
      for (int i = 0; i < n_tokens; ++i)
      {
            ctx = model.generate(ctx, 1);
            std::cout << dl.decode({ctx.back()}) << std::flush;
            if ((int)ctx.size() > BLOCK_SIZE)
                  ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
      }
      std::cout << "\n";
}

static float estimate_loss(GPTLanguageModel &model, DataLoader &dl, const std::string &split, std::mt19937 &rng)
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

static void run_chat(GPTLanguageModel &model, DataLoader &dl, int max_new_tokens)
{
      std::cout << "\n" << std::string(60, '=') << "\n";
      std::cout << "  llm.cpp CHAT MODE (Objective-C++ Build)\n";
      std::cout << "  Type your prompt and press Enter. "
                   "Type 'quit' or 'exit' to leave.\n";
      std::cout << std::string(60, '=') << "\n\n";

      while (!g_interrupted)
      {
            std::cout << "\033[1;32muser>\033[0m ";
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

            std::vector<int> ctx = dl.encode(prompt);
            if (ctx.empty())
            {
                  ctx = {0};
            }

            if ((int)ctx.size() > BLOCK_SIZE)
                  ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());

            std::cout << "\033[1;36mllm>\033[0m ";
            std::cout.flush();

            for (int tok = 0; tok < max_new_tokens && !g_interrupted; ++tok)
            {
                  ctx = model.generate(ctx, 1);
                  std::cout << dl.decode({ctx.back()}) << std::flush;

                  if ((int)ctx.size() > BLOCK_SIZE)
                        ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
            }
            std::cout << "\n\n";
      }
}

int main(int argc, char *argv[])
{
      std::signal(SIGINT, sig_handler);
      
      @autoreleasepool {
            std::cout << "llm.cpp (Objective-C++)\n";

            NSProcessInfo *processInfo = [NSProcessInfo processInfo];
            NSDictionary *environment = [processInfo environment];
            NSArray *arguments = [processInfo arguments];
            NSString *argv0 = arguments.firstObject;

            NSString *ns_data_path = environment[@(DATA_PATH_ENV_VAR.c_str())] ?: @(DEFAULT_CLEANED_PATH.c_str());
            NSString *ns_model_path = environment[@(MODEL_PATH_ENV_VAR.c_str())] ?: @(BEST_MODEL_PATH.c_str());

            bool gen_mode = false;
            bool chat_mode = false;
            int chat_tokens = 200;

            for (NSUInteger i = 1; i < arguments.count; ++i)
            {
                  NSString *a = arguments[i];
                  if ([a isEqualToString:@"--generate"])
                        gen_mode = true;
                  else if ([a isEqualToString:@"--chat"])
                        chat_mode = true;
                  else if ([a isEqualToString:@"--chat-tokens"] && i + 1 < arguments.count)
                        chat_tokens = [arguments[++i] intValue];
                  else
                        ns_data_path = a;
            }

            ns_data_path = choose_existing_path(ns_data_path, argv0);
            ns_model_path = choose_output_path(ns_model_path, argv0);

            std::string data_path = [ns_data_path UTF8String];
            std::string model_path = [ns_model_path UTF8String];
            
            NSFileManager *fm = [NSFileManager defaultManager];

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
            std::cout << "max_seq_len: " << BLOCK_SIZE << "\n";
            std::cout << "vocab_size: " << dl.vocab_size << "\n";
            std::cout << "num_layers: " << N_LAYER << "\n";
            std::cout << "num_heads: " << N_HEAD << "\n";
            std::cout << "channels: " << N_EMBD << "\n";
            std::cout << "num_parameters: " << n_params << "\n";

            if (chat_mode)
            {
                  if (![fm fileExistsAtPath:ns_model_path])
                  {
                        std::cerr << "[ERROR] Cannot start chat because model weights were not found at "
                                  << model_path << "\n";
                        return 1;
                  }

                  model.load(model_path);
                  std::cout << "weights: " << model_path << "\n";
                  std::cout << "max_tokens: " << chat_tokens << "\n";

                  run_chat(model, dl, chat_tokens);
                  return 0;
            }

            if (gen_mode)
            {
                  if (![fm fileExistsAtPath:ns_model_path])
                  {
                        std::cerr << "[ERROR] Cannot generate because model weights were not found at "
                                  << model_path << "\n";
                        return 1;
                  }

                  model.load(model_path);
                  std::cout << "\ngenerating:\n";
                  std::vector<int> ctx = {0};
                  while (!g_interrupted)
                  {
                        ctx = model.generate(ctx, 1);
                        std::cout << dl.decode({ctx.back()}) << std::flush;
                        if ((int)ctx.size() > BLOCK_SIZE)
                              ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
                  }
                  std::cout << "\n";
                  return 0;
            }

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

                  SavedForward saved = forward_save(model,
                                                    batch.first,
                                                    BATCH_SIZE,
                                                    BLOCK_SIZE,
                                                    batch.second,
                                                    true);

                  float batch_loss =
                        model.forward(batch.first, BATCH_SIZE, BLOCK_SIZE, batch.second, false).second;

                  Grads grads = backward(model, saved);
                  apply_grads(model, grads, opt);

                  double step_ms = (wall_secs() - step_start) * 1000.0;
                  int tok_per_sec =
                        (step_ms > 0.0) ? (int)((long)BATCH_SIZE * BLOCK_SIZE / (step_ms / 1000.0)) : 0;

                  bool better = false;
                  if (iter % EVAL_INTERVAL == 0 || iter == MAX_ITERS)
                  {
                        last_val_loss = estimate_loss(model, dl, "val", rng);
                        if (last_val_loss < best_val_loss)
                        {
                              best_val_loss = last_val_loss;
                              model.save(model_path);
                              better = true;
                        }
                  }

                  std::cout << "step" << std::setw(5) << iter << "/" << MAX_ITERS << " | loss "
                            << std::fixed << std::setprecision(6) << batch_loss << " | val " << std::fixed
                            << std::setprecision(6) << last_val_loss << " | lr " << std::scientific
                            << std::setprecision(2) << (float)LEARNING_RATE << " | " << std::fixed
                            << std::setprecision(2) << step_ms << " ms"
                            << " | " << tok_per_sec << " tok/s" << (better ? "  *best*" : "") << "\n";
                  std::cout.flush();

                  if (iter % EVAL_INTERVAL == 0 || iter == MAX_ITERS)
                  {
                        std::cout << "generating:\n";
                        sample_tokens(model, dl, iter == MAX_ITERS ? 10000 : 150);
                  }
            }
      }
      return 0;
}