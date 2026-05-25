// Run with: g++ -std=c++17 -O2 -I. -Iinclude -o Quadtrix torch_main.cpp
// Then train: ./Quadtrix engine/data/cleaned.txt
// Chat : ./Quadtrix engine/data/cleaned.txt --chat

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <csignal>
#include <sstream>
#include <algorithm>

#include "config/config.h"
#include "gpu/dataloader.h"
#include "gpu/model.h"

namespace ansi
{
      const std::string reset = "\033[0m";
      const std::string bold = "\033[1m";
      const std::string dim = "\033[2m";
      const std::string gray = "\033[90m";
      const std::string white = "\033[97m";
      const std::string cyan = "\033[36m";
      const std::string green = "\033[32m";
      const std::string yellow = "\033[33m";
      const std::string red = "\033[31m";
}

static volatile bool g_stop = false;
static void on_sigint(int) { g_stop = true; }

static std::string timestamp()
{
      auto now = std::chrono::system_clock::now();
      auto time = std::chrono::system_clock::to_time_t(now);
      char buf[20];
      std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&time));
      return buf;
}

static double wall_time()
{
      using namespace std::chrono;
      return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void log(const std::string &msg, const std::string &color = ansi::white)
{
      std::cout << ansi::gray << "[" << timestamp() << "]" << ansi::reset
                << " " << color << msg << ansi::reset << "\n";
}

void kv(const std::string &key, const std::string &val)
{
      std::cout << "  " << ansi::gray << key << ansi::reset
                << " " << ansi::white << val << ansi::reset << "\n";
}

static float estimate_loss(GPTLanguageModel &model, DataLoader &dl,
                           const std::string &split, torch::Device device,
                           std::mt19937 &rng)
{
      model->eval();
      torch::NoGradGuard no_grad;
      float total = 0.0f;
      for (int k = 0; k < EVAL_ITERS; ++k)
      {
            auto [x, y] = dl.get_batch(split, BATCH_SIZE, BLOCK_SIZE, device, rng);
            auto [logits, loss] = model->forward(x, y);
            total += loss.item<float>();
      }
      model->train();
      return total / EVAL_ITERS;
}

void chat_mode(GPTLanguageModel &model, DataLoader &dl, torch::Device device)
{
      model->eval();
      std::cout << "\n"
                << ansi::cyan << "chat mode" << ansi::reset
                << ansi::gray << " (ctrl+c to exit)" << ansi::reset << "\n\n";

      std::string input;
      while (!g_stop)
      {
            std::cout << ansi::cyan << "› " << ansi::reset;
            if (!std::getline(std::cin, input) || input.empty())
                  break;

            auto tokens = dl.encode(input);
            auto ctx = torch::from_blob(tokens.data(), {1, (long)tokens.size()},
                                        torch::kInt64)
                           .clone()
                           .to(device);

            std::cout << ansi::white;
            int max_tokens = 200, count = 0;
            while (count < max_tokens && !g_stop)
            {
                  ctx = model->generate(ctx, 1);
                  int64_t tok = ctx[0][-1].item<int64_t>();
                  std::string out = dl.decode({tok});
                  std::cout << out << std::flush;
                  if (++count > 20 && out == "\n")
                        break;
                  if (ctx.size(1) > BLOCK_SIZE)
                        ctx = ctx.slice(1, ctx.size(1) - BLOCK_SIZE);
            }
            std::cout << ansi::reset << "\n\n";
      }
}

int main(int argc, char *argv[])
{
      std::signal(SIGINT, on_sigint);

      torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

      std::cout << "\n"
                << ansi::bold << "quadtrix " << ansi::reset << "\n\n";
      kv("device", device == torch::kCUDA ? "cuda" : "cpu");
      if (device == torch::kCUDA)
      {
            kv("gpu", torch::cuda::get_device_name(0));
      }

      std::string data_path = DEFAULT_DATA_PATH;
      bool gen_mode = false, chat_flag = false;
      for (int i = 1; i < argc; ++i)
      {
            std::string a = argv[i];
            if (a == "--generate")
                  gen_mode = true;
            else if (a == "--chat")
                  chat_flag = true;
            else
                  data_path = a;
      }

      kv("batch", std::to_string(BATCH_SIZE));
      kv("block", std::to_string(BLOCK_SIZE));
      kv("iterations", std::to_string(MAX_ITERS));
      kv("learning_rate", std::to_string(LEARNING_RATE));
      kv("embedding", std::to_string(N_EMBD));
      kv("heads", std::to_string(N_HEAD));
      kv("layers", std::to_string(N_LAYER));

      std::cout << "\n";
      log("loading " + data_path);
      DataLoader dl;
      try
      {
            dl.load(data_path);
      }
      catch (const std::exception &e)
      {
            log(e.what(), ansi::red);
            return 1;
      }

      torch::manual_seed(SEED);
      if (device == torch::kCUDA)
            torch::cuda::manual_seed(SEED);
      std::mt19937 rng(SEED);

      log("building model");
      auto model = GPTLanguageModel(dl.vocab_size, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER);
      model->to(device);

      auto n_params = model->num_params();
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << n_params / 1e6 << "M parameters";
      kv("params", ss.str());

      if (chat_flag)
      {
            torch::load(model, BEST_MODEL_PATH);
            model->to(device);
            chat_mode(model, dl, device);
            return 0;
      }

      if (gen_mode)
      {
            torch::load(model, BEST_MODEL_PATH);
            model->to(device);
            model->eval();
            std::cout << "\n"
                      << ansi::gray << "generating (ctrl+c to stop)"
                      << ansi::reset << "\n\n";
            auto ctx = torch::zeros({1, 1}, torch::TensorOptions()
                                                .dtype(torch::kInt64)
                                                .device(device));
            while (!g_stop)
            {
                  ctx = model->generate(ctx, 1);
                  std::cout << dl.decode({ctx[0][-1].item<int64_t>()}) << std::flush;
                  if (ctx.size(1) > BLOCK_SIZE)
                        ctx = ctx.slice(1, ctx.size(1) - BLOCK_SIZE);
            }
            std::cout << "\n";
            return 0;
      }

      torch::optim::AdamW optimizer(model->parameters(),
                                    torch::optim::AdamWOptions(LEARNING_RATE)
                                        .betas({0.9, 0.999})
                                        .eps(1e-8)
                                        .weight_decay(0.0));

      std::cout << "\n"
                << ansi::bold << "training" << ansi::reset << "\n\n";
      std::cout << ansi::gray << std::left
                << "  " << std::setw(12) << "iter"
                << std::setw(10) << "train"
                << std::setw(10) << "val"
                << std::setw(10) << "time"
                << std::setw(10) << "eta"
                << ansi::reset << "\n";

      float best_val = 1e30f;
      double t0 = wall_time();
      model->train();

      for (int iter = 0; iter <= MAX_ITERS && !g_stop; ++iter)
      {
            if (iter % EVAL_INTERVAL == 0 || iter == MAX_ITERS)
            {
                  float tl = estimate_loss(model, dl, "train", device, rng);
                  float vl = estimate_loss(model, dl, "val", device, rng);
                  double elapsed = wall_time() - t0;
                  double eta = iter > 0 ? elapsed / iter * (MAX_ITERS - iter) : 0.0;
                  bool saved = vl < best_val;

                  if (saved)
                  {
                        best_val = vl;
                        torch::save(model, BEST_MODEL_PATH);
                  }

                  std::string color = tl < 1.5 ? ansi::green : tl < 2.5 ? ansi::yellow
                                                                        : ansi::red;
                  std::cout << "  " << std::left << std::setw(12)
                            << (std::to_string(iter) + "/" + std::to_string(MAX_ITERS))
                            << color << std::fixed << std::setprecision(4)
                            << std::setw(10) << tl << std::setw(10) << vl << ansi::reset
                            << ansi::gray << std::setprecision(0)
                            << std::setw(8) << (int)elapsed << "s"
                            << std::setw(8) << (int)eta << "s" << ansi::reset;
                  if (saved)
                        std::cout << ansi::green << " ✓" << ansi::reset;
                  std::cout << "\n"
                            << std::flush;

                  if (iter == MAX_ITERS)
                        break;
            }

            model->train();
            auto [xb, yb] = dl.get_batch("train", BATCH_SIZE, BLOCK_SIZE, device, rng);
            auto [logits, loss] = model->forward(xb, yb);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
      }

      double total = wall_time() - t0;
      std::cout << "\n";
      log("training complete");
      ss.str("");
      ss << std::fixed << std::setprecision(1) << total << "s";
      kv("time", ss.str());
      ss.str("");
      ss << std::fixed << std::setprecision(4) << best_val;
      kv("best_val", ss.str());

      if (device == torch::kCUDA)
      {
            ss.str("");
            ss << torch::cuda::max_memory_allocated(0) / 1024 / 1024 << "MB";
            kv("peak_vram", ss.str());
      }

      std::cout << "\n"
                << ansi::gray << "generating (ctrl+c to stop)"
                << ansi::reset << "\n\n";
      torch::load(model, BEST_MODEL_PATH);
      model->to(device);
      model->eval();

      auto ctx = torch::zeros({1, 1}, torch::TensorOptions()
                                          .dtype(torch::kInt64)
                                          .device(device));
      while (!g_stop)
      {
            ctx = model->generate(ctx, 1);
            std::cout << dl.decode({ctx[0][-1].item<int64_t>()}) << std::flush;
            if (ctx.size(1) > BLOCK_SIZE)
                  ctx = ctx.slice(1, ctx.size(1) - BLOCK_SIZE);
      }

      std::cout << "\n";
      return 0;
}