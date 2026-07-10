# llm.cpp

<h1 align="center">
<img  width="957" height="233" alt="image" src="https://github.com/user-attachments/assets/bf4511d8-4fb4-449c-b8e7-e936e4d8d164" />
</h1>

[![Build & Test](https://github.com/LMGNU/llm.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/LMGNU/llm.cpp/actions/workflows/ci.yml)  [![Docker Images](https://github.com/LMGNU/llm.cpp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/LMGNU/llm.cpp/actions/workflows/docker-publish.yml) [![Release](https://github.com/LMGNU/llm.cpp/actions/workflows/release.yml/badge.svg)](https://github.com/LMGNU/llm.cpp/actions/workflows/release.yml) [![Deploy Pages](https://github.com/LMGNU/llm.cpp/actions/workflows/jekyll-gh-pages.yml/badge.svg)](https://github.com/LMGNU/llm.cpp/actions/workflows/jekyll-gh-pages.yml)


Language models in dependency-free C++, with no need for 245MB of PyTorch or 107MB of Python to make a transformer actually works. The native path is a decoder-only GPT: tensors, embeddings, multi-head causal self-attention, layer norm, cross-entropy, and a analytical backward pass with AdamW, all in [main.cpp](main.cpp) and [include/](include/). No autograd, no framework - every gradient is derived and written out.

Alongside it sits a parallel PyTorch implementation in [engine/main.py](engine/main.py) and [engine/inference.py](engine/inference.py), so you can train and generate the same architecture with `torch` + `tiktoken` when you want speed instead of transparency. There's also an experimental integrated-GPU path in [iGPU/](engine/iGPU/).

The point of this repo is the C++ core. The PyTorch, FastAPI, and frontend layers exist to make the model usable, but if you're here to learn how a GPT is actually built and trained without a framework doing the work for you, [include/backward.h](include/backward.h) is where to start reading.

***technical notes***: [docs](https://eamon2009.github.io/LLMs/)


## quick start (C++, train + chat)

The fastest way to see the whole pipeline - tokenize, train, checkpoint, generate - using the bundled character-level corpus:

```bash
g++ -std=c++17 -O2 -I. -Iinclude -o quadtrix.exe main.cpp
./quadtrix.exe data/input.txt
```

This trains from scratch on `data/input.txt` and writes the best checkpoint to `best_model.bin`. Once you have a checkpoint, generate or chat with it:

```bash
./quadtrix.exe data/input.txt --generate
./quadtrix.exe data/input.txt --chat --chat-tokens 300
```

debugging tip: drop `-O2` for `-g` when compiling if you want to step through `include/backward.h` or `include/gpt.h` in a debugger — the manual backward pass is much easier to follow one breakpoint at a time.

### runtime arguments

```bash
quadtrix.exe [data_path] [--generate] [--chat] [--chat-tokens N]
```

| Argument | Description |
|---|---|
| `data_path` | Plain-text corpus used to build the tokenizer and train/validation split |
| `--generate` | Load weights and continuously generate text |
| `--chat` | Load weights and start interactive terminal chat |
| `--chat-tokens N` | Max generated tokens per chat response |

| Env var | Default | Description |
|---|---|---|
| `GPT_DATA_PATH` | `data/input.txt` | Override the default training corpus |
| `GPT_MODEL_PATH` | `best_model.bin` | Override the checkpoint path |

## what's actually implemented in C++

No third-party runtime dependency — it builds from `main.cpp`, `config/config.h`, and `include/*.h` alone.

- Character-level tokenizer built directly from the input corpus
- Train/validation split via `DataLoader`
- Token + positional embeddings
- Multi-head causal self-attention with explicit QKV projections
- Pre-layer-norm residual transformer blocks
- Feed-forward MLP with ReLU
- Cross-entropy loss
- **Fully analytical backward pass** — every gradient (attention, layer norm, MLP, embeddings) is derived and coded in `include/backward.h`, not autograd
- AdamW optimizer (first/second moment estimates, weight decay)
- Checkpoint save/load
- Autoregressive generation and terminal chat mode

Hyperparameters live in `config/config.h` and require a rebuild to take effect:

```cpp
static const int BATCH_SIZE   = 4;
static const int BLOCK_SIZE   = 64;
static const int N_EMBD       = 128;
static const int N_HEAD       = 4;
static const int N_LAYER      = 4;
static const float DROPOUT    = 0.2f;
static const float LEARNING_RATE = 3e-4f;
static const int MAX_ITERS    = 3000;
```

For an optimized native build:

```bash
g++ -std=c++17 -O3 -march=native -I. -Iinclude -o quadtrix.exe main.cpp
```

## the PyTorch reference path

[engine/main.py](engine/main.py) trains the same architectural idea with `torch`, `torch.nn`, and GPT-2 BPE tokenization via `tiktoken`, useful when you want to scale past what C++ loops can comfortably train on CPU.

```bash
python engine/main.py
```

It looks for `engine/input.txt` by default; point it elsewhere with `QUADTRIX_TRAIN_DATA` if needed. Run inference against a saved checkpoint:

```bash
python engine/inference.py --checkpoint engine/best_model.pt --prompt "Once upon a time" --max-new-tokens 100
```

## results so far
## Leaderboard

Runs are ranked by validation loss. Lower is better.

| # | Val Loss | Parameters | Time     | Hardware | Description                              |
|--:|----------|------------|----------|----------|------------------------------------------|
| 1 | **0.7176**   | 10.82M     | 61.3 min | NVIDIA T4    | Large-scale run, coherent paragraphs, strong convergence |
| 2 | 0.9250   | 1.99M      | 6.1 min  | NVIDIA T4   | Optimised run, fast training, stable learning            |
| 3 | 1.3145   | 0.82M      | 39.4 min | x64 CPU     | Baseline, small data                                     |
| 4 | 1.6371   | 0.82M      | 76.2 min | x64 CPU      | Extended CPU training, 3000 iterations                   |

All runs: @Eamon2009, 2026.

## Benchmarks

### Runs at a Glance

| Metric            | Character-Level | Small Scale | Large Scale |
|-------------------|------------------|-------------|-------------|
| Parameters        | 0.83M            | 2.00M       | 19.17M      |
| Layers            | 4                | 4           | 4           |
| Embedding dim     | 128              | 200         | 200         |
| Attention heads   | 4                | 4           | 4           |
| Context length    | 64               | 200         | 200         |
| Vocab             | 105 char         | 110 char    | ~50K BPE    |
| Corpus            | TinyStories      | TinyStories | Children's Stories |
| Iterations        | 3,000            | 5,000       | 5,000       |
| Train loss        | 1.5632           | 0.9045      | —           |
| Val loss          | 1.6371           | 0.9301      | —           |
| Gen. gap          | 0.0739           | 0.0256      | —           |

See [run.md](run.md) and the leaderboard in the full docs for more configurations.

## how this differs from similar projects

| Project | Focus | Language | Autograd |
|---|---|---|---|
| nanoGPT / minGPT | Minimal, educational GPT training | Python | PyTorch |
| llama2.c | Inference-only | C | None |
| **llm.cpp** | Training *and* inference, manual backward pass, web UI | C++ / Python / TypeScript | Manual (C++) + PyTorch |

I'd like the C++ core (`main.cpp`, `include/`, `config/`) to stay dependency-free and to stay the part of this repo that explin transformer internals directly. The PyTorch engine, FastAPI middleware, and React frontend are welcome to grow more features, integrations, and UI polish. If you build a port to another language or framework, I'm happy to link to it from a notable-forks section; just open an issue or PR.

## references

- Vaswani et al., "Attention Is All You Need", 2017
- Radford et al., GPT-2 technical work, 2019
- nanoGPT and minGPT as educational reference points

## license

GPL-3.O

