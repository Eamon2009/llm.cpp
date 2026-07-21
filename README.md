# llm.cpp

<h1 align="center">
<img width"2170" height="725" alt="image" src="https://github.com/user-attachments/assets/1049c431-2ec8-4f24-b685-68f2d69adf33" />


</h1>

[![Release](https://img.shields.io/github/v/release/LMGNU/llm.cpp)](https://github.com/LMGNU/llm.cpp/releases) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?logo=gnu)](https://www.gnu.org/licenses/gpl-3.0) [![Docker Images](https://github.com/LMGNU/llm.cpp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/LMGNU/llm.cpp/actions/workflows/docker-publish.yml) 


This project implements language models in dependency-free C++, eliminating the need for PyTorch or Python to train a transformer locally. The core implementation is a decoder-only ***GPT architecture*** featuring custom tensors, embeddings, multi-head causal self-attention, layer normalization, cross-entropy loss, and an analytical backward pass with the AdamW optimizer - all contained within [main.cpp](main.cpp) ,[llm.mm](llm.mm) and the [include/](include) directory also a ***token level [BPE]*** [dataloder.h](LMGNU/include) implementation inside [LMGNU](LMGNU) . With no autograd engine or external frameworks, every gradient is explicitly derived and written out.
The model achieves a validation loss of 1.6371 nats after 76 minutes of CPU training on 31.4 million characters, demonstrating that character-level language modeling at this scale is highly tractable on commodity hardware without external dependencies. On a GPU (CUDA/bfloat16), a validation loss of 2.3918 is reached in under 83 minutes, achieving a peak throughput of 19.6k tokens per second.

## Leaderboar
| S.No.| time | val_bpb | scale | Date | Contributors |
|---|-------------|---------|------|------|--------------|
| 1 | 14.2 hours | 0.6120 | 124M (8x A100) | 2019 | OpenAI (GPT-2 Base) |
| 2 | 2.5 hours | 0.6945 | 50M (H100) | 2024 | Anthropic (Internal Proxy) |
| 3 | 61.3 min | 0.7176 | 10.82M (T4) | Mar 2026 | @Eamon2009 |
| 4 | 3.1 hours | 0.8012 | 14M (A100) | 2024 | Meta (MobileLLM) |
| 5 | 6.1 min | 0.9250 | 1.99M (T4) | Feb 2026 | @Eamon2009 |
| 6 | 39.4 min | 1.3145 | 0.82M (CPU) | Jan 2026 | @Eamon2009 |
| 7 | 76.2 min | 1.6371 | 0.82M (CPU) | Jan 2026 | @Eamon2009 |

More broadly, the primary contribution of this work lies in its absolute transparency. Every gradient in the backward pass is explicitly written and readable, and every tensor operation is a standard C++ function. By exposing exactly what frameworks like PyTorch compute under the hood, this implementation provides a clear educational pathway. We believe that this fundamental understanding is the true foundation of genuine expertise in deep learning.
Alongside it sits a parallel PyTorch implementation in [engine/main.py](engine/main.py) and [engine/inference.py](engine/inference.py), so you can train and generate the same architecture with `torch` + `tiktoken` when you want speed instead of transparency. There's also an experimental integrated-GPU path in [iGPU/](engine/iGPU/). The point of this repo is the C++ core. The PyTorch exist to make the model usable, but if you're here to ***train a GPT without a framework*** doing the work for you, [include/backward.h](include/backward.h) is where to start reading.

---

## From CPU to GPU: The LibTorch Port
The custom C++ backend is transparent but slow: a CPU executes scalar matrix multiplication at roughly 1-10 GFLOP/s. An NVIDIA RTX 4090 delivers ∼80 TFLOP/s-an 8,000–80,000× speedup for the same computation.The LibTorch port replaces the custom backend with PyTorch’s C++ API, gaining cuBLAS-accelerated matrix operations. The transformer architecture remains unchanged only the compute layer is modified. A single line migrates
the model to GPU:
```cpp
model->to(torch::kCUDA)
```
which transfers all parameters to GPU memory; all subsequent `torch::matmul` calls dispatch to cuBLAS automatically.

---

<h1 align="center">
<img width="824" height="249" alt="image" src="https://github.com/user-attachments/assets/5c65daa2-903a-4392-85e3-56442bb82cac" />


</h1>


***technical notes***: [docs](https://eamon2009.github.io/LLMs/)

---

## quick start (C++, train + chat)

The fastest way to see the whole pipeline - tokenize, train, checkpoint, generate - using the bundled character-level corpus:

get the data set first :

``` shell
cd data             # you set the file size for data set
python data_set.py  # also get any dataset from hugging face datasets 
```

```bash
g++ -std=c++17 -O2 -I. -Iinclude -o llm.exe main.cpp
./llm.exe data/input.txt
# also
cd LMGNU
g++ -std=c++17 -O2 -I. -Iinclude -o llm.exe llm.cpp
./llm.exe ../data/input.txt
```

This trains from scratch on `data/input.txt` and writes the best checkpoint to `best_model.bin`. Once you have a checkpoint, generate or chat with it:

```bash
./llm.exe data/input.txt --generate
./llm.exe data/input.txt --chat --chat-tokens 300
```

debugging tip: drop `-O2` for `-g` when compiling if you want to step through `include/backward.h` or `include/gpt.h` in a debugger — the manual backward pass is much easier to follow one breakpoint at a time.

### runtime arguments

```bash
llm.exe [data_path] [--generate] [--chat] [--chat-tokens N]
```

---
## File structure

```text
├── .devops/                        # DevOps, Docker, & proxy configurations
│   ├── docker-compose.dev.yml      # Local development compose setup
│   ├── docker-compose.gpu.yml      # Compose setup with GPU acceleration
│   ├── docker-compose.yml          # Base Docker Compose file
│   ├── Dockerfile                  # Primary build setup
│   ├── Dockerfile.backend          # Backend service Docker setup
│   ├── Dockerfile.cpp              # Native C++ build container setup
│   ├── Dockerfile.frontend         # Frontend UI Docker setup
│   └── nginx.conf                  # Reverse proxy configuration
│
├── .github/                        # CI/CD workflows and repository templates
│   ├── ISSUE_TEMPLATE/             # Bug report & feature request templates
│   ├── workflows/                  # GitHub Actions automation workflows
│   ├── dependabot.yml              # Automated dependency update configuration
│   └── pull_request_template.md    # PR contribution template
│
├── assets/                         # Execution screenshots & benchmark visuals
├── benches/                        # Performance benchmarking & testing suites
│   ├── bech(mm).mm                 # Metal-accelerated benchmark script
│   └── bench.cpp                   # C++ execution benchmark
│
├── config/                         # Global project setup
│   └── config.h                    # C++ global configuration header
│
├── data/                           # Dataset ingestion scripts & raw samples
│   ├── data_set.py                 # Data loader preparation script
│   └── input.txt                   # Sample input dataset text
├── engine/                         # Core execution & inference engines
│   ├── distributed/                # Multi-node / multi-GPU execution
│   │   ├── infer.py                # Distributed inference pipeline
│   │   └── train.py                # Distributed training pipeline
│   ├── iGPU/                       # Integrated GPU specific runtimes
│   │   ├── inference.py
│   │   └── main.py
│   ├── llm.cpp/                    # Low-level CUDA/C++ runtime engine
│   │   ├── config/                 # Engine specific configurations
│   │   ├── include/                # CUDA kernels & C++ architecture headers
│   │   ├── best_model.bin          # Saved model binary weights
│   │   ├── llm.cu                  # CUDA engine implementation
│   │   └── train.mm                # Metal training harness
│   ├── logs/                       # Runtime log files
│   ├── fineweb_dataset.py          # FineWeb parsing logic
│   ├── inference.py                # Python inference script
│   └── main.py                     # Central Python execution entry point
│
├── include/                        # Core C++ neural network header blocks
│   ├── attention.h                 # Attention layer logic
│   ├── backward.h                  # Backpropagation algorithm utilities
│   ├── block.h                     # Transformer block assembly
│   ├── dataloader.h                # C++ data processing pipeline
│   ├── embedding.h                 # Token embedding logic
│   ├── feedforward.h               # Feed-forward layer implementation
│   ├── gpt.h                       # Full GPT model architecture
│   ├── layernorm.h                 # Layer normalization
│   ├── linear.h                    # Dense linear layer
│   ├── tensor.h                    # Tensor data structures
│   └── torch_bridge.h              # PyTorch Interoperability bridge
│
├── scripts/                        # Automation & compilation scripts
│   └── build.sh                    # Build orchestration script
│
├── train_test/                     # Prototyping & experimental scripts
│   ├── model.py                    # Model layout prototype
│   ├── test.c                      # Simple C test harness
│   └── train2.mm                   # Metal training prototype
│
├── .clang-format                   # C++ formatting rules
├── .dockerignore                   # Docker build exclusions
├── .gitattributes                  # Git repository attribute rules
├── .gitignore                      # Git version control exclusions
├── .gitmodules                     # Git submodule links
├── .python-version                 # Environment Python version
├── benchmark.cpp                   # Root benchmark runner
├── CITATION.cff                    # Citation metadata file
├── CODE_OF_CONDUCT.md              # Community behavior rules
├── CONTRIBUTING.md                 # Contributor guidelines
├── LICENSE                         # Repository license terms
├── llm.mm                          # Metal/Objective-C++ source file
├── main.cpp                        # Primary C++ entry point
├── Makefile                        # Compilation build instructions
├── mypy.ini                        # MyPy type checking setup
├── README.md                       # Main repository overview
├── requirements.txt                # Python dependencies list
├── run.md                          # Quick start execution guide
└── SECURITY.md                     # Vulnerability reporting instructions
```
---

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

## What's Actually Implemented in C++

No third-party runtime dependency - it builds from `main.cpp`, `config/config.h`, and `include/*.h` alone.

- **Byte Pair Encoding (BPE) Tokenizer** - Built entirely from scratch. Compiles a custom vocabulary directly from the training corpus by running iterative token-pair merges until it hits a targeted vocabulary threshold.
- Train/validation split via `DataLoader`
- Token + positional embeddings
- Multi-head causal self-attention with explicit QKV projections
- Pre-layer-norm residual transformer blocks
- Feed-forward MLP with ReLU
- Cross-entropy loss
- **Fully analytical backward pass** - every gradient (attention, layer norm, MLP, embeddings) is derived mathematically and coded explicitly in `include/backward.h`, not autograd
- AdamW optimizer (first/second moment estimates, weight decay)
- Checkpoint save/load
- Autoregressive generation and terminal chat mode

Hyperparameters live in `engine/llm.cpp/config/config.h` and require a rebuild to take effect:

```cpp
// note: The c++ version only runs on cpu not on GPU
static const unsigned int SEED = 1337;
static const double TRAIN_SPLIT = 0.9;
static const int BATCH_SIZE = 32; 
static const int BLOCK_SIZE = 64; 
static const int MAX_ITERS = 5000;
static const int EVAL_INTERVAL = 500;
static const float LEARNING_RATE = 5e-4f;
static const int EVAL_ITERS = 25; 
static const int N_EMBD = 128;   
static const int N_HEAD = 2;      
static const int N_LAYER = 4;
static const float DROPOUT = 0.05f;
static const int BPE_VOCAB_SIZE = 2048; 
```

## Character-level implemented in C++

No third-party runtime dependency - it builds from `main.cpp`, `config/config.h`, and `include/*.h` alone.

- Character-level tokenizer built directly from the input corpus
- Train/validation split via `DataLoader`
- Token + positional embeddings
- Multi-head causal self-attention with explicit QKV projections
- Pre-layer-norm residual transformer blocks
- Feed-forward MLP with ReLU
- Cross-entropy loss
- **Fully analytical backward pass** - every gradient (attention, layer norm, MLP, embeddings) is derived and coded in `include/backward.h`, not autograd
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
g++ -std=c++17 -O3 -march=native -fopenmp -I. -Iinclude -o llm.exe main.cpp
```

## the PyTorch reference path

[engine/main.py](engine/main.py) trains the same architectural idea with `torch`, `torch.nn`, and GPT-4 BPE tokenization via `tiktoken`, useful when you want to scale past what C++ loops can comfortably train on CPU.

```bash
cd engine
python fineweb_dataset.py # you can also use data/input.txt also 
python main.py
```

It looks for `engine/input.txt` by default; point it elsewhere with `QUADTRIX_TRAIN_DATA` if needed. Run inference against a saved checkpoint:

```bash
python engine/inference.py --checkpoint engine/best_model.pt --prompt "Once upon a time" --max-new-tokens 100
```

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
| **llm.cpp** | Training *and* inference, manual backward pass | C++ / Python | Manual (C++) + PyTorch |

I'd like the C++ core (`main.cpp`, `include/`, `config/`) to stay dependency-free and to stay the part of this repo that explin transformer internals directly. The PyTorch engine, include, and ci are welcome to grow more features, integrations, and CI polish. If you build a port to another language or framework, I'm happy to link to it from a notable-forks section; just open an issue or PR.

## references

- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762), 2017
- Radford et al., [GPT-2 technical work](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2019
-  [nanoGPT](https://github.com/karpathy/nanoGPT) as educational reference points
-  [HuggingFace](https://huggingface.co/) for fineweb and other datasets.

## Cite

If you find llm.cpp helpful in your research cite as:
```bibtex
@misc{llm.cpp,
  author = {Eamon Sippy},
  title = {llm.cpp: LLM training in C++ \& Python},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/LMGNU/llm.cpp}
}

```


## license

GPL-3.0

