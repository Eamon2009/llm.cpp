# llm.cpp

LLM training in C++17 with no frameworks on the CPU path. The core is ~1,000 lines of dependency-free C++: `main.cpp`, `config/config.h`, and `include/*.h`. If you want to understand what `loss.backward()` actually does without PyTorch hiding the details, this is the place. Thia eliminating the need for PyTorch or Python to train a transformer locally. The core implementation is a decoder-only **GPT architecture** featuring custom tensors, embeddings, multi-head causal self-attention, layer normalization, cross-entropy loss, and an analytical backward pass with the AdamW optimizer all contained within [`main.cpp`](main.cpp), [`llm.mm`](llm.mm), and the [`include/`](include) directory. There is also a **token-level BPE** [`tokenizer.h`](include/tokenizer.h) implementation inside [`include`](include). With no autograd engine or external frameworks, every gradient is explicitly derived and written out. This is not a framework. It is a reference implementation. The kind of thing you build once to prove to yourself that you understand every operation from the matrix multiplications up to the cross-entropy loss, and then you keep around because it turns out to be genuinely useful for training small models on your laptop CPU without fighting a Python environment.

There's also a GPU varient via CUDA in `llmcpp/`, and Apple Silicon Metal support via `llm.mm`, but CUDA pull in external dependencies. The zero-dep CPU build is the reference implementation.

## quick start (CPU)
The idea is simple: take a text file, tokenize it, and train a transformer to predict the next token. The code is organized the way you would actually think about the problem. There is a `GPTLanguageModel` class that holds the parameters, a forward pass that computes logits and loss, a backward pass that walks the computation graph in reverse, and an `AdamW` optimizer that updates the weights. Everything is explicit. If you want to know how gradient accumulation works, or how the causal mask is applied, or how the repetition penalty modifies the logits during sampling, you read the code and it is right there.

You can train a model from scratch, which will save the best checkpoint based on validation loss. You can load that checkpoint and generate text indefinitely. Or you can start an interactive chat session, where a system prompt is prepended to every turn and the model streams tokens back to you in real time. The architecture is fully configurable via a single header file embedding dimension, number of layers, attention heads, context length, learning rate schedule, all of it.

The "I don't even want to install CMake" section.

```bash
cd data
python data_set.py
```
```bash
# Compile
g++ -std=c++17 -O3 -march=native -fopenmp -I. -Iinclude -o llm.exe main.cpp
# run
./llm.exe data/input.txt
```
Just a single training loop packed into a binary that runs on Linux, macOS and Windows.

RAM is measured with get_ram_usage_mb(). On Linux it reads VmRSS from /proc/self/status. On macOS it calls task_info for resident_size. On Windows it pulls WorkingSetSize from K32GetProcessMemoryInfo. The number is printed after every training step, right next to the loss and tokens-per-second. So if you are training on a laptop with 16 GB of RAM and the printed value is climbing past 12 GB, you know immediately there is no guessing about framework overhead or memory fragmentation.You can watch the resident set size jump when the model initializes, hold steady through the forward and backward passes, and tick up during the periodic validation run when a second forward graph is alive. If the number is too high, you reduce BATCH_SIZE or N_LAYER in config.h and recompile. The memory footprint is predictable because every byte is accounted for in the code.

You should see something like:

```text
[DATA]  Total tokens : 3521179
[DATA]  Train tokens : 3169061
[DATA]  Val tokens   : 352118
  +------------------------------------------+------------------------------------------+
  | LLM Architecture                                                                    |
  +------------------------------------------+------------------------------------------+
  | Max Context Length   : 24                | Vocab Size (BPE)     : 2056              |
  | Number of Layers     : 6                 | Attention Heads      : 6                 |
  | Embedding Channels   : 128               | Total Parameters     : 1712904           |
  | Repetition Penalty   : 500               | Repetition Window    : 500               |
  +------------------------------------------+------------------------------------------+

  +-------------------------------------------------------------------------------------+
  | Host Hardware Specs                                                                 |
  +-------------------------------------------------------------------------------------+
  | Host CPU Device      : AMD Ryzen 5 PRO 3500U w/ Radeon...                           |
  | Host RAM (Total)     : 6045 MB                                                      |
  +-------------------------------------------------------------------------------------+

step1/20000(0.01%)|trainloss7.647731|val loss7.663259|lr 3.00e-07|960.52 ms 199 tok/s|ram 70.9 MB
step2/20000(0.01%)|trainloss7.637784 |val loss7.663259|lr 6.00e-07|1243.27 ms|154 tok/s|ram 70.9 MB
step3/20000(0.01%)|trainloss7.658248 |val loss7.663259|lr 9.00e-07|994.39 ms|193 tok/s|ram 70.9 MB
step4/20000(0.02%)|trainloss7.643033 |val loss7.663259|lr 1.20e-06|1025.49 ms|187 tok/s|ram 70.9 MB
step5/20000(0.03%)|trainloss7.623671 |val loss7.663259|lr 1.50e-06|1013.43 ms|189 tok/s|ram 71.0 MB
[SAVE]  Weights written to best_model.bin
step 6/20000(0.03%)|train loss 7.665278|valloss7.654684*|lr .80e-06|1042.40 ms|184 tok/s|ram71.0 MB
best
generating:
What sall I hae a sight of the king's crow�

BAPTISTcs
A good poor man and I will b a man
dnw..han�€.
And let him be the king of the world!
why what a heavy day id?.
step7/20000(0.03%)|trainloss7.678715|val loss 7.654684|lr 2.10e-06|1014.71 ms|189 tok/s|ram 71.2 MB
step8/20000(0.04%)|trainloss7.650753|val loss 7.654684|lr 2.40e-06|1025.41 ms|187 tok/s|ram 71.1 MB
step9/20000(0.04%)|trainloss7.658641|val loss 7.654684|lr 2.70e-06|1058.55 ms|181 tok/s|ram 71.1 MB
 
```

This trains from scratch on `data/input.txt` and writes the best checkpoint to `best_model.bin`. Once you have a checkpoint:

```bash
./llm.exe data/input.txt --generate
./llm.exe data/input.txt --chat --chat-tokens 300
```

debugging tip: drop `-O3` for `-g` when compiling if you want to step through `include/backward.h` or `include/gpt.h` in a debugger. The manual backward pass is much easier to follow one breakpoint at a time.

### runtime arguments

```
llm.exe [data_path] [--generate] [--chat] [--chat-tokens N]
```

| Env var | Default | Description |
|---------|---------|-------------|
| `GPT_DATA_PATH` | `data/input.txt` | Override the default training corpus |
| `GPT_MODEL_PATH` | `best_model.bin` | Override the checkpoint path |

---

## Architecture

Decoder-only GPT, pre-layer-norm residual blocks. Everything is compile-time constants in `config/config.h`:

```cpp
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

What's in the box:
- Token + positional embeddings
- Multi-head causal self-attention with explicit Q/K/V projections
- Feed-forward MLP (ReLU)
- LayerNorm
- Cross-entropy loss
- **Fully analytical backward pass** in `include/backward.h`
- AdamW optimizer (first/second moment estimates)
- Checkpoint save/load
- Autoregressive generation and terminal chat mode

### The Backward Pass

No autograd. No `.backward()` magic. Just C++ loops that do exactly what the math says.

`SavedForward` caches every intermediate from the forward pass: pre-softmax attention scores, post-softmax weights, dropout masks, ReLU inputs, layer norm means and inverse standard deviations. The `backward()` function walks the model in reverse:

1. `backward_cross_entropy` -> `dlogits`
2. LM head -> `backward_linear`, `backward_layernorm`
3. For each block (reverse):
   - FFN branch: dropout -> linear -> ReLU -> linear -> layernorm
   - Residual add
   - MHA branch: dropout -> linear -> split heads -> per-head softmax/QKV backprop
   - Residual add
   - LayerNorm backward
4. Embeddings

`Grads` holds accumulators for every parameter (`GradLinear`, `GradEmbedding`, `GradLayerNorm`, etc.). Every gradient is accumulated, not overwritten, so gradient accumulation across mini-batches works.

`AdamWState` tracks `m` and `v` for every parameter. `apply_grads()` does bias-corrected moments, then `param -= lr * m_hat / (sqrt(v_hat) + eps)`. Note: no weight decay in this version - it's vanilla Adam.

---

## Tokenizer

`include/tokenizer.h`. Two modes, selected automatically by `load()`:

| Mode | When to Use | What It Does |
|------|-------------|--------------|
| **TEXT** | Small/medium datasets that fit in RAM | Reads a `.txt`, trains BPE, stores in `std::vector<int>` |
| **SHARDED** | Large datasets (billions of tokens) | Memory-maps binary shards of `uint16_t` token IDs, streams on demand |

### TEXT Mode

BPE from scratch. Every unique character starts as its own token. Then iterative merge operations: find the most common adjacent pair, merge it, repeat. Uses a linked-list structure (`BPEIndex`) to track active tokens and a hash map (`pair_pos`) for fast pair frequency lookup. The merge table is cached to `tokenizer.bin` so you don't retrain every run.

Encoding: `base_encode()` -> `apply_merges()`. Decoding: vocab lookup, concatenate.

### SHARDED Mode

Pre-tokenize into binary shards. Each shard is a flat stream of `uint16_t` token IDs with a small header. `uint16_t` caps vocab at 65,536 but halves disk I/O and memory bandwidth vs `int32`.

`MMapShard` handles platform-specific memory mapping (`mmap` on POSIX, `MapViewOfFile` on Windows). Move-only semantics prevent accidental copies of file descriptors.

`ShardedSplit` builds a prefix-sum index over token counts, so `token_at(global_idx)` is O(1) pointer arithmetic.

`get_batch()` samples random starting positions and extracts `block_size` consecutive tokens. OpenMP parallelizes across the batch dimension - each thread gets its own RNG seed for determinism.

---

## From CPU to GPU

The custom C++ backend is transparent but slow. A CPU does scalar matrix multiplication at roughly 1–10 GFLOP/s. An RTX 4090 does ~80 TFLOP/s. That's an 8,000–80,000× gap.

Because this is a native CUDA port, everything is built for the GPU from the ground up. You don't need to push the model to the graphics card just initialize GPTLanguageModel, and the memory is automatically allocated in VRAM.
```cpp
GPTLanguageModel model(dl.vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, SEED);
```

There's also a Apple Silicon Metal (`llm.mm`) These are experimental and pull in external dependencies. The `g++` build path is the only truly zero-dependency one.

---

## Benchmarks

These are small character-level models on TinyStories unless noted. Don't expect GPT-2 quality - the point is to see the pipeline work end-to-end.

| Params | Layers | Dim | Heads | Ctx | Vocab | Iters | Val Loss | Time | Hardware |
|--------|--------|-----|-------|-----|-------|-------|----------|------|----------|
| 0.83M | 4 | 128 | 4 | 64 | 105 char | 3,000 | 1.6371 | 76m | CPU (AMD Ryzen) |
| 2.00M | 4 | 200 | 4 | 200 | 110 char | 5,000 | 0.9301 | 86m | CPU x64 |
| 19.17M | 4 | 200 | 4 | 200 | ~50K BPE | 5,000 | 2.3934 | 83m | GPU(bfloat16) |

GPU (CUDA , bfloat16): 2.39 val loss in ~83 min, ~19.6k tok/s. Not the zero-dep path.

---

## Layout

```

.
├── .dockerignore           # Docker configuration to exclude files from image builds
├── .gitattributes          # Git configuration for specific file handling
├── .gitignore              # Git configuration to ignore tracked files
├── CMakeLists.txt          # Main CMake build configuration for the project
├── Dockerfile              # Docker container definitions for isolated environments
├── LICENSE                 # Project open-source license
├── llm.mm                  # Objective-C++ implementation (likely for Apple Metal/Mac support)
├── main.cpp                # Main C++ application entry point
├── README.md               # Project documentation
├── config/                 # Global configuration directory
│   └── config.h            # Header defining hyperparameters and global model settings
├── data/                   # Data processing and export scripts
│   ├── data_set.py         # Python script for managing specific dataset formats
│   ├── dataset.py          # Python script for loading and processing training data
│   └── export.py           # Python script to export PyTorch weights to a C++ binary format
├── include/                # Core CPU/C++ header files for model architecture
│   ├── attention.h         # Self-attention mechanism definitions
│   ├── backward.h          # Backward pass (backpropagation) logic
│   ├── block.h             # Transformer block definitions
│   ├── embedding.h         # Token and positional embedding definitions
│   ├── feedforward.h       # Feedforward neural network definitions
│   ├── gpt.h               # Core GPT model structure and assembly
│   ├── layernorm.h         # Layer normalization definitions
│   ├── linear.h            # Linear (dense) layer definitions
│   ├── llm-cpp.hpp         # Primary library interface header
│   ├── lm.h                # Language Modeling head definitions
│   ├── sampler.h           # Token sampling logic (temperature, top-k, top-p)
│   ├── tensor.h            # Custom CPU tensor data structure
│   ├── tokenizer.h         # Text tokenizer interface
│   └── torch_bridge.h      # Utilities to interface and bridge with PyTorch tensors
├── llmcpp/                 # CUDA/GPU-accelerated implementation directory
│   ├── best_model.bin      # Exported binary weights of the trained model
│   ├── CMakeLists.txt      # CMake configuration specifically for the CUDA module
│   ├── llm.cu              # Main CUDA implementation file for the model
│   ├── train_model_fp32.cu # CUDA script for training the model in FP32 precision
│   ├── config/             # CUDA-specific configuration
│   │   └── config.h        # GPU hyperparameters, thread counts, and block size settings
│   └── include/            # C++ and CUDA headers (.cuh) for GPU operations
│       ├── attention.hpp   # GPU self-attention definitions
│       ├── backward.h      # GPU backward pass definitions
│       ├── block.h         # GPU transformer block definitions
│       ├── bpe.h           # Byte-Pair Encoding tokenizer definitions
│       ├── char_level.h    # Character-level tokenizer definitions
│       ├── cuda_kernels.cuh # Custom CUDA kernel declarations
│       ├── cuda_utils.cuh  # Utility functions for CUDA (memory management, error checking)
│       ├── embedding.h     # GPU embedding layer definitions
│       ├── feedforward.h   # GPU feedforward layer definitions
│       ├── flash_atte..cuh # Optimized FlashAttention CUDA kernels
│       ├── layer.cuh       # Base layer class for CUDA modules
│       ├── layernorm.h     # GPU layer normalization definitions
│       ├── linear.h        # GPU linear layer definitions
│       ├── lm.h            # GPU Language Modeling head definitions
│       ├── model.h         # Complete GPU model assembly definitions
│       ├── sampler.h       # GPU-accelerated sampling logic
│       └── tensor.cuh      # GPU tensor data structure definitions
└── scripts/                # Utility shell scripts
    └── build.sh            # Shell script to compile the project via CMake/Make
```

---

## What This Is and Isn't

**This is:** A readable C++ reference for how transformer training works under the hood. If you've read Karpathy's *llm.c* and want the same concepts in C++ with a backward pass, this is it.

**This isn't:** A production training framework. Models are tiny (sub-20M parameters), there's no distributed training, no gradient checkpointing, no model parallelism, no quantization. If you want to train something useful, use llm.c, nanoGPT, or a real framework.

---

## References
- [Vaswani et al., "Attention Is All You Need", 2017](https://arxiv.org/abs/1706.03762)
- [Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Karpathy, A., **llm.c** Repository](https://github.com/karpathy/llm.c) - LLM training in simple, raw C/CUDA.
- [Karpathy, A., ***Let's reproduce GPT-2 (124M)***, 2024](https://youtu.be/l8pRSuU81PU)- concepts regarding multi-head attention structure, learning rate schedule, and binary token shard loading were implemented using his walkthrough (see the [build-nanogpt README.md](https://github.com/karpathy/build-nanogpt/blob/master/README.md)).

## License

GPL-3.0
