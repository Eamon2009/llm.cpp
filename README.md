# llm.cpp

<h1 align="center">
  <img width="1942" height="700" alt="llm-cpp-lmgnu" src="https://github.com/user-attachments/assets/eca0f789-395f-4457-a867-684b428f6963" />
 


[![LMGNU](https://img.shields.io/badge/LMGNU-powered-56D1A0?logo=https%3A%2F%2Fraw.githubusercontent.com%2FLMGNU%2F.github%2Fmain%2Fprofile%2Fdragon_logo.svg)](https://github.com/LMGNU/llm.cpp/actions/workflows/ci-approval.yml) [![Release](https://img.shields.io/github/v/release/LMGNU/llm.cpp)](https://github.com/LMGNU/llm.cpp/releases) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg?logo=gnu)](https://www.gnu.org/licenses/gpl-3.0) 
</h1>


llm.cpp implements language models in dependency-free C++, eliminating the need for PyTorch or Python to train a transformer locally. The core implementation is a decoder-only ***GPT architecture*** featuring custom tensors, embeddings, multi-head causal self-attention, layer normalization, cross-entropy loss, and an analytical backward pass with the AdamW optimizer - all contained within [main.cpp](main.cpp) ,[llm.mm](llm.mm) and the [include/](include) directory also a ***token level [BPE]*** [tokenizer.h](include/tokenizer.h) implementation inside [include](include). With no autograd engine or external frameworks, every gradient is explicitly derived and written out.
The model achieves a validation loss of 1.6371 nats after 76 minutes of CPU training on 31.4 million characters, demonstrating that character-level language modeling at this scale is highly tractable on commodity hardware without external dependencies. On a GPU (CUDA/bfloat16), a validation loss of 2.3918 is reached in under 83 minutes, achieving a peak throughput of 19.6k tokens per second.

## Board
| S.No. | time | val_bpb / Metric | scale | Date | Contributors |
|---|-------------|------------------|-------|------|--------------|
| 1 | 168 hours | 29.41 PPL (~0.93 BPB) | 124M (32x TPU v3) | Feb 2019 | OpenAI (GPT-2 Small) |
| 2 | 45 min | 3.28 Val Loss (~0.748 BPB) | 124M (8x H100) | May 2024 | Andrej Karpathy (llm.c) |
| 3 | 2.98 min | 3.28 Val Loss (~0.748 BPB) | 124M (8x H100) | Feb 2025 | Keller Jordan et al. (Modded-NanoGPT) |
| 4 | 72 hours | 22.7 PPL (~0.85 BPB) | 125M (32x A100) | Feb 2024 | Meta (MobileLLM-125M) |
| 5 | ~24 hours | ~1.02 BPB | 135M (64x H100) | Jul 2024 | Hugging Face (SmolLM-135M) |
| 6 | 61.3 min | 0.7176 | 10.82M (T4) | Mar 2026 | Eamon |
| 7 | 6.1 min | 0.9250 | 1.99M (T4) | Feb 2026 | Eamon |
| 8 | 39.4 min | 1.3145 | 0.82M (CPU) | July 2026 | Eamon |
| 9 | 76.2 min | 1.6371 | 0.82M (CPU) | Jan 2026 | Eamon |

More broadly, the primary contribution of this work lies in its absolute transparency. Every gradient in the backward pass is explicitly written and readable, and every tensor operation is a standard C++ function. By exposing exactly what frameworks like PyTorch compute under the hood, this implementation provides a clear educational pathway. We believe that this fundamental understanding is the true foundation of genuine expertise in deep learning.
The point of this repo is the C++ core. The PyTorch exist to make the model usable, but if you're here to ***train a GPT without a framework*** doing the work for you, [include/backward.h](include/backward.h) is where to start seeing optimization without torch.

---

## From CPU to GPU
The custom C++ backend is transparent but slow: a CPU executes scalar matrix multiplication at roughly 1-10 GFLOP/sec. An NVIDIA RTX 4090 delivers ∼80 TFLOP/s-an 8,000 - 80,000× speedup for the same computation.The LibTorch port replaces the custom backend with PyTorch’s C++ API, gaining cuBLAS-accelerated matrix operations. The transformer architecture remains unchanged only the compute layer is modified. A single line migrates
the model to GPU:
```cpp
model->to(torch::kCUDA)
```
<img width="824" height="218" alt="image" src="https://github.com/user-attachments/assets/073d1c14-f44e-413b-9f0f-ae35be81d1ed" />


which transfers all parameters to GPU memory; all subsequent `torch::matmul` calls dispatch to cuBLAS automatically.
***visit web***: [llm.app](https://lmgnu.github.io/llm.app)

```mermaid
graph TD
    %% Premium Dark Theme Styling (VS Code Inspired)
    classDef default fill:#1e1e1e,stroke:#444,stroke-width:1px,color:#ccc;
    classDef setup fill:#252526,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef model fill:#252526,stroke:#c586c0,stroke-width:2px,color:#fff;
    classDef math fill:#252526,stroke:#ce9178,stroke-width:2px,color:#fff;
    classDef train fill:#2d221e,stroke:#d7ba7d,stroke-width:2px,color:#fff;
    classDef boxGroup fill:#2d2d30,stroke:#555,stroke-width:1px,color:#e0e0e0,rx:8px,ry:8px;
    classDef trainGroup fill:#332a1e,stroke:#cc8800,stroke-width:1.5px,color:#ffd700,rx:8px,ry:8px;

    %% 1. Application Entry
    MAIN("<b><span style='color:#569cd6;font-size:16px'>main.cpp</span></b><br/><span style='font-size:12px;color:#9cdcfe'>App Execution</span>"):::setup
    
    %% 2. Initialization Setup
    subgraph Init [Memory & Token Setup]
        direction LR
        TOKEN("<b><span style='color:#4ec9b0;font-size:14px'>tokenizer.h</span></b><br/><span style='font-size:11px;color:#d4d4d4'>String -> Token IDs</span>"):::setup
        TENSOR("<b><span style='color:#4ec9b0;font-size:14px'>tensor.h</span></b><br/><span style='font-size:11px;color:#d4d4d4'>Math & Memory Base</span>"):::setup
    end
    class Init boxGroup;
    
    MAIN --> TOKEN
    MAIN --> TENSOR
    
    %% 3. Model Orchestration
    MODEL("<b><span style='color:#c586c0;font-size:16px'>gpt.h / lm.h</span></b><br/><span style='font-size:12px;color:#d4d4d4'>Network Architecture Config</span>"):::model
    
    TOKEN --> MODEL
    TENSOR --> MODEL
    
    %% 4. The Pipeline (Forward Pass)
    subgraph Forward [Transformer Forward Pass]
        direction LR
        EMBED("<span style='color:#ce9178;font-weight:bold'>embedding.h</span>"):::math
        BLOCK("<span style='color:#ce9178;font-weight:bold'>block.h</span>"):::math
        NORM("<span style='color:#ce9178;font-weight:bold'>layernorm.h</span>"):::math
        ATTN("<span style='color:#ce9178;font-weight:bold'>attention.h</span>"):::math
        LIN("<span style='color:#ce9178;font-weight:bold'>linear.h</span>"):::math
        FFN("<span style='color:#ce9178;font-weight:bold'>feedforward.h</span>"):::math

        EMBED ==> BLOCK ==> NORM ==> ATTN ==> LIN ==> FFN
    end
    class Forward boxGroup;
    
    MODEL --> EMBED
    
    %% 5. Training & Checkpointing
    subgraph Training [Backpropagation & Checkpointing]
        direction LR
        BACKWARD("<b><span style='color:#d7ba7d;font-size:15px'>backward.h</span></b><br/><span style='font-size:11px;color:#d4d4d4'>Compute Gradients</span>"):::train
        MODEL_BIN("<b><span style='color:#4ec9b0;font-size:15px'>best_model.bin</span></b><br/><span style='font-size:11px;color:#d4d4d4'>Save Trained Checkpoint</span>"):::train
        
        BACKWARD -.->|Save Checkpoint| MODEL_BIN
    end
    class Training trainGroup;
    
    %% Output to Training
    FFN ==>|Loss & Gradients Flow| BACKWARD
```
---

## quick start (CPU)
<h1 align="center">
<img width="730" height="181" alt="image" src="https://github.com/user-attachments/assets/d510c555-780c-4449-b54b-55ec46bb8d23" />
</h1>


The fastest way to see the whole pipeline - tokenize, train, checkpoint, generate - using the bundled character-level corpus:

get the data set first :

``` shell
cd data             # you set the file size for data set
python data_set.py  # also get any dataset from hugging face datasets 
```

```bash
# run this 
g++ -std=c++17 -O3 -march=native -fopenmp -I. -Iinclude -o llm.exe main.cpp
./llm.exe data/input.txt
```
should see something like this 
```logs

[DATA]  Total tokens : 3521179
[DATA]  Train tokens : 3169061
[DATA]  Val tokens   : 352118

██╗     ██╗     ███╗   ███╗        ██████╗ ██████╗ ██████╗
██║     ██║     ████╗ ████║       ██╔════╝ ██╔══██╗██╔══██╗
██║     ██║     ██╔████╔██║ █████╗██║      ██████╔╝██████╔╝
██║     ██║     ██║╚██╔╝██║ ╚════╝██║      ██╔═══╝ ██╔═══╝
███████╗███████╗██║ ╚═╝ ██║ █████╗╚██████╗ ██║     ██║
╚══════╝╚══════╝╚═╝     ╚═╝ ╚════╝ ╚═════╝ ╚═╝     ╚═╝


  +------------------------------------------+------------------------------------------+
  | LLM Architecture                                                                    |
  +------------------------------------------+------------------------------------------+
  | Max Context Length   : 64                | Vocab Size (BPE)     : 2056              |
  | Number of Layers     : 4                 | Attention Heads      : 2                 |
  | Embedding Channels   : 128               | Total Parameters     : 1328392           |
  | Repetition Penalty   : 10                | Repetition Window    : 10                |
  +------------------------------------------+------------------------------------------+

  +-------------------------------------------------------------------------------------+
  | Host Hardware Specs                                                                 |
  +-------------------------------------------------------------------------------------+
  | Host CPU Device      : AMD Ryzen 5 PRO 3500U w/ Radeon...                           |
  | Host RAM (Total)     : 8045 MB                                                      |
  +-------------------------------------------------------------------------------------+

step 1/5000(0.02%) | train loss 7.650238 | val loss 7.652169  | lr 1.00e-06 |  4016.84 ms |  509 tok/s | ram 189.6 MB
step 2/5000(0.04%) | train loss 7.648808 | val loss 7.652169  | lr 2.00e-06 |  4053.90 ms |  505 tok/s | ram 190.7 MB
step 3/5000(0.06%) | train loss 7.658056 | val loss 7.652169  | lr 3.00e-06 |  4381.06 ms |  467 tok/s | ram 190.7 MB
step 4/5000(0.08%) | train loss 7.648185 | val loss 7.652169  | lr 4.00e-06 |  4514.13 ms |  453 tok/s | ram 189.8 MB
step 5/5000(0.10%) | train loss 7.646149 | val loss 7.652169  | lr 5.00e-06 |  4429.38 ms |  462 tok/s | ram 190.3 MB
step 6/5000(0.12%) | train loss 7.644379 | val loss 7.652169  | lr 6.00e-06 |  4443.55 ms |  460 tok/s | ram 190.8 MB
```

This trains from scratch on `data/input.txt` and writes the best checkpoint to `best_model.bin`. Once you have a checkpoint, generate or chat with it:

```bash
./llm.exe data/input.txt --generate
./llm.exe data/input.txt --chat --chat-tokens 300
```

debugging tip: drop `-O2` for `-g` when compiling if you want to step through `include/backward.h` or `include/gpt.h` in a debugger - the manual backward pass is much easier to follow one breakpoint at a time.

### runtime arguments

```bash
llm.exe [data_path] [--generate] [--chat] [--chat-tokens N]
```
---

## File structure

```text

|-─ .ci/                        # CI/CD pipelines and Docker configurations
├── .github/                    # GitHub Actions workflows and issue templates
├── assets/                     # Project images, banners, and hardware diagrams
├── benches/
│   └── bench.cpp               # C++ benchmarking script for performance testing
├── config/
│   └── config.h                # Global configuration parameters
├── data/
│   ├── dataset.py              # Data loading and preprocessing pipeline
│   ├── data_set.py             # Alternative dataset handling logic
│   ├── export.py               # Script to export models or tensors
│   └── input.txt               # Raw text data used for training/testing
├── docs/                       # Additional documentation and generated reports
├── engine/                     # Core backend implementation
│   ├── llm.pt                  # Primary PyTorch model checkpoint
│   ├── mini-quadtrix.pt        # Minimal PyTorch model for testing
│   └── llm.cpp/                # Low-level C++/CUDA/Metal engine
│       ├── CMakeLists.txt      # Engine-specific build configuration
│       ├── llm.cu              # CUDA implementation for Nvidia GPUs
│       ├── make                # Engine Makefile compilation script
│       ├── train.mm            # Objective-C++ Metal script for Apple Silicon training
│       ├── config/
│       │   └── config.h        # Engine-specific configuration header
│       └── include/            # Neural network mathematical headers
│           ├── attention.h     # Self-attention module definitions
│           ├── cuda_kernels.cuh # Custom CUDA kernel definitions
│           ├── layer.cuh       # Layer abstractions for GPU
│           ├── tensor.cuh      # Core tensor math operations
│           └── ...             # (Other low-level neural net headers)
├── include/                    # High-level C++ API headers
│   ├── attention.h             # High-level attention interfaces
│   ├── gpt.h                   # GPT model architecture definitions
│   ├── llm-cpp.hpp             # Main library interface for external use
│   ├── tokenizer.h             # Text tokenization logic
│   └── torch_bridge.h          # Interoperability layer for PyTorch tensors
├── scripts/
│   └── build.sh                # Automation script for building the project
├── train_test/                 # Experimental and testing scripts
│   ├── model.py                # Python model architecture definitions
│   ├── test.c                  # C-based functional testing
│   └── train2.mm               # Experimental Metal training iterations
├── .clang-format               # Code style rules for C/C++ files
├── .clang-tidy                 # Linter configuration for C/C++ static analysis
├── benchmark.cpp               # Entry point for running system benchmarks
├── CMakeLists.txt              # Root CMake build configuration
├── llm.mm                      # Apple Silicon (Metal) main inference entry point
├── main.cpp                    # Main application C++ entry point
├── README.md                   # Main project documentation
├── requirements.txt            # Python dependencies for the project
└── shards.cpp                  # C++ implementation for handling data shards
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
| **llm.cpp** | Training *and* inference, manual backward pass | C++  | C++ |

I'd like the C++ core (`main.cpp`, `include/`, `config/`) to stay dependency-free and to stay the part of this repo that explin transformer internals directly. The include, and ci are welcome to grow more features, integrations, and CI improvement.If you build a port to another language or framework, I'm happy to link to it from a notable-forks section; just open an issue or PR.

## references

- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), 2017
- Radford et al., ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2 technical work), 2019
- Brown et al., ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) (GPT-3 paper), 2020
- Meta AI, ["The Llama 3 Herd of Models"](https://arxiv.org/abs/2407.21783) (Llama 3 paper), 2024
- Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) repository as an educational reference point
- [HuggingFace Datasets](https://huggingface.co/datasets) for FineWeb and other pretraining/fine-tuning datasets
- Karpathy, A.** (2024). [*Let's reproduce GPT-2 (124M)*](https://youtu.be/l8pRSuU81PU)
 **Note: We express our thanks to Andrej Karpathy for his instructional content. Concepts regarding the multi-head attention structure, learning rate schedule, and binary token shard loading were implemented using his walkthrough.**

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

