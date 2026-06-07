# Quadtrix.cpp (llm.cpp)

<h1 align="center">
  <img width="785" height="261" alt="image" src="https://github.com/user-attachments/assets/7bd2d8c6-d1e3-4ca0-96c0-0161d3cf235a" />
</h1><br>


  [![Release](https://github.com/Eamon2009/Quadtrix.cpp/actions/workflows/release.yml/badge.svg)](https://github.com/Eamon2009/Quadtrix.cpp/actions/workflows/release.yml)  [![Package](https://github.com/Eamon2009/Quadtrix.cpp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/Eamon2009/Quadtrix.cpp/actions/workflows/docker-publish.yml)
  [![CI](https://github.com/Eamon2009/Quadtrix.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/Eamon2009/Quadtrix.cpp/actions/workflows/ci.yml)


A local large language model with a modular, multi-path execution architecture. Train, run inference, and serve a chat interface — all from a single repository, across bare-metal C++, PyTorch, and a React frontend.

> Full technical reference: [docs](https://eamon2009.github.io/LLMs/)

> **Note** — File paths in this document are written for Windows PowerShell. Adjust separators (`\` vs `/`) and absolute paths to match your operating system before running any command.

---

## Leaderboard

Runs are ranked by validation loss. Lower is better.

| # | Val Loss | Parameters | Time | Hardware | Description |
|--:|--:|--:|--:|---|---|
| 1 | **0.7176** | 10.82M | 61.3 min | Colab | Large-scale run, coherent paragraphs, strong convergence |
| 2 | 0.9250 | 1.99M | 6.1 min | T4 GPU | Optimised run, fast training, stable learning |
| 3 | 1.3145 | 0.82M | 39.4 min | CPU | Baseline, small data |
| 4 | 1.6371 | 0.82M | 76.2 min | CPU | Extended CPU training, 3000 iterations |

All runs: @Eamon2009, 2026.

---

## Benchmarks

### Runs at a Glance

| Metric | Character-Level | Small Scale | Large Scale |
|---|--:|--:|--:|
| Parameters | 0.83M | 2.00M | 19.17M |
| Layers | 4 | 4 | 4 |
| Embedding dim | 128 | 200 | 200 |
| Attention heads | 4 | 4 | 4 |
| Context length | 64 | 200 | 200 |
| Vocab | 105 char | 110 char | ~50K BPE |
| Corpus | TinyStories | TinyStories | Children's Stories |
| Iterations | 3,000 | 5,000 | 5,000 |
| Train loss | 1.5632 | 0.9045 | — |
| Val loss | 1.6371 | 0.9301 | — |
| Gen. gap | 0.0739 | 0.0256 | — |

### Comparison to Related Projects

| Project | Focus | Language | Autograd |
|---|---|---|---|
| nanoGPT | Minimal GPT training | Python | PyTorch |
| minGPT | Educational GPT | Python | PyTorch |
| llama2.c | Inference-oriented C | C | None |
| **Quadtrix.cpp** | Training + inference + web UI + multi-backend | C++ / Python / TypeScript | Manual C++ + PyTorch |

---

## Table of Contents

- [System Overview](#system-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [C++ Backend](#c-backend)
- [PyTorch Backend](#pytorch-backend)
- [FastAPI Backend](#fastapi-backend)
- [Frontend](#frontend)
- [NPM CLI](#npm-cli)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Technical Reference](#technical-reference)
- [References](#references)
- [License](#license)

---

## System Overview

```
User Interface
    React + Vite frontend
    PWA assets and service worker

API Layer
    FastAPI app in backend/
    Session storage, request validation, CORS, health checks

Model Backends
    C++ executable       main.cpp + include/
    PyTorch checkpoint   engine/inference.py + engine/best_model.pt

Training and Data
    C++ character-level training from data/input.txt
    PyTorch GPT-2 BPE training from engine/input.txt or QUADTRIX_TRAIN_DATA
```

### Backend Modes

| Backend | Path | Purpose |
|---|---|---|
| C++ CPU | `main.cpp`, `include/`, `config/` | Manual transformer training, generation, terminal chat |
| PyTorch | `engine/main.py`, `engine/inference.py` | Faster training and inference with `torch` and `tiktoken` |
| iGPU | `iGPU/` | Experimental integrated-GPU inference and training |
| API | `backend/main.py` | FastAPI middleware used by the frontend |
| Web UI | `frontend/` | Local chat interface and PWA shell |

---

### Key Files

| File | Role |
|---|---|
| `main.cpp` | C++ entry point for training, generation, and terminal chat |
| `config/config.h` | C++ hyperparameters, data path defaults, checkpoint path defaults |
| `include/tensor.h` | Custom tensor operations used throughout the C++ model |
| `include/gpt.h` | GPT language model implementation and generation path |
| `include/backward.h` | Analytical backpropagation and AdamW optimizer state |
| `data/input.txt` | Default C++ training corpus |
| `engine/main.py` | PyTorch training script |
| `engine/inference.py` | PyTorch checkpoint loading and generation |
| `backend/main.py` | FastAPI application entry point |
| `backend/inference.py` | Backend adapter for PyTorch and C++ model services |
| `frontend/src/` | React chat application source |

---

## Requirements

### Core

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Backend, PyTorch training and inference |
| Node.js | 18+ | Frontend and CLI |
| npm | bundled with Node.js | Frontend dependencies |
| C++ compiler | C++17 | Building `main.cpp` |
| Git | any recent | Cloning and source control |

### Python Dependencies

Declared in `backend/requirements.txt`:

```
fastapi
uvicorn[standard]
pydantic
pydantic-settings
httpx
redis
torch
tiktoken
```

### C++ Dependencies

The native C++ path has no third-party runtime dependency. It builds entirely from:

```
main.cpp
config/config.h
include/*.h
```

---

## Setup

All commands run from the repository root in PowerShell.

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
```

**1. Create a Python virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**2. Install backend and PyTorch dependencies**

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd ..
```

**3. Install frontend dependencies**

```powershell
cd frontend
npm.cmd install
cd ..
```

Use `npm.cmd` on Windows PowerShell if `npm` is blocked by execution policy.

**4. Build the frontend**

```powershell
cd frontend
npm.cmd run build
cd ..
```

**5. Build the C++ executable**

Skip if `quadtrix.exe` already exists. To rebuild:

```powershell
g++ -std=c++17 -O2 -I. -Iinclude -o quadtrix.exe main.cpp
```

For maximum CPU throughput:

```powershell
g++ -std=c++17 -O3 -march=native -I. -Iinclude -o quadtrix.exe main.cpp
```

---

## Running the Project

### Option A — PyTorch chat in the web UI

Terminal 1 (backend):

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3001
```

Terminal 2 (frontend):

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\frontend
npm.cmd run dev
```

Open `http://localhost:5173` and select the PyTorch backend in the UI.

### Option B — C++ terminal chat

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\quadtrix.exe data\input.txt --chat
```

Set tokens per response:

```powershell
.\quadtrix.exe data\input.txt --chat --chat-tokens 300
```

Requires `best_model.bin` unless `GPT_MODEL_PATH` points elsewhere.

### Option C — C++ generation

```powershell
.\quadtrix.exe data\input.txt --generate
```

Streams tokens until interrupted with `Ctrl+C`.

### Option D — C++ training

```powershell
.\quadtrix.exe data\input.txt
```

Writes the best checkpoint to `best_model.bin`.

### Option E — PyTorch training

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\.venv\Scripts\python.exe engine\main.py
```

Point to a custom corpus with `QUADTRIX_TRAIN_DATA`:

```powershell
$env:QUADTRIX_TRAIN_DATA="C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\data\input.txt"
.\.venv\Scripts\python.exe engine\main.py
```

Checkpoint is saved to `best_model.pt`.

---

## C++ Backend

### Runtime Arguments

```
quadtrix.exe [data_path] [--generate] [--chat] [--chat-tokens N]
```

| Argument | Description |
|---|---|
| `data_path` | Plain-text corpus used to build the tokenizer and train/validation split |
| `--generate` | Load weights and stream generated text |
| `--chat` | Load weights and start interactive terminal chat |
| `--chat-tokens N` | Maximum generated tokens per chat response |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GPT_DATA_PATH` | `data/input.txt` | Overrides the default data file |
| `GPT_MODEL_PATH` | `best_model.bin` | Overrides the checkpoint path |

```powershell
$env:GPT_MODEL_PATH="C:\models\quadtrix-best.bin"
.\quadtrix.exe data\input.txt --chat
```

---

## PyTorch Backend

### Training

```powershell
.\.venv\Scripts\python.exe engine\main.py
```

Default configuration in `engine/main.py`:

| Parameter | Value |
|---|---|
| `batch_size` | 16 |
| `block_size` | 32 |
| `max_iters` | 10000 |
| `eval_interval` | 10 |
| `learning_rate` | 1e-3 |
| `n_embd` | 64 |
| `n_head` | 4 |
| `n_layer` | 4 |
| `dropout` | 0.1 |
| Tokenizer | GPT-2 BPE via `tiktoken` |

### Inference

Interactive chat:

```powershell
.\.venv\Scripts\python.exe engine\inference.py --checkpoint "engine\best_model.pt"
```

Single prompt:

```powershell
.\.venv\Scripts\python.exe engine\inference.py \
  --checkpoint "engine\best_model.pt" \
  --prompt "Once upon a time" \
  --max-new-tokens 100 \
  --temperature 1.0
```

| Flag | Description |
|---|---|
| `--checkpoint` | Path to `.pt` checkpoint |
| `--prompt` | Generate from a fixed string instead of starting interactive chat |
| `--max-new-tokens` | Maximum tokens to generate |
| `--temperature` | Sampling temperature |
| `--top-k` | Optional top-k sampling cutoff |

---

## FastAPI Backend

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3001
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Generate a model response |
| `GET` | `/api/health` | Service health check |
| `GET` | `/api/stats` | Runtime statistics |
| `GET` | `/api/sessions` | List all sessions |
| `POST` | `/api/sessions` | Create a session |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `GET` | `/api/sessions/{id}/messages` | Retrieve session message history |
| `POST` | `/api/feedback` | Submit feedback |

### Environment

Create `backend/.env` to override defaults:

```ini
API_PORT=3001
CORS_ORIGINS=http://localhost:5173
REDIS_URL=
LOG_LEVEL=INFO
MAX_SESSIONS=1000
SESSION_TTL_HOURS=24
CPP_SERVER_URL=http://localhost:8080
TORCH_CHECKPOINT_PATH=../engine/best_model.pt
REQUEST_TIMEOUT_SECONDS=60
```

> The checked-in default uses `../engine/best_model .pt` with a space before `.pt`. If your checkpoint is named `best_model.pt`, set `TORCH_CHECKPOINT_PATH=../engine/best_model.pt` explicitly.

> The C++ adapter expects a compatible HTTP service at `CPP_SERVER_URL` with `/health` and `/generate` endpoints. The current `main.cpp` operates in terminal mode only. Use the PyTorch backend for the web UI unless a separate C++ HTTP service is running.

---

## Frontend

### Development

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\frontend
npm.cmd run dev
```

Opens at `http://localhost:5173`.

### Production Build

```powershell
npm.cmd run build
```

### Production Preview

```powershell
npm.cmd run preview
```

Served at `http://localhost:4173`.

### PWA

The frontend ships as an installable web app. Install from Chrome or Edge after running the production preview. The installed app requires the FastAPI backend at the configured API URL.

```
frontend/manifest.webmanifest
frontend/sw.js
frontend/public/manifest.webmanifest
frontend/public/sw.js
frontend/src/registerServiceWorker.ts
```

---

## Configuration

### C++ Hyperparameters

Edit `config/config.h` and recompile to apply any change.

```cpp
static const int   BATCH_SIZE     = 4;
static const int   BLOCK_SIZE     = 64;
static const int   MAX_ITERS      = 3000;
static const int   EVAL_INTERVAL  = 200;
static const float LEARNING_RATE  = 3e-4f;
static const int   EVAL_ITERS     = 10;
static const int   N_EMBD         = 128;
static const int   N_HEAD         = 4;
static const int   N_LAYER        = 4;
static const float DROPOUT        = 0.2f;
```

| Parameter | Meaning |
|---|---|
| `BATCH_SIZE` | Sequences per gradient step |
| `BLOCK_SIZE` | Context length in tokens |
| `N_EMBD` | Embedding width |
| `N_HEAD` | Number of attention heads |
| `N_LAYER` | Number of transformer blocks |
| `DROPOUT` | Dropout probability during training |
| `LEARNING_RATE` | AdamW learning rate |
| `MAX_ITERS` | Total training iterations |
| `EVAL_INTERVAL` | Evaluation and checkpoint interval |

### Scaling Guide

| Goal | Change |
|---|---|
| Better local coherence | Increase `BLOCK_SIZE` |
| Higher model capacity | Increase `N_EMBD` and `N_LAYER` |
| Faster CPU runs | Reduce `N_LAYER` or `N_EMBD` |
| Optimised build | Compile with `-O3 -march=native` |
| More stable loss estimates | Increase `EVAL_ITERS` |

---

## API Reference

Base URL: `http://localhost:3001`

### Health

```
GET /api/health
```

```powershell
Invoke-RestMethod http://localhost:3001/api/health
```

### Stats

```
GET /api/stats
```

```powershell
Invoke-RestMethod http://localhost:3001/api/stats
```

### Chat

```
POST /api/chat
```

PyTorch backend:

```powershell
Invoke-RestMethod `
  -Uri http://localhost:3001/api/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "session_id": null,
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 1.0,
    "stream": false,
    "model_backend": "torch"
  }'
```

C++ backend (requires a compatible C++ HTTP service):

```powershell
Invoke-RestMethod `
  -Uri http://localhost:3001/api/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "session_id": null,
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 1.0,
    "stream": false,
    "model_backend": "cpp"
  }'
```

---

## Troubleshooting

**PowerShell blocks `npm`**

```powershell
npm.cmd install
npm.cmd run dev
npm.cmd run build
```

**Backend cannot import dependencies**

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**PyTorch checkpoint is missing**

```powershell
Test-Path "C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\engine\best_model.pt"
```

If missing, train first:

```powershell
.\.venv\Scripts\python.exe engine\main.py
```

**C++ chat cannot find weights**

Train first to produce `best_model.bin`:

```powershell
.\quadtrix.exe data\input.txt
```

Or point to an existing checkpoint:

```powershell
$env:GPT_MODEL_PATH="C:\path\to\best_model.bin"
.\quadtrix.exe data\input.txt --chat
```

**Frontend cannot reach the backend**

Start the backend and verify the health endpoint:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3001
Invoke-RestMethod http://localhost:3001/api/health
```

**Port already in use**

```powershell
Get-NetTCPConnection -LocalPort 3001
Get-NetTCPConnection -LocalPort 5173
Get-NetTCPConnection -LocalPort 4173
Get-NetTCPConnection -LocalPort 8080
```

**Rebuild C++ after changing hyperparameters**

Any change to `config/config.h` requires recompilation:

```powershell
g++ -std=c++17 -O2 -I. -Iinclude -o quadtrix.exe main.cpp
```

---

## Technical Reference

This section documents the model architecture, training procedure, and internal design decisions. It is not required reading to run the project.

### Model Architecture

Quadtrix is a decoder-only transformer in the GPT family. The forward pass is:

```
Input token IDs  [B, T]
  -> token embedding          lookup table: vocab_size x n_embd
  -> positional embedding     learned table: block_size x n_embd
  -> sum
  -> transformer block x N
       -> layer norm (pre-norm)
       -> masked multi-head self-attention
       -> residual add
       -> layer norm (pre-norm)
       -> feed-forward MLP  (linear -> ReLU -> linear)
       -> residual add
  -> final layer norm
  -> language model head      linear: n_embd -> vocab_size (no bias)
  -> logits  [B, T, vocab_size]
  -> cross-entropy loss
```

Each attention head operates over a `n_embd / n_head` dimensional subspace. The causal mask ensures each position can only attend to itself and positions that precede it. Masked logits are set to negative infinity before softmax, collapsing their probability mass to zero.

### C++ Implementation

The C++ path is the educational core of the project. Every computation is written explicitly — no automatic differentiation, no external tensor library.

**What it implements:**

- Character-level tokenizer built from the input corpus
- Train/validation split via `DataLoader`
- Token and positional embedding tables
- Multi-head causal self-attention
- Feed-forward MLP blocks
- Pre-layer normalisation and residual connections
- Cross-entropy loss
- Manual analytical backward pass
- AdamW optimiser with first and second moment estimates
- Checkpoint saving and loading
- Autoregressive generation
- Terminal chat mode

### Manual Backpropagation

Gradients flow explicitly through every layer in reverse order:

```
dLoss/dLogits
  -> lm_head                  linear projection backward
  -> final layer norm
  -> each transformer block in reverse
       -> feed-forward residual branch
       -> layer norm 2
       -> MLP fc2 -> ReLU -> fc1
       -> attention residual branch
       -> layer norm 1
       -> output projection
       -> each attention head
            -> attention weights @ value
            -> softmax
            -> scaled dot product  QK^T / sqrt(d_k)
            -> causal mask
            -> query / key / value projections
  -> token embedding gradients
  -> position embedding gradients
```

Causal mask entries are excluded from the backward pass — they received no probability mass in the forward pass and therefore accumulate no gradient.

### Optimiser

The C++ path implements AdamW:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          first moment
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2         second moment
m_hat = m_t / (1 - beta1^t)                          bias correction
v_hat = v_t / (1 - beta2^t)
theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})
```

Weight decay is applied directly to the parameter rather than to the gradient, which is the key distinction between AdamW and Adam with L2 regularisation.

### PyTorch Backend

The PyTorch path mirrors the same transformer design using `torch.nn` modules and the GPT-2 BPE tokenizer from `tiktoken`. It targets faster experimentation and GPU acceleration. Training, inference, and checkpoint I/O are handled in `engine/main.py` and `engine/inference.py`.

### Hardware Execution Paths

| Device | Path |
|---|---|
| CPU | Native C++ and PyTorch CPU fallback |
| CUDA | PyTorch CUDA acceleration when a compatible GPU is present |
| iGPU | Experimental scripts in `iGPU/` for integrated-GPU execution |

### Final C++ Run Metrics

| Metric | Value |
|---|---|
| Validation loss | 1.6371 |
| Parameters | 826,985 |
| Vocabulary | 105 characters |
| Training tokens | ~28.3M |
| Validation tokens | ~3.1M |
| Training time | ~76.2 min |
| Architecture | 4 layers, 4 heads, 128 embedding dim |

---

## References

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
- Radford et al., "Language Models are Unsupervised Multitask Learners", 2019.
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT).
- Karpathy, [minGPT](https://github.com/karpathy/minGPT).

---

## License

MIT
