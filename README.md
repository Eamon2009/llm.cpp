# Quadtrix.cpp
---
Quadtrix.cpp is a local large language model project built around a modular, multi-path architecture that allows to choose the right execution strategy for their hardware and workflow. Whether you are working on a bare-metal embedded environment, running experiments on a GPU cluster, serving a REST API, or interacting through a browser-based chat interface, Quadtrix.cpp provides a coherent and composable foundation for each of those scenarios. This is designed to be approachable for people who want to read and modify every layer of the stack, while remaining practical enough for people who simply want to spin up a working local model quickly.
> For full technical reference, check the documentation — <a href="https://eamon2009.github.io/LLMs/" style="color:#1a73e8;text-decoration:underline;" target="_blank"> Docs</a>
 


> [!IMPORTANT]
> Please be aware that several commands listed in this documentation—specifically those involving file paths and directory navigation—should not be directly copied and pasted into your terminal. Because file structures and path syntax (such as / vs \) vary significantly across operating systems like Windows, macOS, and Linux, you must manually adjust these arguments to match your local environment. Ensure you verify your current working directory and replace any placeholder paths with the absolute or relative path specific to your machine to avoid execution errors.

---
##  Architecture

<img width="1016" height="684" alt="image" src="https://github.com/user-attachments/assets/0e9faad4-71a9-4c7f-80e9-1136dfea6e57" />
The diagram shows how tokens enter at the bottom as raw IDs, get converted into vector embeddings with positional information added, then pass upward through a repeated stack of decoder blocks - each block applying masked attention followed by a feed-forward layer, with normalisation wrapping both. At the very top, a linear projection maps those representations to output logits across the vocabulary. The right-hand side zooms into the attention mechanism itself, showing how queries, keys, and values are linearly projected, fed into a scaled dot-product with an optional causal mask and softmax, then concatenated across all heads before being projected back out. The training flow panel on the far right shows this running as a five-step cycle per batch: data loading, forward pass, loss computation, backward pass for gradients, and a weight update. The bottom section confirms the behaviour through training loss, validation loss, and perplexity plots - all three curves descending and converging steadily as steps increase, indicating the model is learning as expected.
 

## v1.1.0 
<img width="2185" height="829" alt="run_20260430_192930" src="https://github.com/user-attachments/assets/c6db061a-aa8d-4d8d-a1e2-1a81418bb613" />
<img width="2442" height="1586" alt="run_20260508_110726" src="https://github.com/user-attachments/assets/ef51d1c3-e28e-4674-8a71-5513e753b174" />


---

## Contents

- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run the Project](#run-the-project)
- [C++ Backend](#c-backend)
- [PyTorch Backend](#pytorch-backend)
- [FastAPI Backend](#fastapi-backend)
- [Frontend](#frontend)
- [NPM CLI](#npm-cli)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Training Details](#training-details)
- [Reports and Results](#reports-and-results)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## System Architecture

Quadtrix has four main layers:

```text
User Interface
  React + Vite frontend
  PWA assets and service worker

API Layer
  FastAPI app in backend/
  Session storage, request validation, CORS, health checks

Model Backends
  C++ executable: main.cpp + include/
  PyTorch checkpoint: engine/inference.py + engine/best_model.pt or engine/best_model .pt

Training and Data
  C++ character-level training from data/input.txt
  PyTorch GPT-2 BPE training from engine/input.txt or QUADTRIX_TRAIN_DATA
```

### Backend Modes

| Backend | Path | Purpose |
|---|---|---|
| C++ CPU | `main.cpp`, `include/`, `config/` | Manual transformer training, generation, terminal chat |
| PyTorch | `engine/main.py`, `engine/inference.py` | Faster training/inference with `torch` and `tiktoken` |
| iGPU | `iGPU/` | Experimental integrated-GPU inference/training scripts |
| API | `backend/main.py` | FastAPI middleware used by the frontend |
| Web UI | `frontend/` | Local chat interface and PWA shell |

## Repository Structure

This is the current project layout, excluding generated caches such as `.git/`, `.venv/`, `.npm-cache/`, and build outputs.

```text
Quadtrix.cpp/
|-- .github/
|   |-- ISSUE_TEMPLATE/
|   |-- workflows/
|   |   |-- ci.yml
|   |   `-- release.yml
|   |-- dependabot.yml
|   `-- pull_request_template.md
|-- backend/
|   |-- middleware/
|   |   |-- error_handler.py
|   |   |-- logging.py
|   |   `-- __init__.py
|   |-- router/
|   |   |-- chat.py
|   |   |-- feedback.py
|   |   |-- health.py
|   |   |-- sessions.py
|   |   `-- __init__.py
|   |-- README.md
|   |-- config.py
|   |-- inference.py
|   |-- main.py
|   |-- models.py
|   |-- requirements.txt
|   |-- server.py
|   `-- session_store.py
|-- bin/
|   `-- quadtrix.js
|-- config/
|   `-- config.h
|-- data/
|   |-- data_set.py
|   `-- input.txt
|-- engine/
|   |-- data/
|   |   |-- data_set.py
|   |   `-- input.txt
|   |-- fine-tune/
|   |   |-- chat.py
|   |   |-- data-set.py
|   |   `-- main.py
|   |-- logs/
|   |-- engine.c
|   |-- export_weights.py
|   |-- fineweb_30mb.txt
|   |-- fineweb_dataset.py
|   |-- inference.py
|   `-- main.py
|-- frontend/
|   |-- public/
|   |   |-- icon.svg
|   |   |-- manifest.webmanifest
|   |   `-- sw.js
|   |-- src/
|   |   |-- api/
|   |   |-- components/
|   |   |-- hooks/
|   |   |-- store/
|   |   |-- types/
|   |   |-- utils/
|   |   |-- App.tsx
|   |   |-- index.css
|   |   |-- main.tsx
|   |   `-- registerServiceWorker.ts
|   |-- index.html
|   |-- manifest.webmanifest
|   |-- package.json
|   |-- tailwind.config.ts
|   |-- tsconfig.json
|   |-- vite.config.ts
|   `-- sw.js
|-- gpu/
|   |-- dataloader.h
|   `-- model.h
|-- iGPU/
|   |-- inference.py
|   `-- main.py
|-- include/
|   |-- attention.h
|   |-- backward.h
|   |-- block.h
|   |-- dataloader.h
|   |-- embedding.h
|   |-- feedforward.h
|   |-- gpt.h
|   |-- layernorm.h
|   |-- linear.h
|   |-- tensor.h
|   `-- torch_bridge.h
|-- model/
|   |-- Cmakelists.txt
|   `-- export_tokenizer.py
|-- scripts/
|   `-- build_torch.ps1
|-- src/
|   |-- torch_example.cpp
|   `-- torch_main.cpp
|-- contributing.md
|-- LICENSE
|-- main.cpp
|-- package.json
|-- quadtrix.exe
|-- quadtrix_training_report.png
|-- README.md
|-- run.md
`-- SECURITY.md
```

### Important Files

| File | Role |
|---|---|
| `main.cpp` | C++ entry point for training, generation, and terminal chat |
| `config/config.h` | C++ hyperparameters, data path defaults, checkpoint path defaults |
| `include/tensor.h` | Custom tensor operations used by the C++ model |
| `include/gpt.h` | GPT language model implementation and generation path |
| `include/backward.h` | Analytical backpropagation and AdamW optimizer state |
| `data/input.txt` | Default C++ training corpus |
| `engine/main.py` | PyTorch training script |
| `engine/inference.py` | PyTorch checkpoint loading and generation |
| `backend/main.py` | FastAPI application entry point |
| `backend/inference.py` | Backend adapter for PyTorch and C++ model services |
| `frontend/src/` | React chat application |
| `bin/quadtrix.js` | Node CLI wrapper for setup, chat, and training |

## Requirements

### Core Requirements

| Tool | Version | Used For |
|---|---:|---|
| Python | 3.10+ recommended | Backend, PyTorch training, PyTorch inference |
| Node.js | 18+ | Frontend and CLI |
| npm | bundled with Node.js | Frontend dependencies |
| C++ compiler | C++17 support | Building `main.cpp` |
| Git | any recent version | Cloning and source control |

### Python Dependencies

The backend installs:

```text
fastapi
uvicorn[standard]
pydantic
pydantic-settings
httpx
redis
torch
tiktoken
```

These are declared in `backend/requirements.txt`.

### C++ Dependencies

The native C++ path has no third-party runtime dependency. It builds from:

```text
main.cpp
config/config.h
include/*.h
```

## Setup

The commands below use PowerShell from the repository root:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
```

### 1. Create a Python Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 2. Install Backend and PyTorch Dependencies

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd ..
```

### 3. Install Frontend Dependencies

```powershell
cd frontend
npm.cmd install
cd ..
```

Use `npm.cmd` on Windows PowerShell if direct `npm` execution is blocked by PowerShell execution policy.

### 4. Build the Frontend

```powershell
cd frontend
npm.cmd run build
cd ..
```

### 5. Build the C++ Executable

If `quadtrix.exe` already exists, this step is optional. To rebuild:

```powershell
g++ -std=c++17 -O2 -I. -Iinclude -o quadtrix.exe main.cpp
```

For extra CPU optimization on GCC/Clang:

```powershell
g++ -std=c++17 -O3 -march=native -I. -Iinclude -o quadtrix.exe main.cpp
```

## Run the Project

<img width="1919" height="1011" alt="Quadtrix UI screenshot" src="https://github.com/user-attachments/assets/744123e5-4b20-424e-8528-474da2c8445a" />

### Option A: Run PyTorch Chat in the Web UI

This is the simplest web path when a PyTorch checkpoint is available.

Terminal 1:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3001
```

Terminal 2:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\frontend
npm.cmd run dev
```

Open:

```text
http://localhost:5173
```

Select the `.pt` or PyTorch backend in the UI. The FastAPI app loads the checkpoint through `engine/inference.py`.

### Option B: Run C++ Terminal Chat

The C++ executable supports terminal chat directly:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\quadtrix.exe data\input.txt --chat
```

Set the number of generated tokens per answer:

```powershell
.\quadtrix.exe data\input.txt --chat --chat-tokens 300
```

The C++ chat path requires `best_model.bin` unless `GPT_MODEL_PATH` points to another checkpoint.

### Option C: Run C++ Generation

```powershell
.\quadtrix.exe data\input.txt --generate
```

Generation streams tokens until interrupted with `Ctrl+C`.

### Option D: Train the C++ Model

```powershell
.\quadtrix.exe data\input.txt
```

Training writes the best checkpoint to:

```text
best_model.bin
```

### Option E: Train the PyTorch Model

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\.venv\Scripts\python.exe engine\main.py
```

The PyTorch script looks for `engine/input.txt` by default. If that file is not present, point the script at an existing corpus with `QUADTRIX_TRAIN_DATA`:

```powershell
$env:QUADTRIX_TRAIN_DATA="C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\data\input.txt"
.\.venv\Scripts\python.exe engine\main.py
```

The PyTorch training script saves its checkpoint as:

```text
best_model.pt
```

## C++ Backend

The C++ implementation is the educational core of the project.

### C++ Features

- Character-level tokenizer built from the input corpus.
- Train/validation split through `DataLoader`.
- Decoder-only transformer architecture.
- Token and positional embeddings.
- Multi-head causal self-attention.
- Feed-forward MLP blocks.
- Pre-layer normalization and residual connections.
- Cross-entropy loss.
- Manual analytical backward pass.
- AdamW optimization.
- Checkpoint saving/loading.
- Autoregressive generation.
- Terminal chat mode.

### C++ Runtime Arguments

```text
quadtrix.exe [data_path] [--generate] [--chat] [--chat-tokens N]
```

| Argument | Description |
|---|---|
| `data_path` | Plain-text corpus used to build the tokenizer and train/validation split |
| `--generate` | Load weights and continuously generate text |
| `--chat` | Load weights and start interactive terminal chat |
| `--chat-tokens N` | Set maximum generated tokens per chat response |

### C++ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GPT_DATA_PATH` | `data/input.txt` | Overrides the default C++ data file |
| `GPT_MODEL_PATH` | `best_model.bin` | Overrides the model checkpoint path |

Example:

```powershell
$env:GPT_MODEL_PATH="C:\models\quadtrix-best.bin"
.\quadtrix.exe data\input.txt --chat
```

## PyTorch Backend

The PyTorch path mirrors the transformer idea with `torch`, `torch.nn`, and `tiktoken`.

### Training

```powershell
.\.venv\Scripts\python.exe engine\main.py
```

Important defaults in `engine/main.py`:

| Parameter | Value |
|---|---:|
| `batch_size` | 16 |
| `block_size` | 32 |
| `max_iters` | 10000 |
| `eval_interval` | 10 |
| `learning_rate` | 1e-3 |
| `n_embd` | 64 |
| `n_head` | 4 |
| `n_layer` | 4 |
| `dropout` | 0.1 |
| Tokenizer | GPT-2 BPE through `tiktoken` |

### CLI Inference

Interactive chat:

```powershell
.\.venv\Scripts\python.exe engine\inference.py --checkpoint "engine\best_model.pt"
```

Single prompt:

```powershell
.\.venv\Scripts\python.exe engine\inference.py --checkpoint "engine\best_model.pt" --prompt "Once upon a time" --max-new-tokens 100 --temperature 1.0
```

Available inference flags:

| Flag | Description |
|---|---|
| `--checkpoint` | Path to `.pt` checkpoint |
| `--prompt` | Generate once instead of starting interactive chat |
| `--max-new-tokens` | Maximum generated tokens |
| `--temperature` | Sampling temperature |
| `--top-k` | Optional top-k sampling cutoff |

## FastAPI Backend

The production-style API lives in `backend/main.py`.

Start it with:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3001
```

The FastAPI backend provides:

- `POST /api/chat`
- `GET /api/health`
- `GET /api/stats`
- session creation/listing/deletion
- message persistence through in-memory or Redis-backed session storage
- feedback capture
- CORS configuration for the frontend
- PyTorch checkpoint loading through `backend/inference.py`

### Backend Environment

Create `backend/.env` if you want to override defaults:

```text
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

The checked-in backend default is `../engine/best_model .pt` with a space before `.pt`. If your checkpoint is named `best_model.pt`, set `TORCH_CHECKPOINT_PATH=../engine/best_model.pt` in `backend/.env`.

Note: the FastAPI C++ adapter expects a C++-compatible HTTP service at `CPP_SERVER_URL` with `/health` and `/generate`. The current `main.cpp` source provides terminal training/chat/generation. Use the PyTorch backend for the web UI unless you have a compatible C++ HTTP service running.

## Frontend

The frontend is a React + TypeScript + Vite app.

### Development Server

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\frontend
npm.cmd run dev
```

Open:

```text
http://localhost:5173
```

### Production Build

```powershell
npm.cmd run build
```

### Preview Production Build

```powershell
npm.cmd run preview
```

Open the preview URL, usually:

```text
http://localhost:4173
```

### PWA Files

The frontend includes installable web app files:

```text
frontend/manifest.webmanifest
frontend/sw.js
frontend/public/manifest.webmanifest
frontend/public/sw.js
frontend/src/registerServiceWorker.ts
```

Install from Chrome or Edge after running the production preview. The installed app still needs the FastAPI backend running at the configured API URL.

## NPM CLI

The root package exposes a `quadtrix` CLI through `bin/quadtrix.js`.

```text
quadtrix chat [--api-port 3001] [--web-port 5173] [--no-open]
quadtrix train --backend cpp [--data data/input.txt]
quadtrix train --backend python
quadtrix setup
```

From the repository root:

```powershell
npm.cmd install
npm.cmd run build:frontend
node bin\quadtrix.js setup
```

Run the packaged chat wrapper:

```powershell
node bin\quadtrix.js chat --api-port 3001 --web-port 5173
```

Train through the wrapper:

```powershell
node bin\quadtrix.js train --backend cpp --data data\input.txt
node bin\quadtrix.js train --backend python --data data\input.txt
```

## Configuration

### C++ Configuration

Edit `config/config.h` and rebuild the C++ executable.

```cpp
static const int BATCH_SIZE = 4;
static const int BLOCK_SIZE = 64;
static const int MAX_ITERS = 3000;
static const int EVAL_INTERVAL = 200;
static const float LEARNING_RATE = 3e-4f;
static const int EVAL_ITERS = 10;
static const int N_EMBD = 128;
static const int N_HEAD = 4;
static const int N_LAYER = 4;
static const float DROPOUT = 0.2f;
```

### C++ Model Shape

| Parameter | Meaning |
|---|---|
| `BATCH_SIZE` | Number of sequences per gradient step |
| `BLOCK_SIZE` | Context length in tokens |
| `N_EMBD` | Embedding width |
| `N_HEAD` | Number of attention heads |
| `N_LAYER` | Number of transformer blocks |
| `DROPOUT` | Dropout probability during training |
| `LEARNING_RATE` | AdamW learning rate |
| `MAX_ITERS` | Total training iterations |
| `EVAL_INTERVAL` | Evaluation/checkpoint interval |

### Scaling Guide

| Goal | Change |
|---|---|
| Better local coherence | Increase `BLOCK_SIZE` |
| Higher model capacity | Increase `N_EMBD` and `N_LAYER` |
| Faster CPU runs | Use fewer layers or lower embedding width |
| Faster optimized build | Compile with `-O3 -march=native` |
| More stable loss estimates | Increase `EVAL_ITERS` |

## API Reference

Base URL:

```text
http://localhost:3001
```

### Health

```text
GET /api/health
```

Example:

```powershell
Invoke-RestMethod http://localhost:3001/api/health
```

### Stats

```text
GET /api/stats
```

Example:

```powershell
Invoke-RestMethod http://localhost:3001/api/stats
```

### Chat

```text
POST /api/chat
```

PyTorch backend example:

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

C++ backend example, when a compatible C++ HTTP service is available:

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

### Sessions

```text
GET    /api/sessions
POST   /api/sessions
DELETE /api/sessions/{id}
GET    /api/sessions/{id}/messages
```

### Feedback

```text
POST /api/feedback
```

## Training Details

Quadtrix uses a decoder-only transformer in the GPT family.

```text
Input token IDs [B, T]
  -> token embedding
  -> positional embedding
  -> transformer block x N
       -> layer norm
       -> masked multi-head self-attention
       -> residual add
       -> layer norm
       -> feed-forward MLP
       -> residual add
  -> final layer norm
  -> language-model head
  -> logits [B, T, vocab_size]
  -> cross-entropy loss
```

### C++ Manual Backpropagation

The C++ path implements gradients explicitly through:

- Cross-entropy and softmax.
- Final projection layer.
- Final layer normalization.
- Feed-forward MLP layers.
- ReLU activation.
- Attention projection.
- Scaled dot-product attention.
- Causal mask.
- Query, key, and value projections.
- Token and positional embeddings.

Gradient flow:

```text
dLoss/dLogits
  -> lm_head
  -> final layer norm
  -> each transformer block in reverse
       -> feed-forward residual branch
       -> layer norm 2
       -> MLP fc2, activation, fc1
       -> attention residual branch
       -> layer norm 1
       -> output projection
       -> each attention head
            -> attention weights @ value
            -> softmax
            -> scaled QK^T
            -> query/key/value projections
  -> token embedding gradients
  -> position embedding gradients
```

### Causal Masking

Future tokens are masked before softmax so each position can only attend to current and previous positions. Masked entries receive zero useful probability mass and therefore do not contribute meaningful gradient during the attention backward pass.

### Optimizer

The C++ path uses AdamW-style updates with:

- First moment estimate.
- Second moment estimate.
- Weight decay.
- Learning-rate-controlled parameter updates.

## Reports and Results

### Leaderboard

| s.no | Time | Val BPB / Loss | Core | Description | Date | Contributors |
|---:|---:|---:|---:|---|---|---|
| 0 | 39.4 min | 1.3145 | 0.82M | Quadtrix CPU baseline, small data, fragmented output | 2026 | @Eamon2009 |
| 1 | 61.3 min | 0.7176 | 10.82M | Quadtrix Colab large-scale run, coherent paragraphs, strong convergence | 2026 | @Eamon2009 |
| 2 | 6.1 min | 0.9250 | 1.99M | Quadtrix T4 optimized run, fast training, stable learning, basic coherence | 2026 | @Eamon2009 |
| 3 | 76.2 min | 1.6371 | ~0.82M | Quadtrix.cpp extended CPU training, 3000 iterations | 2026 | @Eamon2009 |

### Hardware Execution Backends

| Device | Execution Path |
|---|---|
| CPU | Native C++ implementation and PyTorch CPU fallback |
| CUDA | PyTorch CUDA acceleration when available |
| iGPU | Experimental scripts for integrated-GPU style execution paths |

### Training Metrics

Final C++ run metrics documented in the project:

| Metric | Value |
|---|---:|
| Validation loss | 1.6371 |
| Parameters | 826,985 |
| Vocabulary | 105 characters |
| Training tokens | ~28.3M |
| Validation tokens | ~3.1M |
| Training time | ~76.2 min |
| Architecture | 4 layers, 4 heads, 128 embedding width |

### Python (PyTorch)

![Training Report](quadtrix_training_report.png)

### C++

<img width="1417" height="501" alt="C++ training screenshot" src="https://github.com/user-attachments/assets/52e8eccf-49d1-47dd-ac44-2f572852230a" />

### Training Comparison

| Metric | Character-Level | Small Scale | Large Scale |
|---|---:|---:|---:|
| Parameters | 0.83M | 2.00M | 19.17M |
| Layers | 4 | 4 | 4 |
| Embedding Dim | 128 | 200 | 200 |
| Attention Heads | 4 | 4 | 4 |
| Context Length | 64 | 200 | 200 |
| Corpus | TinyStories | TinyStories | Children's Stories |
| Vocab Size | 105 char | 110 char | ~50K BPE |
| Total Iterations | 3,000 | 5,000 | 5,000 |
| Hardware | CPU/CUDA | CUDA T4 | Unknown |
| Final Train Loss | 1.5632 | 0.9045 | Unknown |
| Final Val Loss | 1.6371 | 0.9301 | Unknown |
| Generalization Gap | 0.0739 | 0.0256 | Unknown |

### Comparison to Related Projects

| Project | Focus | Language | Autograd |
|---|---|---|---|
| nanoGPT | Minimal GPT training | Python | PyTorch |
| minGPT | Educational GPT | Python | PyTorch |
| llama2.c | Inference-oriented C implementation | C | None |
| Quadtrix.cpp | Training, inference, web UI, multi-backend experiments | C++/Python/TypeScript | Manual C++ + PyTorch |

## Troubleshooting

### PowerShell Blocks `npm`

Use:

```powershell
npm.cmd install
npm.cmd run dev
npm.cmd run build
```

### Python Backend Cannot Import Dependencies

Install dependencies into the repository virtual environment:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### PyTorch Checkpoint Is Missing

Check the configured checkpoint path:

```powershell
Test-Path "C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\engine\best_model.pt"
```

If it is missing, train first:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\.venv\Scripts\python.exe engine\main.py
```

### C++ Chat Cannot Find Weights

The C++ chat and generation modes need `best_model.bin`.

Train first:

```powershell
.\quadtrix.exe data\input.txt
```

Or point to an existing checkpoint:

```powershell
$env:GPT_MODEL_PATH="C:\path\to\best_model.bin"
.\quadtrix.exe data\input.txt --chat
```

### Frontend Cannot Reach Backend

Start the FastAPI backend:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3001
```

Then check:

```powershell
Invoke-RestMethod http://localhost:3001/api/health
```

### Port Already in Use

Check common ports:

```powershell
Get-NetTCPConnection -LocalPort 3001
Get-NetTCPConnection -LocalPort 5173
Get-NetTCPConnection -LocalPort 4173
Get-NetTCPConnection -LocalPort 8080
```

### Rebuild C++ After Changing Hyperparameters

Any change in `config/config.h` requires recompilation:

```powershell
g++ -std=c++17 -O2 -I. -Iinclude -o quadtrix.exe main.cpp
```

## References

- Vaswani et al., "Attention Is All You Need", 2017.
- Radford et al., GPT-2 technical work, 2019.
- nanoGPT and minGPT as educational reference points.

## License

MIT
