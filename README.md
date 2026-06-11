# Quadtrix.cpp

<h1 align="center">
<img width="963" height="258" alt="image" src="https://github.com/user-attachments/assets/b49ea49d-5d62-419b-a5b3-5d5f84153242" />
<img width="957" height="233" alt="image" src="https://github.com/user-attachments/assets/6655701f-f423-49c1-9b54-967522abe2bf" />



</h1>
Quadtrix.cpp is an experimental local LLM project that combines training, inference, and a chat interface in a single repository across three execution paths: a fully from-scratch C++ implementation, a PyTorch backend, and a React frontend.The C++ path implements a decoder-only transformer entirely by hand — no external tensor libraries, no automatic differentiation. It includes character-level tokenization, manual backpropagation, AdamW optimization, checkpointing, and autoregressive generation. This makes it a clear educational reference for how transformers work under the hood. The PyTorch path offers faster training and inference with BPE tokenization via tiktoken, while the React frontend provides a clean ChatGPT-style web UI on top of either backend.You can train small character-level models on CPU or Colab, or move to BPE-based models with the PyTorch backend. All paths are designed to be hackable and runnable on consumer hardware without requiring massive GPU clusters.


Full technical reference: [docs](https://eamon2009.github.io/LLMs/)

## Leaderboard

Runs are ranked by validation loss. Lower is better.

| # | Val Loss | Parameters | Time     | Hardware | Description                              |
|--:|----------|------------|----------|----------|------------------------------------------|
| 1 | **0.7176**   | 10.82M     | 61.3 min | Colab    | Large-scale run, coherent paragraphs, strong convergence |
| 2 | 0.9250   | 1.99M      | 6.1 min  | T4 GPU   | Optimised run, fast training, stable learning            |
| 3 | 1.3145   | 0.82M      | 39.4 min | CPU      | Baseline, small data                                     |
| 4 | 1.6371   | 0.82M      | 76.2 min | CPU      | Extended CPU training, 3000 iterations                   |

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

### Comparison to Related Projects

| Project       | Focus                                      | Language          | Autograd             |
|---------------|--------------------------------------------|-------------------|----------------------|
| nanoGPT       | Minimal GPT training                       | Python            | PyTorch              |
| minGPT        | Educational GPT                            | Python            | PyTorch              |
| llama2.c      | Inference-oriented C                       | C                 | None                 |
| **Quadtrix.cpp** | Training + inference + web UI + multi-backend | C++ / Python / TypeScript | Manual C++ + PyTorch |


# Getting started
**Create a Python virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

** Install backend and PyTorch dependencies**

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd ..
```
**Train**
```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\.venv\Scripts\python.exe engine\main.py
```
**Build the C++ executable**

Skip if `quadtrix.exe` already exists. To rebuild:

```powershell
g++ -std=c++17 -O2 -I. -Iinclude -o quadtrix.exe main.cpp
```

For maximum CPU throughput:

```powershell
g++ -std=c++17 -O3 -march=native -I. -Iinclude -o quadtrix.exe main.cpp
```

---


## Technical Reference
Quadtrix is a decoder-only transformer. The architecture follows the standard GPT design with pre-layer normalization, causal self-attention, and residual connections.
The C++ implementation is fully from scratch:

- Character-level tokenizer
- Manual tensor operations
- Analytical backpropagation
- AdamW optimizer with bias correction
- Checkpoint save/load
- Autoregressive generation

** The PyTorch path uses torch.nn and tiktoken for faster experimentation and GPU acceleration.**

## File Structure 
```
Quadtrix.cpp/
├── main.cpp
├── config/
│   └── config.h
├── include/
│   ├── tensor.h
│   ├── gpt.h
│   └── backward.h
├── data/
│   └── input.txt
├── engine/
│   ├── main.py
│   ├── inference.py
│   └── best_model.pt
├── backend/
│   ├── main.py
│   ├── inference.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   └── ...
└── quadtrix.exe (after build)
```

## References
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
- Radford et al., "Language Models are Unsupervised Multitask Learners", 2019.
- Karpathy, nanoGPT
- Karpathy, minGPT

## License
MIT
