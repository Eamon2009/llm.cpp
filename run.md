# llm.cpp

llm.cpp is a local GPT-style language model project with multiple runtime paths:

- Native C++ inference and training through `llm.exe` / `main.cpp`
- PyTorch checkpoint inference through `engine/inference.py` and `engine/best_model .pt`


## Requirements

- Python 3.10+
- C++17 compiler if you want to rebuild the C++ executable

## 1. Python Setup

From the repo root:

```powershell
cd C:\Users\Admin\llm.cpp  these are just examples 
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install backend and PyTorch inference dependencies:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 8. PyTorch CLI Inference

Interactive chat:

```powershell
cd C:\Users\Admin\llm.cpp
.\.venv\Scripts\python.exe engine\inference.py --checkpoint "engine\best_model .pt"
```

Generate once:

```powershell
.\.venv\Scripts\python.exe engine\inference.py --checkpoint "engine\best_model .pt" --prompt "Hello" --max-new-tokens 100 --temperature 1.0
```

## 9. PyTorch Training

Main training:

```powershell
cd C:\Users\Admin\llm.cpp
.\.venv\Scripts\python.exe engine\main.py
```

## 10. C++ Build and Run

Build manually:

```powershell
g++ -std=c++17 -O2 -I. -Iinclude -o llm.exe main.cpp
```

Train from scratch:

```powershell
.\llm.exe data\input.txt
```

Terminal chat:

```powershell
.\llm.exe data\input.txt --chat
```

Raw generation:

```powershell
.\llm.exe data\input.txt --generate
```

## License

GPL 3.0 
