# llm.cpp [Guide]

Commands to build and run llm.cpp.  
All commands assume your **working directory is the repo root** unless noted otherwise.

---

## 1. Prepare data

```bash
# From repo root
python data/data_set.py
```

This creates `data/input.txt`.  
To use a different corpus, set the environment variable:

```bash
export GPT_DATA_PATH=path/to/your/corpus.txt
```

---

## 2. Build

### Linux - CPU (x64 or arm64)

```bash
g++ -std=c++17 -O3 -march=native -fopenmp -I. -Iinclude -o llm main.cpp
```

> **Ubuntu 24.04 arm64 only:** use `g++-14` instead of `g++`.

### Linux - Vulkan

```bash
# Install first: sudo apt-get install -y libvulkan-dev vulkan-tools
g++ -std=c++17 -O3 -DNDEBUG -DENABLE_VULKAN -I. -Iinclude -o llm main.cpp -lvulkan
```

### Linux - CUDA 12 / 13

The CUDA source is in `llmcpp/`. Build from the repo root using the correct path:

```bash
export LIBRARY_PATH="$CUDA_PATH/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"

# Option A: compile llm.cu directly
nvcc -O3 -std=c++17 -DENABLE_CUDA -I. -Illmcpp/include -lcublas -o llm llmcpp/llm.cu

# Option B: compile main.cpp with CUDA linking (if llm.cu is absent)
g++ -std=c++17 -O3 -DNDEBUG -DENABLE_CUDA -I. -Illmcpp/include -L"$CUDA_PATH/lib64" -o llm main.cpp -lcudart -lcublas
```

### Linux - OpenCL

```bash
# Install first: sudo apt-get install -y ocl-icd-opencl-dev opencl-headers
g++ -std=c++17 -O3 -DNDEBUG -DENABLE_OPENCL -I. -Iinclude -o llm main.cpp -lOpenCL
```

### Linux - SYCL (Intel)

```bash
# Install the Intel oneAPI DPC++ compiler first
clang++ -std=c++17 -O3 -DNDEBUG -DENABLE_SYCL -fsycl -fsycl-targets=spir64 -I. -Iinclude -o llm main.cpp
```

### macOS - Apple Silicon (CPU)

```bash
clang++ -std=c++17 -O3 -DNDEBUG -arch arm64 -I. -Iinclude -o llm main.cpp
```

### macOS - Apple Silicon (Metal)

```bash
if [ -f "llm.mm" ]; then
  clang++ -std=c++17 -O3 -DNDEBUG -arch arm64 -DENABLE_METAL -I. -Iinclude \
    llm.mm -framework Foundation -framework Metal -framework MetalPerformanceShaders -o llm
else
  clang++ -std=c++17 -O3 -DNDEBUG -arch arm64 -DENABLE_METAL -I. -Iinclude -o llm main.cpp
fi
```

### Windows - x64 (CPU)

Open **"x64 Native Tools Command Prompt for VS 2022"** and paste:

```cmd
cl /nologo /std:c++17 /O2 /DNDEBUG /EHsc /Iinclude /I. main.cpp /Fe:llm.exe
```

### Windows - arm64 (CPU)

Open **"x64_arm64 Native Tools Command Prompt for VS 2022"** and paste:

```cmd
cl /nologo /std:c++17 /O2 /DNDEBUG /EHsc /Iinclude /I. main.cpp /Fe:llm.exe
```

### Windows  Vulkan

Open **"x64 Native Tools Command Prompt for VS 2022"** and paste:

```cmd
cl /nologo /std:c++17 /O2 /DNDEBUG /DENABLE_VULKAN /EHsc /Iinclude /I. /I"%VULKAN_SDK%\Include" main.cpp /Fe:llm.exe /link "%VULKAN_SDK%\Lib\vulkan-1.lib"
```

### Windows - CUDA 12 / 13

Open **"x64 Native Tools Command Prompt for VS 2022"** and paste:

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64

if exist "llmcpp\llm.cu" (
  nvcc -O3 -std=c++17 -DENABLE_CUDA -I. -Illmcpp\include -o llm.exe llmcpp\llm.cu
) else (
  cl /nologo /std:c++17 /O2 /DNDEBUG /DENABLE_CUDA /EHsc /Iinclude /I. /I"%CUDA_PATH%\include" main.cpp /Fe:llm.exe /link "%CUDA_PATH%\lib\x64\cudart.lib"
)
```

### Windows - OpenCL

Open **"x64 Native Tools Command Prompt for VS 2022"** and paste:

```cmd
cl /nologo /std:c++17 /O2 /DNDEBUG /DENABLE_OPENCL /EHsc /Iinclude /I. /I"%OPENCL_INCLUDE%" main.cpp /Fe:llm.exe /link "%OPENCL_LIB%\OpenCL.lib"
```

### WASM - CPU

```bash
# Requires Emscripten (emsdk activated)
emcc -O3 -std=c++17 -DNDEBUG -s WASM=1 -s EXPORT_ALL=1 -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 -s EXPORT_NAME="LLMModule" -I. -Iinclude main.cpp -o llm.js
```

### WASM - SIMD

```bash
# Requires Emscripten (emsdk activated)
emcc -O3 -std=c++17 -DNDEBUG -msimd128 -msse -msse2 -s WASM=1 -s EXPORT_ALL=1 \
  -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s EXPORT_NAME="LLMModule" -I. -Iinclude main.cpp -o llm.js
```

### iOS - XCFramework

```bash
# Build device slice
SDK=$(xcrun --sdk iphoneos --show-sdk-path)
clang++ -std=c++17 -O3 -DNDEBUG -arch arm64 -isysroot "$SDK" -I. -Iinclude -c main.cpp -o llm-ios-arm64.o
ar rcs libllm-ios-arm64.a llm-ios-arm64.o

# Build simulator slice
SDK=$(xcrun --sdk iphonesimulator --show-sdk-path)
clang++ -std=c++17 -O3 -DNDEBUG -arch x86_64 -isysroot "$SDK" -I. -Iinclude -c main.cpp -o llm-ios-sim-x86_64.o
ar rcs libllm-ios-sim-x86_64.a llm-ios-sim-x86_64.o

# Bundle
xcodebuild -create-xcframework -library libllm-ios-arm64.a -library libllm-ios-sim-x86_64.a -output LLM.xcframework
```

### Debug build (step through backward pass)

Replace `-O3` with `-g` and drop `-DNDEBUG`:

```bash
g++ -std=c++17 -g -fopenmp -I. -Iinclude -o llm main.cpp
```

---

## 3. Run

### Train from scratch

```bash
./llm data/input.txt
```

The best checkpoint is saved to `best_model.bin` automatically.

### Generate text

```bash
./llm data/input.txt --generate
```

### Interactive chat

```bash
./llm data/input.txt --chat --chat-tokens 300
```

### Override paths

```bash
export GPT_DATA_PATH=my_corpus.txt
export GPT_MODEL_PATH=my_checkpoint.bin
./llm
```

---

## 4. Tune hyper-parameters

Edit **`config/config.h`** then recompile:

| Constant | Default | What it controls |
|----------|---------|----------------|
| `BATCH_SIZE` | `32` | Mini-batch size |
| `BLOCK_SIZE` | `64` | Max context length |
| `N_EMBD` | `128` | Embedding dimension |
| `N_HEAD` | `2` | Attention heads |
| `N_LAYER` | `4` | Transformer layers |
| `MAX_ITERS` | `5000` | Training steps |
| `LEARNING_RATE` | `5e-4f` | Peak LR |
| `DROPOUT` | `0.05f` | Dropout rate |
| `BPE_VOCAB_SIZE` | `2048` | Token vocabulary size |

**Out of RAM?** Lower `BATCH_SIZE`, `BLOCK_SIZE`, `N_LAYER`, or `N_EMBD`, then rebuild.

---

## 5. Command-line quick reference

```
llm [data_path] [--generate] [--chat] [--chat-tokens N]
```

| Flag | Effect |
|------|--------|
| *(none)* | Train |
| `--generate` | Generate text from `best_model.bin` |
| `--chat` | Interactive chat |
| `--chat-tokens N` | Max response length |

---

*No PyTorch required. CMake required(optional). Just a compiler.*
