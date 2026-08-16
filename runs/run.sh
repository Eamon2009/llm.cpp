#!/bin/bash
set -euo pipefail

CUDA_ROOT="${CUDA_PATH:-/usr/local/cuda}"
export LIBRARY_PATH="$CUDA_ROOT/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"

[ -f data/input.txt ] || python data/data_set.py

if [ -f llmcpp/llm.cu ]; then
    nvcc -O3 -std=c++17 -DENABLE_CUDA -I. -Illmcpp/include -lcublas -o llm llmcpp/llm.cu
elif [ -f llm.cu ]; then
    nvcc -O3 -std=c++17 -DENABLE_CUDA -I. -Iinclude -lcublas -o llm llm.cu
else
    g++ -std=c++17 -O3 -DNDEBUG -DENABLE_CUDA -I. -Illmcpp/include -L"$CUDA_ROOT/lib64" -o llm main.cpp -lcudart -lcublas
fi

./llm data/input.txt "$@"