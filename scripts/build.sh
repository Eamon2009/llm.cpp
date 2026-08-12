#!/usr/bin/env bash
set -euo pipefail

# llm.cpp build script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[1/3] Preparing dataset..."
cd data
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: Python not found. Install python3 or python."
    exit 1
fi
$PYTHON data_set.py
cd "$SCRIPT_DIR"
if command -v cmake &> /dev/null && [ -f "CMakeLists.txt" ]; then
    echo "[2/3] CMake found. Building with CMake..."
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cmake --build . --config Release -j"$JOBS"
    cd "$SCRIPT_DIR"
    echo "[3/3] Build complete. Binary: build/llm.exe"
    exit 0
fi
echo "[2/3] CMake not found or no CMakeLists.txt. Falling back to direct compiler..."

detect_compiler() {
    if command -v cl &> /dev/null; then
        echo "msvc"
    elif command -v clang++ &> /dev/null; then
        echo "clang"
    elif command -v g++ &> /dev/null; then
        echo "gcc"
    else
        echo "none"
    fi
}

COMPILER=$(detect_compiler)
echo "        Detected compiler: $COMPILER"

case "$COMPILER" in
    msvc)
        echo "[3/3] Building with MSVC (cl)..."
        cl /std:c++17 /O2 /openmp /I. /Iinclude /EHsc /Follm.exe main.cpp
        echo "        Build complete. Binary: llm.exe"
        ;;
    clang)
        echo "[3/3] Building with Clang..."
        clang++ -std=c++17 -O3 -march=native -fopenmp -I. -Iinclude -o llm.exe main.cpp
        echo "        Build complete. Binary: llm.exe"
        ;;
    gcc)
        echo "[3/3] Building with GCC..."
        g++ -std=c++17 -O3 -march=native -fopenmp -I. -Iinclude -o llm.exe main.cpp
        echo "        Build complete. Binary: llm.exe"
        ;;
    none)
        echo "Error: No supported C++ compiler found."
        echo "       Install one of: g++ (GCC), clang++ (Clang), or cl (MSVC)"
        exit 1
        ;;
esac
