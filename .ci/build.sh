#!/bin/bash

set -e

OS=$1
COMPILER=$2

echo "Starting build on $OS using $COMPILER..."

if [ "$OS" == "Linux" ]; then
    if [ "$COMPILER" == "gcc" ]; then
        g++ -std=c++17 -O3 -fopenmp -I. -Iinclude -o llm main.cpp
    elif [ "$COMPILER" == "clang" ]; then
        clang++ -std=c++17 -O3 -fopenmp -I. -Iinclude -o llm main.cpp
    fi

elif [ "$OS" == "macOS" ]; then
    if [ -f "llm.mm" ]; then
        clang++ -std=c++17 -O3 -I. -Iinclude main.cpp llm.mm -framework Foundation -framework Metal -framework MetalPerformanceShaders -o llm
    else
        clang++ -std=c++17 -O3 -I. -Iinclude main.cpp -o llm
    fi

elif [ "$OS" == "Windows" ]; then
    if [ "$COMPILER" == "gcc" ]; then
        g++ -std=c++17 -O3 -fopenmp -I. -Iinclude -o llm.exe main.cpp
    elif [ "$COMPILER" == "msvc" ]; then
        cl /std:c++17 /O2 /openmp /I. /Iinclude main.cpp /Fe:llm.exe
    fi
fi

echo "Build completed successfully!"
