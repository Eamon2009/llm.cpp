# llm.cpp - CPU-only, zero-dependency build as it should be 
# Build: docker build -t llm.cpp-cpu .
# Run:   docker run --rm -it -v $(pwd)/data:/app/data llm.cpp-cpu

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends     g++     make     python3     python3-pip     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what the CPU build needs
COPY main.cpp ./
COPY config/ ./config/
COPY include/ ./include/
COPY data/ ./data/
COPY scripts/build.sh ./scripts/

# Build the CPU binary
RUN g++ -std=c++17 -O3 -march=native -fopenmp -I. -Iinclude -o llm.exe main.cpp

# Default: train on bundled data
ENTRYPOINT ["./llm.exe"]
CMD ["data/input.txt"]
