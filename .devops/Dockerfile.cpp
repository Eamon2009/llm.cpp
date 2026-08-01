FROM ubuntu:24.04 AS builder

LABEL stage=builder

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_TYPE=Release
ARG CMAKE_EXTRA_FLAGS=""

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    ninja-build \
    ccache \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Consolidated COPY commands (and removed non-breaking spaces)
COPY main.cpp benchmark.cpp ./
COPY config/ ./config/
COPY include/ ./include/
COPY data/ ./data/

# Build step: If CMakeLists exists, use cmake; else fall back to direct g++
RUN set -e; \
    if [ -f model/CMakeLists.txt ] || [ -f CMakeLists.txt ]; then \
        cmake -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            ${CMAKE_EXTRA_FLAGS} .; \
        cmake --build build --parallel "$(nproc)"; \
        # Ensure the compiled binary is moved to the expected location for the next stage
        cp build/quadtrix /usr/local/bin/quadtrix; \
    else \
        g++ -std=c++17 -O3 -march=native \
            -I. -Iinclude \
            -o /usr/local/bin/quadtrix \
            main.cpp; \
    fi

FROM ubuntu:24.04 AS runtime

# Consolidate labels into a single layer
LABEL org.opencontainers.image.title="llm.cpp Engine" \
      org.opencontainers.image.description="C++ transformer engine for local LM inference" \
      org.opencontainers.image.source="https://github.com/LMGNU/llm.cpp"

# Install runtime dependencies and create a non-root user for security
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -d /app -s /bin/bash appuser

WORKDIR /app

# Copy the binary and data from the builder stage
COPY --from=builder /usr/local/bin/quadtrix /usr/local/bin/quadtrix
COPY --from=builder /src/data/ ./data/

# Give ownership of the working directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

VOLUME ["/models"]

ENV GPT_DATA_PATH=/app/data/input.txt \
    GPT_MODEL_PATH=/models/best_model.bin

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/quadtrix"]
CMD ["data/input.txt", "--chat"]
