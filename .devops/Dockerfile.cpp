
FROM ubuntu:24.04 AS builder

LABEL stage=builder

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_TYPE=Release
ARG CMAKE_EXTRA_FLAGS=""

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

COPY main.cpp        ./
COPY benchmark.cpp   ./
COPY config/         ./config/
COPY include/        ./include/
COPY data/           ./data/

# If model/Cmakelists.txt exists, use cmake; else fall back to direct g++
RUN set -e; \
    if [ -f model/Cmakelists.txt ] || [ -f CMakeLists.txt ]; then \
    cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    ${CMAKE_EXTRA_FLAGS} .; \
    cmake --build build --parallel "$(nproc)"; \
    else \
    g++ -std=c++17 -O3 -march=native \
    -I. -Iinclude \
    -o /usr/local/bin/quadtrix \
    main.cpp; \
    fi
FROM ubuntu:24.04 AS runtime

LABEL org.opencontainers.image.title="Quadtrix.cpp Engine"
LABEL org.opencontainers.image.description="C++ transformer engine for local LM inference"
LABEL org.opencontainers.image.source="https://github.com/Eamon2009/Quadtrix.cpp"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/bin/quadtrix /usr/local/bin/quadtrix
COPY --from=builder /src/data/ ./data/
VOLUME ["/models"]

ENV GPT_DATA_PATH=/app/data/input.txt \
    GPT_MODEL_PATH=/models/best_model.bin

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/quadtrix"]
CMD ["data/input.txt", "--chat"]
