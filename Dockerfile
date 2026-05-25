FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++ \
      python3 \
      python3-pip \
      python3-venv \
      curl \
      ca-certificates \
      && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
      && apt-get install -y --no-install-recommends nodejs \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .
RUN g++ -std=c++17 -O2 -I. -Iinclude -o quadtrix main.cpp
RUN cd frontend \
      && npm ci \
      && npm run build
RUN python3 -m venv /venv \
      && /venv/bin/pip install --upgrade pip --quiet \
      && /venv/bin/pip install -r backend/requirements.txt --quiet

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE:-ubuntu:24.04} AS runtime

LABEL org.opencontainers.image.title="Quadtrix.cpp"
LABEL org.opencontainers.image.description="Local LLM with C++/PyTorch backends and React UI"
LABEL org.opencontainers.image.source="https://github.com/Eamon2009/Quadtrix.cpp"
LABEL org.opencontainers.image.version="1.1.0"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
      PYTHONUNBUFFERED=1 \
      PATH="/venv/bin:$PATH"

# Runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      supervisor \
      curl \
      ca-certificates \
      && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
      && apt-get install -y --no-install-recommends nodejs \
      && npm install -g serve --quiet \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /venv              /venv
COPY --from=builder /build/quadtrix    /app/quadtrix
COPY --from=builder /build/frontend/dist /app/frontend/dist
COPY --from=builder /build/backend     /app/backend
COPY --from=builder /build/engine      /app/engine
COPY supervisord.conf       /etc/supervisor/conf.d/quadtrix.conf
COPY docker-entrypoint.sh   /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh /app/quadtrix \
      && mkdir -p /var/log/supervisor /app/models
VOLUME ["/app/models"]
ENV TORCH_CHECKPOINT_PATH=/app/models/best_model.pt \
      GPT_MODEL_PATH=/app/models/best_model.bin \
      API_PORT=3001 \
      CORS_ORIGINS=http://localhost:8080 \
      LOG_LEVEL=INFO \
      MAX_SESSIONS=1000 \
      SESSION_TTL_HOURS=24
EXPOSE 3001 8080

ENTRYPOINT ["/app/entrypoint.sh"]