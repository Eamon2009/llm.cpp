# =============================================================================
# Quadtrix.cpp — Makefile  (llama.cpp-style convenience targets)
# =============================================================================

.PHONY: all build clean run dev gpu train bench logs ps shell help

SHELL := /bin/bash
SCRIPT := ./scripts/build.sh

# ── Native C++ ───────────────────────────────────────────────────────────────
CC     := g++
CFLAGS := -std=c++17 -O3 -march=native
IFLAGS := -I. -Iinclude
TARGET := quadtrix
SRCS   := main.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(IFLAGS) -o $@ $^
	@echo "✓ Built $(TARGET)"

# Optimised release (same flags, explicit target)
release: $(SRCS)
	$(CC) $(CFLAGS) $(IFLAGS) -DNDEBUG -o $(TARGET) $^
	strip $(TARGET)

# Debug build
debug: $(SRCS)
	$(CC) -std=c++17 -O0 -g -fsanitize=address,undefined \
	      $(IFLAGS) -o $(TARGET)-debug $^

benchmark-bin: benchmark.cpp
	$(CC) $(CFLAGS) $(IFLAGS) -o quadtrix-bench $^

clean-native:
	rm -f $(TARGET) $(TARGET)-debug quadtrix-bench

# ── Docker / Compose targets ─────────────────────────────────────────────────
build:
	$(SCRIPT) up

run: build
	@echo "Stack already started."

dev:
	$(SCRIPT) dev

gpu:
	$(SCRIPT) gpu

train-cpp:
	$(SCRIPT) train-cpp

train-torch:
	$(SCRIPT) train-torch

bench:
	$(SCRIPT) bench

logs:
	$(SCRIPT) logs

ps:
	$(SCRIPT) ps

shell:
	$(SCRIPT) shell $(SERVICE)

clean:
	$(SCRIPT) clean

# ── Misc ─────────────────────────────────────────────────────────────────────
format:
	find . \( -name "*.cpp" -o -name "*.h" \) \
	  ! -path "./build/*" \
	  | xargs clang-format -i --style=LLVM

lint-py:
	ruff check backend/ engine/

help:
	@echo ""
	@echo "  Quadtrix.cpp — make targets"
	@echo ""
	@echo "  Native:"
	@echo "    make              Build C++ binary (native)"
	@echo "    make release      Stripped release binary"
	@echo "    make debug        Debug binary with ASan/UBSan"
	@echo "    make clean-native Remove native build artifacts"
	@echo "    make format       Run clang-format on all C++ files"
	@echo ""
	@echo "  Docker:"
	@echo "    make build        docker compose up --build (CPU)"
	@echo "    make dev          Hot-reload dev stack"
	@echo "    make gpu          CUDA GPU stack"
	@echo "    make train-cpp    Train with C++ inside Docker"
	@echo "    make train-torch  Train with PyTorch inside Docker"
	@echo "    make bench        Run benchmark"
	@echo "    make logs         Tail all logs"
	@echo "    make ps           Show container status"
	@echo "    make shell        Shell into backend (SERVICE=frontend to change)"
	@echo "    make clean        Remove containers + volumes"
	@echo ""
