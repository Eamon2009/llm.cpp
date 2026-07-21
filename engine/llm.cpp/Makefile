.PHONY: all build clean run dev gpu train bench logs ps shell help release debug benchmark-bin clean-native format

SHELL := /bin/bash
SCRIPT := ./scripts/build.sh
CC     := g++
CFLAGS := -std=c++17 -O3 -march=native -fopenmp
IFLAGS := -I. -Iinclude
TARGET := llm
SRCS   := llm.cpp
HDRS   := $(wildcard include/*.h)

all: $(TARGET)
$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(IFLAGS) -o $@ $(SRCS)
	@echo "Built $(TARGET) with OpenMP parallel support."

release: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(IFLAGS) -DNDEBUG -o $(TARGET) $(SRCS)
	strip $(TARGET)
debug: $(SRCS) $(HDRS)
	$(CC) -std=c++17 -O0 -g -fsanitize=address,undefined $(IFLAGS) -o $(TARGET)-debug $(SRCS)

benchmark-bin: benchmark.cpp $(HDRS)
	$(CC) $(CFLAGS) $(IFLAGS) -o llm-bench benchmark.cpp

clean-native:
	rm -f $(TARGET) $(TARGET)-debug llm-bench

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

format:
	find . \( -name "*.cpp" -o -name "*.h" \) \
	  ! -path "./build/*" \
	  | xargs clang-format -i --style=LLVM

help:
	@echo ""
	@echo "  llm.cpp — make targets"
	@echo ""
	@echo "  Native:"
	@echo "    make              Build C++ binary with OpenMP parallel support"
	@echo "    make release      Stripped release binary"
	@echo "    make debug        Debug binary with ASan/UBSan"
	@echo "    make clean-native Remove native build artifacts"
	@echo "    make format       Run clang-format on all C++ files"
	@echo ""