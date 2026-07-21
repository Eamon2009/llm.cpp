.PHONY: all release debug benchmark-bin clean-native format help

SHELL  := /bin/bash
CC     := g++
CFLAGS := -std=c++17 -O3 -march=native
IFLAGS := -I. -Iinclude
TARGET := llm
SRCS   := main.cpp
HDRS   := $(wildcard include/*.h)

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(IFLAGS) -o $@ $(SRCS)
	@echo "Built $(TARGET)"

release: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(IFLAGS) -DNDEBUG -o $(TARGET) $(SRCS)
	strip $(TARGET)
debug: $(SRCS) $(HDRS)
	$(CC) -std=c++17 -O0 -g -fsanitize=address,undefined $(IFLAGS) -o $(TARGET)-debug $(SRCS)

benchmark-bin: benchmark.cpp $(HDRS)
	$(CC) $(CFLAGS) $(IFLAGS) -o llm-bench benchmark.cpp

clean-native:
	rm -f $(TARGET) $(TARGET)-debug llm-bench

format:
	find . \( -name "*.cpp" -o -name "*.h" \) \
	  ! -path "./build/*" \
	  | xargs clang-format -i --style=LLVM

help:
	@echo ""
	@echo "  llm.cpp — native make targets"
	@echo ""
	@echo "    make              Build C++ binary (native)"
	@echo "    make release      Stripped release binary"
	@echo "    make debug        Debug binary with ASan/UBSan"
	@echo "    make benchmark-bin Build benchmark binary"
	@echo "    make clean-native Remove native build artifacts"
	@echo "    make format       Run clang-format on all C++ files"
	@echo ""