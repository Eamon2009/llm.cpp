#pragma once

// Aggregator — include this one header to get the full Day 1 runtime.
// Each sub-header is small and independently loadable.

#include "common.h"   // macros, enums, error checks, dtype helpers
#include "tensor.cuh" // TensorShape, Tensor struct
#include "memory.cuh" // allocators, tensor_alloc_*, tensor_free, transfers
#include "reduce.cuh" // warpReduceSum/Max/Min, blockReduceSum/Max
