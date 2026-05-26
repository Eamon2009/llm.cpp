#pragma once

#include "tensor.cuh"

namespace quadtrix {
namespace cuda {

struct CheckpointMetadata {
    int vocab_size = 0;
    int max_sequence_length = 0;
    int num_layers = 0;
    int num_heads = 0;
    int channels = 0;
};

inline bool load_checkpoint_metadata(const char*, CheckpointMetadata*) {
    return false;
}

inline bool save_tensor_checkpoint(const char*, const TensorView&) {
    return false;
}

}  // namespace cuda
}  // namespace quadtrix
