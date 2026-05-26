#pragma once

#include <cstddef>
#include <cstdint>

namespace quadtrix {
namespace cuda {

struct TokenBatchView {
    const std::int32_t* inputs = nullptr;
    const std::int32_t* targets = nullptr;
    int batch_size = 0;
    int sequence_length = 0;
};

class DataLoader {
public:
    DataLoader() = default;

    bool next(TokenBatchView* batch) {
        if (batch != nullptr) {
            *batch = {};
        }
        return false;
    }
};

}  // namespace cuda
}  // namespace quadtrix
