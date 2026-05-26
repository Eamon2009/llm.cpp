#pragma once

#include <cmath>

namespace quadtrix {
namespace cuda {

inline float cosine_learning_rate(
    int step,
    int warmup_steps,
    int total_steps,
    float max_lr,
    float min_lr) {
    if (step < warmup_steps) {
        return max_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
    }
    if (step >= total_steps) {
        return min_lr;
    }
    const float progress = static_cast<float>(step - warmup_steps) / static_cast<float>(total_steps - warmup_steps);
    const float coeff = 0.5f * (1.0f + std::cos(3.14159265358979323846f * progress));
    return min_lr + coeff * (max_lr - min_lr);
}

}  // namespace cuda
}  // namespace quadtrix
