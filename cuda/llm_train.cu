#include "includes/adamw.cuh"
#include "includes/checkpoint.h"
#include "includes/common.h"
#include "includes/dataloader.h"
#include "includes/global_norm.cuh"
#include "includes/logger.h"
#include "includes/memory.cuh"
#include "includes/runtime.cuh"
#include "includes/schedulers.h"
#include "includes/tensor.cuh"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace quadtrix {
namespace cuda {
namespace {

struct TrainingConfig {
    int batch_size = 1;
    int sequence_length = 8;
    int channels = 32;
    int num_heads = 4;
    int vocab_size = 256;
    int total_steps = 1;
    int warmup_steps = 1;
    float max_lr = 1.0e-4f;
    float min_lr = 1.0e-5f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float epsilon = 1.0e-8f;
    float weight_decay = 0.01f;
    float grad_clip = 1.0f;
};

struct TrainingBuffers {
    Tensor params;
    Tensor grads;
    Tensor first_moment;
    Tensor second_moment;
    Tensor norm_partials;
    Tensor activations;
    Tensor grad_activations;
};

bool parse_int_arg(const char* arg, const char* prefix, int* out) {
    const std::size_t prefix_len = std::strlen(prefix);
    if (std::strncmp(arg, prefix, prefix_len) != 0) {
        return false;
    }
    *out = std::atoi(arg + prefix_len);
    return true;
}

TrainingConfig parse_config(int argc, char** argv) {
    TrainingConfig config;
    for (int i = 1; i < argc; ++i) {
        parse_int_arg(argv[i], "--steps=", &config.total_steps) ||
            parse_int_arg(argv[i], "--batch=", &config.batch_size) ||
            parse_int_arg(argv[i], "--seq=", &config.sequence_length) ||
            parse_int_arg(argv[i], "--channels=", &config.channels) ||
            parse_int_arg(argv[i], "--heads=", &config.num_heads) ||
            parse_int_arg(argv[i], "--vocab=", &config.vocab_size);
    }
    return config;
}

bool validate_config(const TrainingConfig& config) {
    if (config.batch_size <= 0 || config.sequence_length <= 0 || config.channels <= 0 ||
        config.num_heads <= 0 || config.vocab_size <= 0 || config.total_steps <= 0) {
        return false;
    }
    return config.channels % config.num_heads == 0;
}

TrainingBuffers allocate_buffers(const TrainingConfig& config, int device_id) {
    const std::int64_t param_dims[] = {config.channels * config.channels};
    const std::int64_t partial_dims[] = {256};
    const std::int64_t activation_dims[] = {config.batch_size, config.sequence_length, config.channels};

    TrainingBuffers buffers;
    buffers.params = Tensor(param_dims, 1, DType::F32, device_id);
    buffers.grads = Tensor(param_dims, 1, DType::F32, device_id);
    buffers.first_moment = Tensor(param_dims, 1, DType::F32, device_id);
    buffers.second_moment = Tensor(param_dims, 1, DType::F32, device_id);
    buffers.norm_partials = Tensor(partial_dims, 1, DType::F32, device_id);
    buffers.activations = Tensor(activation_dims, 3, DType::F32, device_id);
    buffers.grad_activations = Tensor(activation_dims, 3, DType::F32, device_id);
    return buffers;
}

Status zero_training_buffers(TrainingBuffers& buffers, cudaStream_t stream) {
    Status status = memset_device(buffers.params.data(), 0, buffers.params.bytes(), stream);
    if (!status.ok) {
        return status;
    }
    status = memset_device(buffers.grads.data(), 0, buffers.grads.bytes(), stream);
    if (!status.ok) {
        return status;
    }
    status = memset_device(buffers.first_moment.data(), 0, buffers.first_moment.bytes(), stream);
    if (!status.ok) {
        return status;
    }
    return memset_device(buffers.second_moment.data(), 0, buffers.second_moment.bytes(), stream);
}

Status run_optimizer_step(TrainingBuffers& buffers, const TrainingConfig& config, int step, cudaStream_t stream) {
    AdamWConfig adamw;
    adamw.learning_rate = cosine_learning_rate(step, config.warmup_steps, config.total_steps, config.max_lr, config.min_lr);
    adamw.beta1 = config.beta1;
    adamw.beta2 = config.beta2;
    adamw.epsilon = config.epsilon;
    adamw.weight_decay = config.weight_decay;
    adamw.step = step + 1;

    return adamw_update(
        buffers.params.view(),
        buffers.grads.view(),
        buffers.first_moment.view(),
        buffers.second_moment.view(),
        adamw,
        1.0f,
        stream);
}

}  // namespace
}  // namespace cuda
}  // namespace quadtrix

int main(int argc, char** argv) {
    using namespace quadtrix::cuda;

    const TrainingConfig config = parse_config(argc, argv);
    if (!validate_config(config)) {
        log_message(LogLevel::Error, "invalid training config");
        return 1;
    }

    const int devices = device_count();
    if (devices <= 0) {
        log_message(LogLevel::Error, "no CUDA devices found");
        return 2;
    }

    DeviceGuard guard(0);
    Stream stream = Stream::create();
    TrainingBuffers buffers = allocate_buffers(config, 0);
    Status zero_status = zero_training_buffers(buffers, stream.handle);
    if (!zero_status.ok) {
        return 3;
    }

    DataLoader dataloader;
    TokenBatchView batch;
    (void)dataloader.next(&batch);

    for (int step = 0; step < config.total_steps; ++step) {
        const float lr = cosine_learning_rate(step, config.warmup_steps, config.total_steps, config.max_lr, config.min_lr);
        log_message(LogLevel::Info, "step %d lr %.8f", step + 1, lr);

        Status update_status = run_optimizer_step(buffers, config, step, stream.handle);
        if (!update_status.ok) {
            log_message(LogLevel::Error, "optimizer step failed");
            return 4;
        }
    }

    stream.synchronize();
    log_message(LogLevel::Info, "Quadtrix CUDA training orchestrator finished dry run on %d device(s)", devices);
    return 0;
}
