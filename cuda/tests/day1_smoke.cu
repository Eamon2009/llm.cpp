#include "../includes/memory.cuh"
#include "../includes/runtime.cuh"
#include "../includes/tensor.cuh"

#include <cstdio>

int main() {
    using namespace quadtrix::cuda;

    if (device_count() <= 0) {
        std::fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }

    const std::int64_t dims[] = {2, 3};
    Tensor tensor(dims, 2, DType::F32, 0);

    float host_in[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float host_out[6] = {};

    Stream stream = Stream::create();
    Status h2d = copy_h2d(tensor.data(), host_in, sizeof(host_in), stream.handle);
    if (!h2d.ok) {
        return 2;
    }

    Status d2h = copy_d2h(host_out, tensor.data(), sizeof(host_out), stream.handle);
    if (!d2h.ok) {
        return 3;
    }
    stream.synchronize();

    for (int i = 0; i < 6; ++i) {
        if (host_out[i] != host_in[i]) {
            std::fprintf(stderr, "Mismatch at %d: got %f expected %f\n", i, host_out[i], host_in[i]);
            return 4;
        }
    }

    std::printf("Day 1 CUDA runtime smoke test passed\n");
    return 0;
}
