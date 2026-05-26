#include "../includes/zero.cuh"

#include <cstdio>

int main() {
    using namespace quadtrix::cuda;

    const ShardRange r0 = zero_shard_range(10, 3, 0);
    const ShardRange r1 = zero_shard_range(10, 3, 1);
    const ShardRange r2 = zero_shard_range(10, 3, 2);

    if (r0.offset != 0 || r0.length != 4) {
        return 1;
    }
    if (r1.offset != 4 || r1.length != 3) {
        return 2;
    }
    if (r2.offset != 7 || r2.length != 3) {
        return 3;
    }

    std::printf("Day 6 distributed helper smoke test passed\n");
    return 0;
}
