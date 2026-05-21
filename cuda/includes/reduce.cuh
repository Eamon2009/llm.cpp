#pragma once
#include "common.h"

#ifdef __CUDACC__

static constexpr unsigned FULL_MASK = 0xffffffff;
// Warp reductions
__device__ QX_INLINE float warpReduceSum(float v)
{
    v += __shfl_xor_sync(FULL_MASK, v, 16);
    v += __shfl_xor_sync(FULL_MASK, v, 8);
    v += __shfl_xor_sync(FULL_MASK, v, 4);
    v += __shfl_xor_sync(FULL_MASK, v, 2);
    v += __shfl_xor_sync(FULL_MASK, v, 1);
    return v;
}
__device__ QX_INLINE float warpReduceMax(float v)
{
    v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 16));
    v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 8));
    v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 4));
    v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 2));
    v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 1));
    return v;
}
__device__ QX_INLINE float warpReduceMin(float v)
{
    v = fminf(v, __shfl_xor_sync(FULL_MASK, v, 16));
    v = fminf(v, __shfl_xor_sync(FULL_MASK, v, 8));
    v = fminf(v, __shfl_xor_sync(FULL_MASK, v, 4));
    v = fminf(v, __shfl_xor_sync(FULL_MASK, v, 2));
    v = fminf(v, __shfl_xor_sync(FULL_MASK, v, 1));
    return v;
}
__device__ QX_INLINE float warpBroadcast(float v)
{
    return __shfl_sync(FULL_MASK, v, 0);
}
__device__ QX_INLINE float blockReduceSum(float v, float *smem)
{
    int lane = threadIdx.x % QX_WARP_SIZE;
    int wid = threadIdx.x / QX_WARP_SIZE;
    v = warpReduceSum(v);
    if (lane == 0)
        smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < blockDim.x / QX_WARP_SIZE) ? smem[lane] : 0.f;
    if (wid == 0)
        v = warpReduceSum(v);
    return v;
}
__device__ QX_INLINE float blockReduceMax(float v, float *smem)
{
    int lane = threadIdx.x % QX_WARP_SIZE;
    int wid = threadIdx.x / QX_WARP_SIZE;
    v = warpReduceMax(v);
    if (lane == 0)
        smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < blockDim.x / QX_WARP_SIZE) ? smem[lane] : QX_NEG_INF_F32;
    if (wid == 0)
        v = warpReduceMax(v);
    return v;
}

#endif // __CUDACC__
