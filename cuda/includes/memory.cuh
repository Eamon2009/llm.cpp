#pragma once
#include "common.h"
#include "tensor.cuh"
#include <cstring>
static inline void *qx_host_alloc(size_t n)
{
    void *p = malloc(n);
    if (!p && n)
    {
        perror("[QX] malloc");
        exit(1);
    }
    return p;
}
static inline void qx_host_free(void *p)
{
    free(p);
}

static inline void *qx_pinned_alloc(size_t n)
{
    void *p = nullptr;
    CUDA_CHECK(cudaMallocHost(&p, n));
    return p;
}
static inline void qx_pinned_free(void *p)
{
    if (p)
        CUDA_CHECK(cudaFreeHost(p));
}

static inline void *qx_device_alloc(size_t n, int dev = 0)
{
    CUDA_CHECK(cudaSetDevice(dev));
    void *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, ROUND_UP(n, QX_MEM_ALIGN)));
    return p;
}
static inline void qx_device_free(void *p)
{
    if (p)
        CUDA_CHECK(cudaFree(p));
}
static inline void qx_device_zero(void *p, size_t n, cudaStream_t s = 0)
{
    if (p && n)
        CUDA_CHECK(cudaMemsetAsync(p, 0, n, s));
}
// Tensor allocators
static inline Tensor *tensor_alloc_device(const TensorShape &sh, DType dt,
                                          int dev = 0, cudaStream_t s = 0,
                                          const char *name = "")
{
    Tensor *t = (Tensor *)calloc(1, sizeof(Tensor));
    t->shape = sh;
    t->dtype = dt;
    t->mem_loc = MEM_DEVICE;
    t->owns_data = true;
    t->device_id = dev;
    strncpy(t->name, name, 63);
    t->data = qx_device_alloc((size_t)sh.numel() * dtype_size(dt), dev);
    qx_device_zero(t->data, (size_t)sh.numel() * dtype_size(dt), s);
    return t;
}

static inline Tensor *tensor_alloc_host(const TensorShape &sh, DType dt,
                                        bool pinned = false, const char *name = "")
{
    Tensor *t = (Tensor *)calloc(1, sizeof(Tensor));
    t->shape = sh;
    t->dtype = dt;
    t->mem_loc = pinned ? MEM_HOST_PINNED : MEM_HOST;
    t->owns_data = true;
    t->device_id = -1;
    strncpy(t->name, name, 63);
    size_t nb = (size_t)sh.numel() * dtype_size(dt);
    t->data = pinned ? qx_pinned_alloc(nb) : calloc(1, nb);
    return t;
}

static inline void tensor_free(Tensor *t)
{
    if (!t)
        return;
    if (t->owns_data && t->data)
    {
        if (t->mem_loc == MEM_DEVICE)
            qx_device_free(t->data);
        else if (t->mem_loc == MEM_HOST_PINNED)
            qx_pinned_free(t->data);
        else
            free(t->data);
    }
    free(t);
}
static inline void tensor_h2d(Tensor *dst, const Tensor *src, cudaStream_t s = 0)
{
    CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, dst->nbytes(), cudaMemcpyHostToDevice, s));
}
static inline void tensor_d2h(Tensor *dst, const Tensor *src, cudaStream_t s = 0)
{
    CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, dst->nbytes(), cudaMemcpyDeviceToHost, s));
}
static inline void tensor_d2d(Tensor *dst, const Tensor *src, cudaStream_t s = 0)
{
    CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, dst->nbytes(), cudaMemcpyDeviceToDevice, s));
}
