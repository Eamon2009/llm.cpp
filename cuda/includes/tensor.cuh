#pragma once
#include "common.h"
// TensorShape — dimensions + strides (row-major by default)
struct QX_ALIGN_16 TensorShape
{
    int dims[QX_MAX_DIMS];
    int strides[QX_MAX_DIMS];
    int ndim;
    int _pad;

    QX_HOST_DEVICE QX_INLINE int64_t numel() const
    {
        int64_t n = 1;
        for (int i = 0; i < ndim; i++)
            n *= dims[i];
        return n;
    }

    QX_HOST QX_INLINE void compute_strides()
    {
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * dims[i + 1];
    }

    QX_HOST QX_INLINE bool is_contiguous() const
    {
        int expected = 1;
        for (int i = ndim - 1; i >= 0; i--)
        {
            if (strides[i] != expected)
                return false;
            expected *= dims[i];
        }
        return true;
    }
};

static inline TensorShape make_shape(const int *d, int ndim)
{
    TensorShape s;
    s.ndim = ndim;
    s._pad = 0;
    for (int i = 0; i < ndim; i++)
        s.dims[i] = d[i];
    for (int i = ndim; i < QX_MAX_DIMS; i++)
    {
        s.dims[i] = 1;
        s.strides[i] = 1;
    }
    s.compute_strides();
    return s;
}
static inline TensorShape make_shape1d(int a)
{
    int d[] = {a};
    return make_shape(d, 1);
}
static inline TensorShape make_shape2d(int a, int b)
{
    int d[] = {a, b};
    return make_shape(d, 2);
}
static inline TensorShape make_shape3d(int a, int b, int c)
{
    int d[] = {a, b, c};
    return make_shape(d, 3);
}
static inline TensorShape make_shape4d(int a, int b, int c, int e)
{
    int d[] = {a, b, c, e};
    return make_shape(d, 4);
}
// Tensor — primary data carrier (host struct, kernels get raw pointers)
struct Tensor
{
    void *data;
    TensorShape shape;
    DType dtype;
    MemLocation mem_loc;
    bool owns_data;
    int device_id;
    char name[64];

    template <typename T>
    QX_HOST_DEVICE QX_INLINE T *as()
    {
        return reinterpret_cast<T *>(data);
    }
    template <typename T>
    QX_HOST_DEVICE QX_INLINE const T *as() const
    {
        return reinterpret_cast<const T *>(data);
    }

    QX_HOST QX_INLINE size_t nbytes() const { return (size_t)shape.numel() * dtype_size(dtype); }
    QX_HOST_DEVICE QX_INLINE int dim(int i) const { return shape.dims[i]; }
    QX_HOST_DEVICE QX_INLINE int ndim() const { return shape.ndim; }
    QX_HOST_DEVICE QX_INLINE int64_t numel() const { return shape.numel(); }
};
