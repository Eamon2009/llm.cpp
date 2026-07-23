#ifndef CUDA_TENSOR_CUH
#define CUDA_TENSOR_CUH

#include "cuda_utils.cuh"

#include <cassert>
#include <numeric>
#include <vector>

struct CUDATensor
{
      std::vector<int> shape;
      float *d_data = nullptr;   // VRAM device pointer
      std::vector<float> h_data; // Host RAM cache
      bool is_on_device = false;

      CUDATensor() = default;

      CUDATensor(std::vector<int> sh, float fill = 0.0f) : shape(std::move(sh))
      {
            int total = numel();
            h_data.assign(total, fill);
      }

      ~CUDATensor()
      {
            free_device();
      }

      // Move semantics for efficient memory transfers
      CUDATensor(CUDATensor &&other) noexcept
            : shape(std::move(other.shape)), h_data(std::move(other.h_data)), d_data(other.d_data),
              is_on_device(other.is_on_device)
      {
            other.d_data = nullptr;
            other.is_on_device = false;
      }

      CUDATensor &operator=(CUDATensor &&other) noexcept
      {
            if (this != &other)
            {
                  free_device();
                  shape = std::move(other.shape);
                  h_data = std::move(other.h_data);
                  d_data = other.d_data;
                  is_on_device = other.is_on_device;
                  other.d_data = nullptr;
                  other.is_on_device = false;
            }
            return *this;
      }

      int numel() const
      {
            int n = 1;
            for (int d : shape)
                  n *= d;
            return n;
      }

      int ndim() const
      {
            return static_cast<int>(shape.size());
      }

      void free_device()
      {
            if (d_data)
            {
                  cudaFree(d_data);
                  d_data = nullptr;
                  is_on_device = false;
            }
      }

      void allocate_device()
      {
            if (!is_on_device)
            {
                  size_t bytes = numel() * sizeof(float);
                  CUDA_CHECK(cudaMalloc(&d_data, bytes));
                  is_on_device = true;
            }
      }

      void to_device()
      {
            allocate_device();
            if (!h_data.empty())
            {
                  size_t bytes = numel() * sizeof(float);
                  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
            }
      }

      void to_host()
      {
            if (is_on_device)
            {
                  h_data.resize(numel());
                  size_t bytes = numel() * sizeof(float);
                  CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
            }
      }
};

#endif