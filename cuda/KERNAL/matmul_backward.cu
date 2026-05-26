#include "../includes/matmul.cuh"

namespace quadtrix {
namespace cuda {

BlasStatus matmul_backward_input(
    BlasHandle& handle,
    const TensorView& grad_output,
    const TensorView& weight,
    TensorView grad_input,
    cudaStream_t stream,
    float alpha,
    float beta) {
    return matmul(
        handle,
        grad_output,
        MatmulTranspose::None,
        weight,
        MatmulTranspose::Transpose,
        grad_input,
        alpha,
        beta,
        stream);
}

BlasStatus matmul_backward_weight(
    BlasHandle& handle,
    const TensorView& input,
    const TensorView& grad_output,
    TensorView grad_weight,
    cudaStream_t stream,
    float alpha,
    float beta) {
    return matmul(
        handle,
        input,
        MatmulTranspose::Transpose,
        grad_output,
        MatmulTranspose::None,
        grad_weight,
        alpha,
        beta,
        stream);
}

}  // namespace cuda
}  // namespace quadtrix
