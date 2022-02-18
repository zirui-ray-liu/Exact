#include <torch/extension.h>

using torch::Tensor;

#define ELU_NUM_THREADS 512
template <typename scalar_t>
__global__ void elu_backward_cuda_kernel(const scalar_t* __restrict__ grad_output,
                                         scalar_t* __restrict__ grad_input,
                                         const scalar_t* __restrict__ data,
                                         int64_t N,
                                         float alpha){
    const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        if (data[id] >= 0){
            grad_input[id] = grad_output[id];
        }else{
            grad_input[id] = alpha * grad_output[id] * (static_cast<scalar_t>(std::exp(data[id])));
        }
    }                                
}

Tensor _elu_backward_cuda(Tensor grad_output, Tensor data, float alpha) {
    int64_t n_elements = 1;
    for (size_t i = 0; i < grad_output.dim(); ++i) {
        n_elements *= grad_output.size(i);
    }

    int threads = ELU_NUM_THREADS;
    int blocks = (n_elements + threads - 1) / threads;
    Tensor grad_input = torch::empty_like(grad_output);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "_elu_backward_cuda", ([&] {
        elu_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(), data.data_ptr<scalar_t>(),
            n_elements, alpha);
    }));
    return grad_input;
}