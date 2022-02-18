/*
 * Cuda operators for quantization and packing
 */

#include <torch/extension.h>
#include <torch/torch.h>

#include "ext_common.h"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;
using torch::Tensor;
using torch::IntArrayRef;

// Declarations for functions in ext_quantization_cuda_kernel.cu
// Pack and unpack
Tensor pack_single_precision_cuda(
    Tensor data, Tensor min, Tensor max, Tensor scale, int bits, bool stochastic);
Tensor unpack_single_precision_cuda(
    Tensor data, int bits, Tensor scale, Tensor min, int64_t N, int group_size);

// ActQuantizedReLU
std::pair<Tensor, Tensor> act_quantized_relu_forward_cuda(Tensor data);
Tensor act_quantized_relu_backward_cuda(Tensor grad_output, Tensor mask);

// ActQuantizedLeakyReLU
std::pair<Tensor, Tensor> act_quantized_leakyrelu_forward_cuda(Tensor data, float slope);
Tensor act_quantized_leakyrelu_backward_cuda(Tensor grad_output, Tensor mask, float slope);

// ActQuantizedDropout
std::pair<Tensor, Tensor> act_quantized_dropout_forward_cuda(Tensor data, float p);
Tensor act_quantized_dropout_backward_cuda(Tensor grad_output, Tensor mask, float p1m);


// Pack/Unpack
Tensor pack_single_precision(Tensor data,
                                                Tensor min,
                                                Tensor max,
                                                Tensor scale,
                                                int bits,
                                                bool stochastic) {
  CHECK_CUDA_TENSOR_DIM_FLOAT(data, 2);
  CHECK_CUDA_TENSOR_DIM_FLOAT(min, 1);
  CHECK_CUDA_TENSOR_DIM_FLOAT(max, 1);
  CHECK_CUDA_TENSOR_DIM_FLOAT(scale, 1);
  return pack_single_precision_cuda(data, min, max, scale, bits, stochastic);
}

Tensor unpack_single_precision(Tensor data,
                               int bits,
                               Tensor scale,
                               Tensor min,
                               int64_t N,
                               int64_t group_size) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 1, torch::kInt8);
  CHECK_CUDA_TENSOR_DIM_FLOAT(scale, 1);
  CHECK_CUDA_TENSOR_DIM_FLOAT(min, 1);

  return unpack_single_precision_cuda(data, bits, scale, min,
                                      N, group_size);
}


// Activation quantized relu: use compressed bit stream to store activation
class ActQuantizedReLU : public Function<ActQuantizedReLU> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input) {
    Tensor output, mask;
    std::tie(output, mask) = act_quantized_relu_forward_cuda(input);
    ctx->save_for_backward({mask});
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    return {act_quantized_relu_backward_cuda(grad_outputs[0], saved[0])};
  }
};

Tensor act_quantized_relu(Tensor input) {
  CHECK_CUDA_TENSOR_FLOAT(input);
  return ActQuantizedReLU::apply(input);
}

// Activation quantized leakyrelu: use compressed bit stream to store activation
class ActQuantizedLeakyReLU : public Function<ActQuantizedLeakyReLU> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, float slope) {
    Tensor output, mask;
    std::tie(output, mask) = act_quantized_leakyrelu_forward_cuda(input, slope);
    ctx->save_for_backward({mask});
    ctx->saved_data["slope"] = slope;
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    float slope = ctx->saved_data["slope"].toDouble();
    return {act_quantized_leakyrelu_backward_cuda(grad_outputs[0], saved[0], slope), Tensor()};
  }
};

Tensor act_quantized_leaky_relu(Tensor input, float slope) {
  CHECK_CUDA_TENSOR_FLOAT(input);
  return ActQuantizedLeakyReLU::apply(input, slope);
}


// Activiation quantized dropout: use compressed bit stream to store masks
class ActQuantizedDropout : public Function<ActQuantizedDropout> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, float p, bool train) {
    Tensor output, mask;
    if (!train){
      return input;
    }
    std::tie(output, mask) = act_quantized_dropout_forward_cuda(input, p);
    ctx->save_for_backward({mask});
    ctx->saved_data["p1m"] = 1. - p;
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    float p1m = ctx->saved_data["p1m"].toDouble();
    return {act_quantized_dropout_backward_cuda(grad_outputs[0], saved[0], p1m), Tensor(), Tensor()};
  }
};

Tensor act_quantized_dropout(Tensor input, float p, bool train) {
  CHECK_CUDA_TENSOR_FLOAT(input);
  return ActQuantizedDropout::apply(input, p, train);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_single_precision", &pack_single_precision);
  m.def("unpack_single_precision", &unpack_single_precision);
  m.def("act_quantized_relu", &act_quantized_relu);
  m.def("act_quantized_leaky_relu", &act_quantized_leaky_relu);
  m.def("act_quantized_dropout", &act_quantized_dropout);
}