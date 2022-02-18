import pdb
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.cuda.amp import custom_fwd, custom_bwd

from torch_sparse import matmul
from .conf import config
import exact.cpp_extension.spmm as spmm 

import exact.cpp_extension.backward_func as ext_backward_func
import exact.cpp_extension.quantization as ext_quantization

from exact.utils import empty_cache, get_memory_usage, compute_tensor_bytes, swap_to_cpu, cast_low_bit_int



def quantize_and_pack(data, bits, mn, mx):
    if config.simulate:
        N = data.shape[0]
        output = data  
        B = 2 ** bits - 1
        mn = mn - 1e-6
        mx = mx + 1e-6
        scale = B / (mx - mn) 
        output = (output - mn.view(-1, 1)) * scale.view(-1, 1)
        if config.stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)

        output = F.relu(output)
        output = output.round_().int()
    else:
        # Pack to bitstream
        assert type(bits) == int
        pack_func = ext_quantization.pack_single_precision
        scale = (2 ** bits - 1) / (mx - mn) 
        output = pack_func(data, mn, mx, scale.to(data.dtype), bits, config.stochastic)
        if config.swap:
            output = swap_to_cpu(output)
    return output, scale


def dequantize_and_unpack(data, shape, bits, scale, mn):
    if config.simulate:
        data = data / scale.view(-1, 1) + mn.view(-1, 1)
    else:
        if config.swap:
            data = data.cuda()
        N = shape[0]
        num_features = int(np.prod(shape[1:]))
        # Unpack bitstream
        assert type(bits) == int
        unpack_func = ext_quantization.unpack_single_precision
        data = unpack_func(data, bits, scale, mn, N, num_features)
    return data


def no_scheme_compute_quantization_bits(input):
    N = input.shape[0]
    input_flatten = input.view(N, -1)
    mn, mx = torch.min(input_flatten, 1)[0], torch.max(input_flatten, 1)[0]
    b = config.activation_compression_bits[0]
    return input_flatten, b, mn, mx


def quantize_activation(input, scheme):
    if not config.compress_activation:
        if config.swap:
            input = swap_to_cpu(input)
        return input, None, None, None
    if scheme:
        input_groups, q_bits, q_min, mx = scheme.compute_quantization_bits(input)
    else:
        input_groups, q_bits, q_min, mx = no_scheme_compute_quantization_bits(input)

    q_input, q_scale = quantize_and_pack(input_groups, q_bits, q_min, mx)
    if input.dtype == torch.float32:
        return q_input, q_bits, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16)
    else:
        return q_input, q_bits, q_scale, q_min


def dequantize_activation(quantized, q_input_shape):
    if not config.compress_activation:
        ret = quantized[0]
        if config.swap:
            ret = ret.cuda(non_blocking=True)
        return ret

    q_input, q_bits, q_scale, q_min = quantized
    if q_scale.dtype == torch.bfloat16:
        q_scale = q_scale.to(torch.float32)
        q_min = q_min.to(torch.float32)
    input = dequantize_and_unpack(q_input, q_input_shape, q_bits, q_scale, q_min)
    return input.contiguous()


linear_layer_ct = 0
qmatmul_layer_ct = 0
bn_layer_ct = 0
total_act_mem = 0
GPU = 0

@torch.no_grad()
def input2rp(input, kept_acts):
    assert len(input.size()) == 2
    rand_mat_size = (input.shape[1], kept_acts)
    # Create random matrix
    def gen_rad_mat(rm_size, feat_size, device, dtype):
        bern = torch.randint(2, size=rm_size, device=device, requires_grad=False, dtype=dtype)
        return (2.0 * bern - 1) / feat_size **0.5

    rand_matrix = gen_rad_mat(rand_mat_size, kept_acts, input.device, input.dtype)
    dim_reduced_input = torch.matmul(input, rand_matrix)
    return dim_reduced_input, rand_matrix


@torch.no_grad()
def rp2input(dim_reduced_input, input_shape, rand_matrix):
    assert len(dim_reduced_input.size()) == 2
    input = torch.matmul(dim_reduced_input, rand_matrix.t())    
    return input.view(input_shape)


class qlinear(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias=None, quantized=None, randmat=None, scheme=None, rp=True):
        if quantized is not None:
            if randmat is None:
                assert rp is False or config.kept_frac == 1.0
                # quantized somewhere before
                ori_input_shape, proj_input_shape = input.shape, input.shape
            else:
                assert (rp is True and config.kept_frac < 1.0) and (input.shape[1] == randmat.shape[0])
                ori_input_shape, proj_input_shape = input.shape, torch.Size([input.shape[0], randmat.shape[1]])
                # this is quantized random projected data
        else:
            if config.kept_frac < 1.0 and rp:
                kept_acts = int(config.kept_frac * input.shape[1] + 0.999)
                dim_reduced_input, randmat = input2rp(input, kept_acts)
                ori_input_shape, proj_input_shape = input.shape, dim_reduced_input.shape
            else:
                dim_reduced_input, randmat = input, None
                ori_input_shape, proj_input_shape = input.shape, input.shape
            quantized = quantize_activation(dim_reduced_input, scheme)

        empty_cache(config.empty_cache_threshold)

        ctx.scheme = scheme
        ctx.saved = quantized, weight, bias, randmat
        ctx.other_args = ori_input_shape, proj_input_shape, rp
        res = F.linear(input, weight, bias)
        return res

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        quantized, weight, bias, randmat = ctx.saved
        ori_input_shape, q_input_shape, rp = ctx.other_args
        input = dequantize_activation(quantized, q_input_shape)
        if config.kept_frac < 1.0 and rp:
            input = rp2input(input, ori_input_shape, randmat)
        del quantized, ctx.saved
        empty_cache(config.empty_cache_threshold)
        grad_input, grad_weight, grad_bias = ext_backward_func.linear_backward(grad_output, input, weight, bias)
        # grad_input = grad_output.mm(weight)
        # grad_weight = grad_output.t().mm(input)
        # if bias is not None:
        #     grad_bias = grad_output.sum(0)
        # else:
        #     grad_bias = None
        del input, grad_output
        empty_cache(config.empty_cache_threshold)
        return grad_input, grad_weight, grad_bias, None, None, None, None



class qelu(Function):
    @staticmethod
    def forward(ctx, input, alpha, scheme=None):
        quantized = quantize_activation(input, scheme)
        empty_cache(config.empty_cache_threshold)
        ctx.scheme = scheme
        ctx.saved = quantized
        ctx.other_args = input.shape, alpha
        res = F.elu(input, alpha)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        # if ctx.scheme:
        #     ctx.scheme.set_scale(grad_output)

        quantized = ctx.saved
        q_input_shape, alpha = ctx.other_args
        input = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved
        empty_cache(config.empty_cache_threshold)
        grad_input = ext_backward_func._elu_backward_cuda(grad_output, input, alpha)
        return grad_input, None, None

class qbatch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps, scheme):
        quantized = quantize_activation(input, scheme)

        if training:
            output, save_mean, save_var, reserve, _ = ext_backward_func._batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps, True)

        else:
            output, save_mean, save_var = ext_backward_func.native_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
            reserve = None

        # output, save_mean, save_var = ext_backward_func.native_batch_norm(
        #     input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
        # reserve = None
        ctx.scheme = scheme
        ctx.other_args = input.shape
        ctx.saved = (quantized, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        quantized, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved

        q_input_shape = ctx.other_args

        input = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved

        empty_cache(config.empty_cache_threshold)

        # if training:
        #     input = input.contiguous()
        #     grad_input, grad_weight, grad_bias = ext_backward_func.cudnn_batch_norm_backward(
        #         input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reserve)
        # else:
        #     grad_input, grad_weight, grad_bias = ext_backward_func.native_batch_norm_backward(
        #         grad_output, input, weight, running_mean, running_var, save_mean, save_var, training, eps,
        #         [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]]
        #     )

        grad_input, grad_weight, grad_bias = ext_backward_func.native_batch_norm_backward(
            grad_output, input, weight, running_mean, running_var, save_mean, save_var, training, eps,
            [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]]
        )
        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None


class qspmm_sum(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, row, rowptr, col, value, colptr, csr2csc, has_value, other, quantized=None, randmat=None, scheme=None):
        result = spmm.spmm_sum_fw(row, rowptr, col, value, colptr, csr2csc, other)
        if quantized is not None:
            if randmat is None:
                assert config.kept_frac == 1.0
                # quantized somewhere before
                ori_input_shape, proj_input_shape = other.shape, other.shape
            else:
                assert config.kept_frac < 1.0 and (other.shape[1] == randmat.shape[0])
                ori_input_shape, proj_input_shape = other.shape, torch.Size([other.shape[0], randmat.shape[1]])
                # this is quantized random projected data
        else:
            if config.kept_frac < 1.0:
                kept_acts = int(config.kept_frac * other.shape[1] + 0.999)
                dim_reduced_input, randmat = input2rp(other, kept_acts)
                ori_input_shape, proj_input_shape = other.shape, dim_reduced_input.shape
            else:
                dim_reduced_input, randmat = other, None
                ori_input_shape, proj_input_shape = other.shape, other.shape
            quantized = quantize_activation(dim_reduced_input, scheme)
        
        empty_cache(config.empty_cache_threshold)
        ctx.saved = row, rowptr, col, value, colptr, csr2csc, quantized, randmat
        ctx.other_args = has_value, ori_input_shape, proj_input_shape, value.requires_grad if has_value else False, other.requires_grad
        ctx.scheme = scheme
        return result


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        row, rowptr, col, value, colptr, csr2csc, quantized, randmat = ctx.saved
        row = col if row is None else row
        value = col if value is None else value
        colptr = col if colptr is None else colptr
        csr2csc = col if csr2csc is None else csr2csc
        has_value, ori_input_shape, q_input_shape, value_requires_grad, mat_requires_grad = ctx.other_args
        other = dequantize_activation(quantized, q_input_shape)
        if config.kept_frac < 1.0:
            other = rp2input(other, ori_input_shape, randmat)
        del quantized, ctx.saved
        empty_cache(config.empty_cache_threshold)
        grad_value, grad_mat = spmm.spmm_sum_bw(row, rowptr, col, value, colptr, csr2csc, other, grad_outputs, 
                                                has_value, value_requires_grad, mat_requires_grad)
        del other
        empty_cache(config.empty_cache_threshold)
        return None, None, None, grad_value, None, None, None, grad_mat, None, None, None


class qspmm_mean(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, row, rowptr, col, value, rowcount, colptr, csr2csc, has_value, other, 
                quantized=None, randmat=None, scheme=None):
        result = spmm.spmm_mean_fw(row, rowptr, col, value, rowcount, colptr, csr2csc, other)
        if quantized is not None:
            if randmat is None:
                assert config.kept_frac == 1.0
                # quantized somewhere before without random projection
                ori_input_shape, proj_input_shape = other.shape, other.shape
            else:
                assert config.kept_frac < 1.0 and (other.shape[1] == randmat.shape[0])
                ori_input_shape, proj_input_shape = other.shape, torch.Size([other.shape[0], randmat.shape[1]])
                # this is quantized random projected data
        else:
            if config.kept_frac < 1.0:
                kept_acts = int(config.kept_frac * other.shape[1] + 0.999)
                dim_reduced_input, randmat = input2rp(other, kept_acts)
                ori_input_shape, proj_input_shape = other.shape, dim_reduced_input.shape
            else:
                dim_reduced_input, randmat = other, None
                ori_input_shape, proj_input_shape = other.shape, other.shape
            quantized = quantize_activation(dim_reduced_input, scheme)
        empty_cache(config.empty_cache_threshold)
        ctx.saved = row, rowptr, col, value, rowcount, colptr, csr2csc, quantized, randmat
        ctx.other_args = has_value, ori_input_shape, proj_input_shape, value.requires_grad if has_value else False, other.requires_grad
        ctx.scheme = scheme
        return result

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        row, rowptr, col, value, rowcount, colptr, csr2csc, quantized, randmat = ctx.saved
        row = col if row is None else row
        value = col if value is None else value
        rowcount = col if rowcount is None else rowcount
        colptr = col if colptr is None else colptr
        csr2csc = col if csr2csc is None else csr2csc
        has_value, ori_input_shape, q_input_shape, value_requires_grad, mat_requires_grad = ctx.other_args
        # here is one ugly trick: if we know value does not need gradient,
        # we actually do not need the ``other'' matrix to calculate the gradient.
        # So here we just pass a dummy matrix to the CUDA kernel.
        # TODO: engineering optimization.
        if value_requires_grad:
            other = dequantize_activation(quantized, q_input_shape)
            if config.kept_frac < 1.0:
                other = rp2input(other, ori_input_shape, randmat)
        else:
            if quantized[2].dtype == torch.bfloat16:
                dtype = torch.float
            else:
                dtype = quantized[2].dtype 
            other = torch.tensor([1.], dtype=dtype, device=quantized[2].device)
        del quantized, ctx.saved
        empty_cache(config.empty_cache_threshold)
        grad_value, grad_mat = spmm.spmm_mean_bw(row, rowptr, col, value, rowcount, colptr, csr2csc, other, grad_outputs, 
                                                 has_value, value_requires_grad, mat_requires_grad)
        del other
        empty_cache(config.empty_cache_threshold)
        return None, None, None, grad_value, None, None, None, None, grad_mat, None, None, None


class qspmm_max(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, rowptr, col, value, has_value, other, 
                quantized=None, randmat=None, scheme=None):
        output, arg_out = spmm.spmm_max_fw(rowptr, col, value, other)
        if quantized is None:
            quantized = quantize_activation(other, scheme)
        else:
            assert isinstance(quantized, tuple)
        empty_cache(config.empty_cache_threshold)
        ctx.saved = col, value, quantized, arg_out
        ctx.other_args = has_value, other.shape, value.requires_grad if has_value else False, other.requires_grad
        ctx.mark_non_differentiable(arg_out)
        ctx.scheme = scheme
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        col, value, quantized, arg_out = ctx.saved
        value = col if value is None else value
        has_value, q_input_shape, value_requires_grad, mat_requires_grad = ctx.other_args
        other = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved
        empty_cache(config.empty_cache_threshold)
        grad_value, grad_mat = spmm.spmm_max_bw(col, value, other, arg_out, grad_outputs, 
                                                has_value, value_requires_grad, mat_requires_grad)
        return None, None, grad_value, None, grad_mat, None, None, None


class qspmm_min(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, rowptr, col, value, has_value, other, 
                quantized=None, randmat=None, scheme=None):
        output, arg_out =  spmm.spmm_min_fw(rowptr, col, value, other)
        if quantized is None:
            quantized = quantize_activation(other, scheme)
        else:
            assert isinstance(quantized, tuple)
        empty_cache(config.empty_cache_threshold)
        ctx.saved = col, value, quantized, arg_out
        ctx.other_args = has_value, other.shape, value.requires_grad if has_value else False, other.requires_grad
        ctx.mark_non_differentiable(arg_out)
        ctx.scheme = scheme
        return output
   
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        col, value, quantized, arg_out = ctx.saved
        value = col if value is None else value
        has_value, q_input_shape, value_requires_grad, mat_requires_grad = ctx.other_args
        other = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved
        empty_cache(config.empty_cache_threshold)
        # if ctx.scheme:
        #     ctx.scheme.set_scale(grad_outputs)
        grad_value, grad_mat = spmm.spmm_min_bw(col, value, other, arg_out, grad_outputs, 
                                                has_value, value_requires_grad, mat_requires_grad)
        return None, None, grad_value, None, grad_mat, None, None, None