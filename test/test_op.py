"""Test the activation quantized ops"""

import copy
import numpy as np
from numpy import random
import torch
import torch_sparse
from torch.nn import functional as F


from timeit_v2 import py_benchmark

import exact.cpp_extension.quantization as ext_quantization
from exact import get_memory_usage, compute_tensor_bytes
from exact.conf import config
from exact.ops import qlinear, qbatch_norm, qelu
from exact.spmm import qmatmul

config.activation_compression_bits = [2]
config.compress_activation = True

def test_relu_correctness():
    print("========== ReLU Correctness Test ==========")

    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.randn(10000, 128).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            output = func(data)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad]]

        output_ref, grad_data_ref = test_implementation(F.relu)
        output_us, grad_data_us = test_implementation(ext_quantization.act_quantized_relu)

        np.testing.assert_allclose(output_ref, output_us)
        np.testing.assert_allclose(grad_data_ref, grad_data_us)


def test_relu_memory():
    print("========== ReLU Memory Test ==========")

    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.randn(10000, 128).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            before = get_memory_usage(0)

            for i in range(10):
                data = func(data)

            after = get_memory_usage(0)

            return after - before

        usage_ref = test_implementation(F.relu)
        usage_us = test_implementation(ext_quantization.act_quantized_relu)

        print("Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
        print("Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))


def test_relu_speed():
    print("========== ReLU Speed Test ==========")

    for dtype in ['float32']:
        print(f"test {dtype}...")

        data_np = np.random.randn(100000, 128).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            stmt = "func(data)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            output = func(data)
            head = torch.ones_like(output)
            stmt = "output.backward(head, retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            return t_forward, t_backward

        forward_ref, backward_ref = test_implementation(F.relu)
        forward_us, backward_us = test_implementation(ext_quantization.act_quantized_relu)

        print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Quantized. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))

def test_dropout_speed():
    print("========== Dropout Speed Test ==========")
    p = 0.5
    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.randn(100000, 128).astype(dtype)

        def test_implementation(func, p):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            stmt = "func(data, p, True)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            output = func(data, p, True)
            head = torch.ones_like(output)
            stmt = "output.backward(head, retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            return t_forward, t_backward

        forward_ref, backward_ref = test_implementation(F.dropout, p)
        forward_us, backward_us = test_implementation(ext_quantization.act_quantized_dropout, p)

        print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Quantized. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))

def test_linear_speed():
    print("========== Linear Speed Test ==========")

    for dtype in ['float32']:
        print(f"test {dtype}...")

        data_np = np.random.randn(100000, 128).astype(dtype)
        w_np = np.random.randn(128, 128).astype(dtype)
        b_np = np.random.randn(128).astype(dtype)
        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()
            weight = torch.tensor(w_np).to("cuda").requires_grad_()
            bias = torch.tensor(b_np).to("cuda").requires_grad_()
            stmt = "func(data, weight, bias)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            output = func(data, weight, bias)
            head = torch.ones_like(output)
            stmt = "output.backward(head, retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            return t_forward, t_backward

        forward_ref, backward_ref = test_implementation(F.linear)
        forward_us, backward_us = test_implementation(qlinear.apply)

        print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Quantized. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


def test_leaky_relu_correctness():
    print("========== LeakyReLU Correctness Test ==========")

    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.randn(10000, 128).astype(dtype)

        def test_implementation(func, neg_slope):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            output = func(data, neg_slope)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad]]

        output_ref, grad_data_ref = test_implementation(F.leaky_relu, 0.2)
        output_us, grad_data_us = test_implementation(ext_quantization.act_quantized_leaky_relu, 0.2)

        np.testing.assert_allclose(output_ref, output_us)
        np.testing.assert_allclose(grad_data_ref, grad_data_us)


def test_leaky_relu_memory():
    print("========== LeakyReLU Memory Test ==========")

    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.randn(100000, 128).astype(dtype)

        def test_implementation(func, neg_slope):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            before = get_memory_usage(0)

            for i in range(10):
                data = func(data, neg_slope)

            after = get_memory_usage(0)

            return after - before

        usage_ref = test_implementation(F.leaky_relu, 0.2)
        usage_us = test_implementation(ext_quantization.act_quantized_leaky_relu, 0.2)

        print("Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
        print("Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))


def test_leaky_relu_speed():
    print("========== LeakyReLU Speed Test ==========")

    for dtype in ['float32']:
        print(f"test {dtype}...")

        data_np = np.random.randn(10000, 128).astype(dtype)

        def test_implementation(func, neg_slope):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            stmt = "func(data, neg_slope)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            output = func(data, neg_slope)
            head = torch.ones_like(output)
            stmt = "output.backward(head, retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            return t_forward, t_backward

        forward_ref, backward_ref = test_implementation(F.leaky_relu, 0.2)
        forward_us, backward_us = test_implementation(ext_quantization.act_quantized_leaky_relu, 0.2)

        print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Quantized. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


def test_elu_correctness():
    print("========== ELU Correctness Test ==========")

    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.randn(100, 128).astype(dtype)
        alpha = 0.2
        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            output = func(data, alpha)
            output.backward(torch.ones_like(output))
            return [x.detach().cpu().numpy() for x in [output, data.grad]]

        output_ref, grad_data_ref = test_implementation(F.elu)
        outputs, grads = [], []
        for _ in range(10):
            output_us, grad_data_us = test_implementation(qelu.apply)
            np.testing.assert_allclose(output_ref, output_us)
            grads.append(grad_data_us)
        grads = np.stack(grads, 0)
        grad_mean = grads.mean(0)
        grad_std = grads.std(0)
        bias = np.linalg.norm(grad_mean - grad_data_ref)
        print('Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(grad_data_ref), bias, np.linalg.norm(grad_std)))


def test_elu_memory():
    print("========== ELU Memory Test ==========")
    alpha = 0.0
    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.randn(10000, 128).astype(dtype)
        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()
            output = data
            before = get_memory_usage(0)

            for i in range(10):
                output = func(output, alpha)
            after = get_memory_usage(0) - compute_tensor_bytes(output)
            if func == F.elu:
                after += compute_tensor_bytes(data)
            return after - before

        usage_ref = test_implementation(F.elu)
        usage_us = test_implementation(qelu.apply)
        print("Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
        print("Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))
        print("Ratio: %.2f" % (usage_ref / usage_us))


def test_linear_correctness():
    print("========== Linear Correctness Test ==========")
    data_np = np.random.randn(100000, 128).astype('float32')
    data = torch.tensor(data_np).to("cuda")
    # data = arxiv_data.x.continguous()
    ce = torch.nn.CrossEntropyLoss().cuda()
    y = torch.empty(100000, dtype=torch.long).random_(4).cuda()

    def test_implementation(func, weight, bias):
        pred = func(data, weight, bias)
        pred = F.relu(pred)
        pred = pred.reshape(pred.shape[0], 4, pred.shape[1] // 4).mean(2)
        loss = ce(pred, y)
        weight.grad = None
        bias.grad = None
        loss.backward()
        return weight.grad.cpu().numpy()

    w = torch.randn((128, 128), requires_grad=True, device='cuda')
    b = torch.randn((128,), requires_grad=True, device='cuda')
    qw = torch.randn((128, 128), requires_grad=True, device='cuda')
    qb = torch.randn((128,), requires_grad=True, device='cuda')
    with torch.no_grad():
        qw.copy_(w)
        qb.copy_(b)
    true_grad = test_implementation(F.linear, w, b)
    grads = []
    for i in range(10):
        grads.append(test_implementation(qlinear.apply, qw, qb))
    grads = np.stack(grads, 0)
    grad_mean = grads.mean(0)
    grad_std = grads.std(0)
    bias = np.linalg.norm(grad_mean - true_grad)
    print('Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(true_grad), bias, np.linalg.norm(grad_std)))


def test_linear_memory():
    print("========== Linear Memory Test ==========")
    data_np = np.random.randn(169343, 128).astype('float32')
    w = torch.randn((128, 128), requires_grad=True, device='cuda')
    b = torch.randn((128,), requires_grad=True, device='cuda')
    qw = torch.randn((128, 128), requires_grad=True, device='cuda')
    qb = torch.randn((128,), requires_grad=True, device='cuda')
    with torch.no_grad():
        qw.copy_(w)
        qb.copy_(b)

    def test_implementation(func, weight, bias, n_layers):
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        output = data

        before = get_memory_usage(0)

        for i in range(n_layers):
            output = func(output, weight, bias)

        after = get_memory_usage(0) - compute_tensor_bytes([output])
        if func == F.linear:
            after += compute_tensor_bytes([data])

        return after - before

    usage_ref = test_implementation(F.linear, w, b, 5)
    usage_us = test_implementation(qlinear.apply, qw, qb, 5)
    print("5 layer: Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("5 layer: Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))
    print("5 layer: Ratio: %.2f" % (usage_ref / usage_us))
    print("")

    usage_ref = test_implementation(F.linear, w, b, 10)
    usage_us = test_implementation(qlinear.apply, qw, qb, 10)
    print("10 layer: Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("10 layer: Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))
    print("10 layer: Ratio: %.2f" % (usage_ref / usage_us))
    print("")


def test_bn_correctness():
    # arguments and test data
    N, CI = 160000, 128
    data_np = np.random.randn(N, CI).astype('float32') * 0.01
    running_mean_np = np.random.randn(CI).astype('float32')
    running_var_np = np.random.randn(CI).astype('float32')
    bn_weight_np = np.random.randn(CI).astype('float32')
    bn_bias_np = np.random.randn(CI).astype('float32')
    training = True

    bn_scheme = None
    config.compress_activation = False

    def test_implementation(func):
        torch.manual_seed(0)
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        running_mean = torch.tensor(running_mean_np).to("cuda")
        running_var = torch.tensor(running_var_np).to("cuda")
        bn_weight = torch.tensor(bn_weight_np).to("cuda").requires_grad_()
        bn_bias = torch.tensor(bn_bias_np).to("cuda").requires_grad_()

        if func == F.batch_norm:
            output = func(data, running_mean, running_var, bn_weight, bn_bias, training, 0.1, 1e-5)
        else:
            output = func(data, running_mean, running_var, bn_weight, bn_bias, training, 0.1, 1e-5, bn_scheme)

        output.backward(torch.ones_like(output))

        return [x.detach().cpu().numpy() for x in [output, data.grad, bn_weight.grad, bn_bias.grad]]

    output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(F.batch_norm)
    output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(qbatch_norm.apply)

    atol = 1e-3
    rtol = 1e-3
    print("========== BN Correctness Test ==========")
    np.testing.assert_allclose(output_ref, output_us, atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol, rtol=rtol)


def test_bn_speed():
    print("========== Batch norm Speed Test ==========")
    N, CI = 100000, 128
    data_np = np.random.randn(N, CI).astype('float32') * 0.01
    running_mean_np = np.random.randn(CI).astype('float32')
    running_var_np = np.random.randn(CI).astype('float32')
    bn_weight_np = np.random.randn(CI).astype('float32')
    bn_bias_np = np.random.randn(CI).astype('float32')
    config.compress_activation = False

    def test_implementation(func):
        # torch.manual_seed(0)
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        running_mean = torch.tensor(running_mean_np).to("cuda")
        running_var = torch.tensor(running_var_np).to("cuda")
        bn_weight = torch.tensor(bn_weight_np).to("cuda").requires_grad_()
        bn_bias = torch.tensor(bn_bias_np).to("cuda").requires_grad_()
        training = True
        a, b = 0.1, 1e-5
        bn_scheme = None
        if func == F.batch_norm:
            stmt = "func(data, running_mean, running_var, bn_weight, bn_bias, training, a, b)"
        else:
            stmt = "func(data, running_mean, running_var, bn_weight, bn_bias, training, a, b, bn_scheme)"
        t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                 setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

        if func == F.batch_norm:
            output = func(data, running_mean, running_var, bn_weight, bn_bias, training, a, b)
        else:
            output = func(data, running_mean, running_var, bn_weight, bn_bias, training, a, b, None)
        tmp = torch.ones_like(output)
        stmt = "output.backward(tmp, retain_graph=True)"
        t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                  setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

        return t_forward, t_backward

    forward_ref, backward_ref = test_implementation(F.batch_norm)
    forward_us, backward_us = test_implementation(qbatch_norm.apply)

    print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
          (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
    print("Quantized. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
          (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


def test_dropout_correctness():
    print("========== Dropout Correctness Test ==========")
    data_np = np.ones((232965, 602)).astype('float32')
    data = torch.tensor(data_np).to("cuda").requires_grad_()
    p = 0.5
    for training in [True]:
        output_ref = F.dropout(data, p, training)
        output_ours = ext_quantization.act_quantized_dropout(data, p, training)
        output_ref.backward(torch.ones_like(output_ref))
        grad_ref = copy.deepcopy(data.grad.detach().cpu().numpy())
        data.grad = None
        output_ours.backward(torch.ones_like(output_ours))
        grad_ours = data.grad.detach().cpu().numpy()
        tmp1 = (output_ours == 0.0).type(torch.float).cpu().numpy()
        tmp2 = (grad_ours==0.0).astype(float)
        assert np.sum(tmp1 - tmp2) == 0.0 
        assert torch.max(output_ref) == torch.max(output_ours)
        assert np.max(grad_ref) == np.max(grad_ours) 
    

def test_dropout_memory():
    print("========== Dropout Memory Test ==========")
    data_np = np.random.randn(1024, 128).astype('float32')

    def test_implementation(func, p, training):
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        before = get_memory_usage(0)

        for i in range(10):
            data = func(data, p, training)

        after = get_memory_usage(0)

        return after - before
    p = 0.2
    training = True
    usage_ref = test_implementation(F.dropout, p, training)
    usage_us = test_implementation(ext_quantization.act_quantized_dropout, p, training)

    print("Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))


def test_spmm_matmul_correctness():
    print('============test spmm correctness============')
    N, D = 10000, 128
    nnz = 50000
    i = torch.randint(high=N, size=(2, nnz), dtype=torch.int64)
    v = torch.randn(size=(nnz,))
    dense_mat_cpu = torch.randn(N, D)
    def test_implementation_has_value(func, reduce):
        v_ = v.clone().requires_grad_()
        torch_sp = torch.sparse_coo_tensor(i, v_, [N, N])
        tsp = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(torch_sp).cuda()
        dense_mat = dense_mat_cpu.cuda().requires_grad_()
        output = func(tsp, dense_mat, reduce)
        output.backward(torch.ones_like(output))
        return [x.detach().cpu().numpy() for x in [output, dense_mat.grad, v_.grad]]

    def test_implementation_non_value(func, reduce):
        tsp = torch_sparse.SparseTensor.from_edge_index(i, sparse_sizes=[N, N]).cuda()
        dense_mat = dense_mat_cpu.cuda().requires_grad_()
        output = func(tsp, dense_mat, reduce)
        output.backward(torch.ones_like(output))
        return [x.detach().cpu().numpy() for x in [output, dense_mat.grad]]

    for reduce in ['sum', 'mean', 'max', 'min']:
        print(f'============test spmm {reduce} correctness============')
        output_ref, grad_data_ref, grad_value_ref = test_implementation_has_value(torch_sparse.matmul, reduce)
        output_ref_nonvalue, grad_data_ref_nonvalue  = test_implementation_non_value(torch_sparse.matmul, reduce)
        value_grads, data_grads = [], []
        for _ in range(10):
            output_us, grad_data_us, grad_value_us = test_implementation_has_value(qmatmul, reduce)
            np.testing.assert_allclose(output_ref, output_us)
            value_grads.append(grad_value_us)
            data_grads.append(grad_data_us)
        value_grads = np.stack(value_grads, 0)
        value_grads_mean = value_grads.mean(0)
        value_grads_std = value_grads.std(0)
        bias = np.linalg.norm(value_grads_mean - grad_value_ref)
        print('Value Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(grad_value_ref), bias, np.linalg.norm(value_grads_std)))

        data_grads = np.stack(data_grads, 0)
        data_grads_mean = data_grads.mean(0)
        data_grads_std = data_grads.std(0)
        bias = np.linalg.norm(data_grads_mean - grad_data_ref)
        print('Data Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(grad_data_ref), bias, np.linalg.norm(data_grads_std)))

        data_grads = []
        for _ in range(10):
            output_us, grad_data_us = test_implementation_non_value(qmatmul, reduce)
            np.testing.assert_allclose(output_ref_nonvalue, output_us)
            data_grads.append(grad_data_us)

        data_grads = np.stack(data_grads, 0)
        data_grads_mean = data_grads.mean(0)
        data_grads_std = data_grads.std(0)
        bias = np.linalg.norm(data_grads_mean - grad_data_ref_nonvalue)
        print('(value.requires_grad = False) Data Grad = {}, Bias = {}, Std = {}'.format(
            np.linalg.norm(grad_data_ref_nonvalue), bias, np.linalg.norm(data_grads_std)))


def test_spmm_matmul_memory():
    print("========== Spmm Memory Test ==========")
    N, D = 10000, 128
    nnz = 50000
    data_np = np.random.randn(N, D).astype('float32')
    # w = torch.randn((D, D), requires_grad=True, device='cuda')
    # qw = torch.randn((D, D), requires_grad=True, device='cuda')
    # with torch.no_grad():
    #     qw.copy_(w)
    index = torch.randint(high=N, size=(2, nnz), dtype=torch.int64)
    value = torch.randn(size=(nnz,))
    torch_sp = torch.sparse_coo_tensor(index, value, [N, N])

    def test_implementation(func, n_layers, reduce):
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        tsp = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(torch_sp).cuda()
        tsp.fill_cache_()
        output = data
        before = get_memory_usage(0)

        for i in range(n_layers):
            output = func(tsp, output, reduce)

        after = get_memory_usage(0) - compute_tensor_bytes([output])
        
        if func == torch_sparse.matmul:
            after += compute_tensor_bytes(data)
        return after - before
    
    for reduce in ['sum', 'mean', 'max', 'min']:
        print(f'============test spmm {reduce} Memory============')    
        usage_ref = test_implementation(torch_sparse.matmul, 10, reduce)
        usage_us = test_implementation(qmatmul, 10, reduce)
        print("Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
        print("Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))
        print("Ratio: %.2f" % (usage_ref / usage_us))
        print("")


if __name__ == "__main__":
    # test_relu_correctness()
    # test_relu_memory()
    # test_relu_speed()
    # test_leaky_relu_correctness()
    # test_leaky_relu_memory()
    # test_leaky_relu_speed()
    # test_elu_correctness()
    # test_elu_memory()
    # test_linear_correctness()
    # test_linear_memory()
    # test_linear_speed()
    # print(torch.backends.cudnn.version())
    # test_bn_correctness()
    # test_bn_speed()
    # test_dropout_correctness()
    # test_dropout_memory()
    # test_dropout_speed()
    test_spmm_matmul_correctness()
    # test_spmm_matmul_memory()