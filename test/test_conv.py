import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.norm import msg_norm

import torch_sparse
from torch_geometric.nn import GCNConv, SAGEConv, GCN2Conv, GENConv


from exact import config, QGCNConv, QSAGEConv, QGCN2Conv, QGENConv
from exact import get_memory_usage, compute_tensor_bytes


config.activation_compression_bits = [2]
num_samples = 10000
in_channels, out_channels = 100, 128
nnz = 50000
i = torch.randint(high=num_samples, size=(2, nnz), dtype=torch.long)
adj_t = torch_sparse.SparseTensor.from_edge_index(i, sparse_sizes=[num_samples, num_samples]).cuda()
adj_t.fill_cache_()


def test_gcnconv_correctness():
    layer = GCNConv(in_channels, out_channels).cuda()
    qlayer = QGCNConv(in_channels, out_channels).cuda()
    x = torch.randn((num_samples, in_channels)).cuda()
    y = torch.empty(num_samples, dtype=torch.long).random_(out_channels).cuda()
    print('==================test GCNConv correctness==================')
    with torch.no_grad():
        qlayer.weight.copy_(layer.weight.t())

    ce = nn.CrossEntropyLoss().cuda()

    def get_grad(model):
        pred = model(x, adj_t)
        pred = F.relu(pred)
        loss = ce(pred, y)
        model.weight.grad = None
        model.bias.grad = None
        loss.backward()
        return model.weight.grad.cpu().numpy()

    true_grad = get_grad(layer).T
    grads = []
    for i in range(10):
        grads.append(get_grad(qlayer))
    grads = np.stack(grads, 0)
    grad_mean = grads.mean(0)
    grad_std = grads.std(0)
    bias = np.linalg.norm(grad_mean - true_grad)
    print('Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(true_grad), bias, np.linalg.norm(grad_std)))


def test_gcnconv_memory():
    print("========== Gcnconv Memory Test ==========")
    adj_t_ = adj_t.set_diag()
    deg = adj_t_.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t_ = deg_inv_sqrt.view(-1, 1) * adj_t_ * deg_inv_sqrt.view(1, -1)

    layer = GCNConv(in_channels, in_channels, cached=False, normalize=False).cuda()
    qlayer = QGCNConv(in_channels, in_channels, cached=False, normalize=False).cuda()
    x = torch.randn((num_samples, in_channels))
    with torch.no_grad():
        qlayer.weight.copy_(layer.weight.t())


    def test_implementation(model, n_layers):
        data = x.to("cuda").requires_grad_()
        output = data
        before = get_memory_usage(0)
        for _ in range(n_layers):
            output = model(output, adj_t_)
        after = get_memory_usage(0) - compute_tensor_bytes([output])
        if model.__class__.__name__ == 'GCNConv':
            after += compute_tensor_bytes([data])
        return after - before

    usage_ref = test_implementation(layer, 5)
    usage_us = test_implementation(qlayer, 5)
    print("5 layer: Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("5 layer: Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))
    print("5 layer: Ratio: %.2f" % (usage_ref / usage_us))

    usage_ref = test_implementation(layer, 10)
    usage_us = test_implementation(qlayer, 10)
    print("10 layer: Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("10 layer: Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))
    print("10 layer: Ratio: %.2f" % (usage_ref / usage_us))



def test_sageconv_correctness():
    layer = SAGEConv(in_channels, out_channels, root_weight=True, bias=True).cuda()
    qlayer = QSAGEConv(in_channels, out_channels, root_weight=True, bias=True).cuda()
    x = torch.randn((num_samples, in_channels)).cuda()
    y = torch.empty(num_samples, dtype=torch.long).random_(out_channels).cuda()
    print('==================test SAGEConv correctness==================')
    with torch.no_grad():
        qlayer.lin_l.weight.copy_(layer.lin_l.weight)
        qlayer.lin_r.weight.copy_(layer.lin_r.weight)
        qlayer.lin_l.bias.copy_(layer.lin_l.bias)

    ce = nn.CrossEntropyLoss().cuda()

    def get_grad(model):
        pred = model(x, adj_t)
        pred = F.relu(pred)
        loss = ce(pred, y)
        model.lin_l.weight.grad = None
        model.lin_r.weight.grad = None
        loss.backward()
        return [lin.weight.grad.cpu().numpy() for lin in [model.lin_l, model.lin_r]]

    lin_l_true_grad, lin_r_true_grad = get_grad(layer)
    lin_l_grads, lin_r_grads = [], []
    for i in range(10):
        lin_l_grads.append(get_grad(qlayer)[0])
        lin_r_grads.append(get_grad(qlayer)[1])
    
    for idx, (true_grad, grads) in enumerate(zip([lin_l_true_grad, lin_r_true_grad], [lin_l_grads, lin_r_grads])):
        if idx == 0:
            prefix = 'lin l '
        else:
            prefix = 'lin r '
        grads = np.stack(grads, 0)
        grad_mean = grads.mean(0)
        grad_std = grads.std(0)
        bias = np.linalg.norm(grad_mean - true_grad)
        print(prefix + 'Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(true_grad), bias, np.linalg.norm(grad_std)))


def test_sageconv_memory():
    print("========== SAGEconv Memory Test ==========")
    root_weight = True
    layer = SAGEConv(in_channels, in_channels, root_weight=root_weight, bias=True).cuda()
    qlayer = QSAGEConv(in_channels, in_channels, root_weight=root_weight, bias=True).cuda()
    x = torch.randn((num_samples, in_channels))
    with torch.no_grad():
        qlayer.lin_l.weight.copy_(layer.lin_l.weight)
        qlayer.lin_l.bias.copy_(layer.lin_l.bias)
        if root_weight:
            qlayer.lin_r.weight.copy_(layer.lin_r.weight)


    def test_implementation(model, n_layers):
        data = x.to("cuda").requires_grad_()
        output = data
        before = get_memory_usage(0)
        for _ in range(n_layers):
            output = model(output, adj_t)
        after = get_memory_usage(0) - compute_tensor_bytes([output])
        if model.__class__.__name__ == 'SAGEConv':
            after += compute_tensor_bytes([data])
        return after - before

    usage_ref = test_implementation(layer, 5)
    usage_us = test_implementation(qlayer, 5)
    print("5 layer: Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("5 layer: Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))
    print("5 layer: Ratio: %.2f" % (usage_ref / usage_us))

    usage_ref = test_implementation(layer, 10)
    usage_us = test_implementation(qlayer, 10)
    print("10 layer: Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("10 layer: Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))
    print("10 layer: Ratio: %.2f" % (usage_ref / usage_us))


def test_genconv_correctness():
    layer = GENConv(in_channels, out_channels, num_layers=1).cuda()
    qlayer = QGENConv(in_channels, out_channels, num_layers=1).cuda()
    # pdb.set_trace()
    x = torch.randn((num_samples, in_channels)).cuda()
    y = torch.empty(num_samples, dtype=torch.long).random_(out_channels).cuda()
    print('==================test SAGEConv correctness==================')
    with torch.no_grad():
        for lin, qlin in zip(layer.mlp, qlayer.mlp):
            qlin.weight.copy_(lin.weight)
            qlin.bias.copy_(lin.bias)


    ce = nn.CrossEntropyLoss().cuda()

    def get_grad(model):
        pred = model(x, adj_t)
        pred = F.relu(pred)
        loss = ce(pred, y)
        for lin in model.mlp:
            lin.weight.grad = None
            lin.bias.grad = None
        loss.backward()
        return model.mlp[0].weight.grad.cpu().numpy()

    true_grad = get_grad(layer)
    grads = []
    for i in range(10):
        grads.append(get_grad(qlayer))
    grads = np.stack(grads, 0)
    grad_mean = grads.mean(0)
    grad_std = grads.std(0)
    bias = np.linalg.norm(grad_mean - true_grad)
    print('Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(true_grad), bias, np.linalg.norm(grad_std)))


def test_genconv_memory():
    print("========== GENConv Memory Test ==========")
    layer = GENConv(in_channels*5, in_channels*5, num_layers=3, aggr='softmax_sg').cuda()
    qlayer = QGENConv(in_channels*5, in_channels*5, num_layers=3, aggr='softmax_sg').cuda()
    x = torch.randn((num_samples, in_channels*5))
    with torch.no_grad():
        for lin, qlin in zip(layer.mlp, qlayer.mlp):
            if not isinstance(lin, torch.nn.Linear):
                continue
            qlin.weight.copy_(lin.weight)
            qlin.bias.copy_(lin.bias)
    def test_implementation(model, n_layers):
        data = x.to("cuda").requires_grad_()
        output = data
        before = get_memory_usage(0)
        for _ in range(n_layers):
            output = model(output, adj_t)
        after = get_memory_usage(0) - compute_tensor_bytes([output])
        if model.__class__.__name__ == 'SAGEConv':
            after += compute_tensor_bytes([data])
        return after - before


    usage_ref = test_implementation(layer, 5)
    usage_us = test_implementation(qlayer, 5)
    print("5 layer: Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("5 layer: Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))
    print("5 layer: Ratio: %.2f" % (usage_ref / usage_us))

    usage_ref = test_implementation(layer, 10)
    usage_us = test_implementation(qlayer, 10)
    print("10 layer: Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("10 layer: Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))
    print("10 layer: Ratio: %.2f" % (usage_ref / usage_us))

if __name__ == '__main__':
    test_gcnconv_correctness()
    test_gcnconv_memory()
    test_sageconv_correctness()
    test_sageconv_memory()
    test_genconv_correctness()
    test_genconv_memory()