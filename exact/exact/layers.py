from torch._C import qscheme
from torch_geometric.nn.conv import GCNConv, SAGEConv, GCN2Conv, GraphConv, GENConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, Optional, Union, OptPairTensor, Size
from torch_geometric.utils import softmax
from torch_sparse import SparseTensor

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Sequential
from torch_scatter import scatter, scatter_softmax

from exact.ops import rp2input, input2rp
import exact.cpp_extension.quantization as ext_quantization
from .utils import compute_tensor_bytes, get_memory_usage
from .ops import qlinear, qbatch_norm, qelu, quantize_activation
from .conf import config
from .qsheme import QScheme
from .spmm import qmatmul, QMatmul

QGCNConv_layer_ct = 0
total_act_mem = 0
GPU = 0


def qaddmm(input, mat1, mat2, beta=1., alpha=1):
    return beta * input + alpha * qlinear.apply(mat1, mat2, None)



class QMLP(Sequential):
    def __init__(self, channels, norm,
                 bias=True, dropout=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(QLinear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(QBatchNorm1d(channels[i], affine=True))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(QReLU())
                m.append(QDropout(dropout))
        super(QMLP, self).__init__(*m)


class QGENConv(GENConv):
    def __init__(self, *args, num_layers=2, norm='batch', **kwargs):
        super().__init__(*args, norm=norm, num_layers=num_layers, **kwargs)
        channels = [self.in_channels]
        for _ in range(num_layers - 1):
            channels.append(self.in_channels * 2)
        channels.append(self.out_channels)
        self.mlp = QMLP(channels, norm)
        self.reset_parameters()


    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        msg = x_j if edge_attr is None else x_j + edge_attr
        return ext_quantization.act_quantized_relu(msg) + self.eps


class QGCN2Conv(GCN2Conv):
    def __init__(self, *args, **kwargs):
        super(QGCN2Conv, self).__init__(*args, **kwargs)


    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        x.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0[:x.size(0)]

        if self.weight2 is None:
            out = x.add_(x_0)
            out = qaddmm(out, out, self.weight1, beta=1. - self.beta,
                               alpha=self.beta)
        else:
            out = qaddmm(x, x, self.weight1, beta=1. - self.beta,
                               alpha=self.beta)
            out += qaddmm(x_0, x_0, self.weight2, beta=1. - self.beta,
                                alpha=self.beta)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
            return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return qmatmul(adj_t, x, reduce=self.aggr)


class QSAGEConv(SAGEConv):
    def __init__(self, *args, **kwargs):
        super(QSAGEConv, self).__init__(*args, **kwargs)
        in_channels = self.in_channels
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        # reset the lin_l and lin_l from nn.Linear to Qlinear
        bias = self.lin_l.bias is not None
        self.lin_l = QLinear(in_channels[0], self.out_channels, bias=bias, rp=True)
        if self.root_weight:
            self.lin_r = QLinear(in_channels[1], self.out_channels, bias=False, rp=True)
        self.msg_and_aggr_func = QMatmul(self.aggr)
        if config.single_precision:
            self.scheme = QScheme('sage')
        else:
            self.scheme = None
        self.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
            l_eq_r = True
        else:
            l_eq_r = False
        
        # To avoid wasting memory, 
        # if x is a tensor, we quantize it only once and pass it to the qmatmul and lin_r.
        if l_eq_r:
            if config.kept_frac < 1.0:
                kept_acts = int(config.kept_frac * x[0].shape[1] + 0.999)
                dim_reduced_input, randmat = input2rp(x[0], kept_acts)
            else:
                dim_reduced_input, randmat = x[0], None
            quantized = quantize_activation(dim_reduced_input, self.scheme)
        else:
            quantized, randmat = None, None
        # quantized = None
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, quantized=quantized, randmat=randmat)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r, quantized, randmat)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message_and_aggregate(self, adj_t, x, quantized=None, randmat=None) -> Tensor:
        # adj_t = adj_t.set_value(None, layout=None)
        return self.msg_and_aggr_func(adj_t, x[0], quantized=quantized, randmat=randmat)


class QGraphConv(GraphConv):
    def __init__(self, *args, **kwargs):
        super(QGraphConv, self).__init__(*args, **kwargs)
        in_channels = self.in_channels
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        bias = self.lin_l.bias is not None
        self.lin_l = QLinear(in_channels[0], self.out_channels, bias=bias, rp=True)
        self.lin_r = QLinear(in_channels[1], self.out_channels, bias=False, rp=True)
        self.msg_and_aggr_func = QMatmul(self.aggr)
        self.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)
        return out


    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return self.msg_and_aggr_func(adj_t, x[0])


class QGCNConv(GCNConv):
    def __init__(self, *args, **kwargs):
        super(QGCNConv, self).__init__(*args, **kwargs)
        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels))
        self.msg_and_aggr_func = QMatmul(self.aggr)
        if config.single_precision:
            self.lin_scheme = QScheme('linear')
        else:
            self.lin_scheme = None
        self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        if self.training:
            x = qlinear.apply(x, self.weight, None, None, self.lin_scheme)
        else:
            x = F.linear(x, self.weight, None)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias
        return out

    def message_and_aggregate(self, adj_t, x):
        return self.msg_and_aggr_func(adj_t, x)


class QLinear(torch.nn.Linear):
    def __init__(self, input_features, output_features, bias=True, rp=True):
        super(QLinear, self).__init__(input_features, output_features, bias)
        if config.single_precision:
            self.scheme = QScheme('linear')
        else:
            self.scheme = None
        self.rp = rp

    def forward(self, input, quantized=None, randmat=None):
        if self.training:
            return qlinear.apply(input, self.weight, self.bias, quantized, randmat, self.scheme, self.rp)
        else:
            return super(QLinear, self).forward(input)


class QReLU(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return ext_quantization.act_quantized_relu(input)
        else:
            return F.relu(input)


class QLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ext_quantization.act_quantized_leaky_relu(input, self.negative_slope)


class QELU(torch.nn.Module):
    def __init__(self, alpha, inplace=False):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        if self.training:
            return qelu.apply(input, self.alpha)
        else:
            return F.elu(input, self.alpha)


class QDropout(torch.nn.Module):
    def __init__(self, p):
        self.p = p
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return ext_quantization.act_quantized_dropout(input, self.p, self.training)
        else:
            return F.dropout(input, self.p, self.training)


class QDropout2(torch.nn.Module):
    def __init__(self, p):
        self.p = p
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return ext_quantization.act_quantized_dropout(input, self.p, self.training) * (1 - self.p)
        else:
            return F.dropout(input, self.p, self.training) * (1 - self.p)


class QBatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(QBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if config.single_precision:
            self.scheme = QScheme('batchnorm')
        else:
            self.scheme = None

    def forward(self, input):
        if not self.training:
            return super(QBatchNorm1d, self).forward(input)

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return qbatch_norm.apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps, self.scheme)
