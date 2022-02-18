from exact.layers import QDropout, QLinear
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import zeros
import exact.cpp_extension.quantization as ext_quantization


class CustomGATConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, 
                 residual=True, use_attn_dst=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(CustomGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.residual = residual
        self.use_attn_dst = use_attn_dst

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        if residual:
            self.res_fc = Linear(in_channels, heads * out_channels, bias=False)
        else:
            self.register_buffer("res_fc", None)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        if self.use_attn_dst:
            self.att_r = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.att_r = self.att_l
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_normal_(self.lin_l.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.lin_r.weight, gain=gain)
        if isinstance(self.res_fc, Linear):
            torch.nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.att_l, gain=gain)
        if self.use_attn_dst:
            torch.nn.init.xavier_normal_(self.att_r, gain=gain)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None):
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if self.use_attn_dst:
                alpha_r = (x_r * self.att_r).sum(dim=-1)
            else:
                alpha_r = alpha_l
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)


        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.residual:
            out += self.res_fc(x)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={}, residual={}, use_attn_dst={})'.format(self.__class__.__name__,
                                                                           self.in_channels, self.out_channels, 
                                                                           self.heads, self.residual, 
                                                                           self.use_attn_dst)


class QCustomGATConv(CustomGATConv):
    def __init__(self, *args, **kwargs):
        super(QCustomGATConv, self).__init__(*args, **kwargs)
        if isinstance(self.in_channels, int):
            self.lin_l = QLinear(self.in_channels, self.heads * self.out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = QLinear(self.in_channels[0], self.heads * self.out_channels, False)
            self.lin_r = QLinear(self.in_channels[1], self.heads * self.out_channels, False)

        if self.residual:
            self.res_fc = Linear(self.in_channels, self.heads * self.out_channels, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.dropout_module = QDropout(p=self.dropout)

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = ext_quantization.act_quantized_leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = self.dropout_module(alpha)
        return x_j * alpha.unsqueeze(-1)