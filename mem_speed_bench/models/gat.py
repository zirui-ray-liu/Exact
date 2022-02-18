from typing import Optional
from exact.gatconv import CustomGATConv

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_sparse import SparseTensor



class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, num_heads: int, 
                 dropout: float = 0.0, input_drop: float = 0.0, 
                 attn_drop: float = 0.0, edge_drop: float = 0.0,
                 use_attn_dst: bool = True, residual: bool = False,
                 batch_norm: bool = False,):
        super(GAT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = torch.nn.Dropout(p=dropout)
        self.input_dropout = torch.nn.Dropout(p=input_drop)
        self.activation = torch.nn.ReLU()
        self.batch_norm = batch_norm
        self.residual = residual
        self.use_attn_dst = use_attn_dst
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.edge_drop = edge_drop
        self.num_layers = num_layers

        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = hidden_channels * num_heads
            out_dim = hidden_channels
            output_heads = num_heads
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
                output_heads = 1
            conv = CustomGATConv(in_dim, out_dim, output_heads, dropout=attn_drop, 
                                 add_self_loops=False, residual=residual, use_attn_dst=use_attn_dst)
            self.convs.append(conv)

        if self.batch_norm:
            self.bns = ModuleList()
            for i in range(num_layers - 1):
                bn = BatchNorm1d(hidden_channels * num_heads)
                self.bns.append(bn)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()


    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        x = self.input_dropout(x)
        if self.training and self.edge_drop > 0:
            nnz = adj_t.nnz()
            bound = int(nnz * self.edge_drop)
            perm = torch.randperm(nnz).cuda()
            eids = perm[bound:]
            row, col, value = adj_t.storage._row, adj_t.storage._col, adj_t.storage._value
            row, col, value = row[eids], col[eids], value[eids]
            adj_t = SparseTensor(row=row, col=col, value=value, 
                                    sparse_sizes=(adj_t.size(0), adj_t.size(1)), is_sorted=False)
        for idx, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.batch_norm:
                x = self.bns[idx](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.convs[-1](x, adj_t)
        return x


    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t):
        x = self.convs[layer](x, adj_t)
        if layer < self.num_layers - 1:
            if self.batch_norm:
                x = self.bns[layer](x)
            x = self.activation(x)
        return x