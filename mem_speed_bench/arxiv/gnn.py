import pdb
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, SAGEConv
from torch_sparse import SparseTensor

from exact import config, CustomGATConv



class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads,
                 dropout, input_drop, attn_drop, edge_drop, residual, use_attn_dst):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(CustomGATConv(in_channels, hidden_channels, num_heads, 
                                        dropout=attn_drop, add_self_loops=False, residual=residual, use_attn_dst=use_attn_dst))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * num_heads))
        for _ in range(num_layers -2):
            self.convs.append(
                CustomGATConv(hidden_channels * num_heads, hidden_channels, num_heads, dropout=attn_drop, add_self_loops=False, 
                              residual=residual, use_attn_dst=use_attn_dst))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * num_heads))
        self.convs.append(CustomGATConv(hidden_channels * num_heads, out_channels, 1, dropout=attn_drop, add_self_loops=False, 
                                        residual=residual, use_attn_dst=use_attn_dst))
        self.input_dropout = torch.nn.Dropout(p=input_drop)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = torch.nn.ReLU()
        self.edge_drop = edge_drop


    def forward(self, x, adj_t):
        x = self.input_dropout(x)
        for i, conv in enumerate(self.convs[:-1]):
            if self.training and self.edge_drop > 0:
                nnz = adj_t.nnz()
                bound = int(nnz * self.edge_drop)
                perm = torch.randperm(nnz).cuda()
                eids = perm[bound:]
                row, col, value = adj_t.storage._row, adj_t.storage._col, adj_t.storage._value
                row, col, value = row[eids], col[eids], value[eids]
                adj_t = SparseTensor(row=row, col=col, value=value, 
                                     sparse_sizes=(adj_t.size(0), adj_t.size(1)), is_sorted=False)
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.act(x)
            if not config.debug_remove_dropout:
                x = self.dropout(x)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if not config.debug_remove_bn:
            for bn in self.bns:
                bn.reset_parameters()


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, normalize=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True, normalize=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, normalize=False))
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = torch.nn.ReLU()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if not config.debug_remove_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.act(x)
            if not config.debug_remove_dropout:
                x = self.dropout(x)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = torch.nn.ReLU()


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if not config.debug_remove_bn:
            for bn in self.bns:
                bn.reset_parameters()


    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.act(x)
            if not config.debug_remove_dropout:
                x = self.dropout(x)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)