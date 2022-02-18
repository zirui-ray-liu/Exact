from tqdm import tqdm
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import ModuleList, BatchNorm1d, Linear
from torch_sparse import SparseTensor
from torch_geometric.nn import GCN2Conv



class GCN2(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float, theta: float = None, 
                 shared_weights: bool = True, dropout: float = 0.0, drop_input: bool = True,
                 batch_norm: bool = False, residual: bool = False):
        super(GCN2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = torch.nn.Dropout(p=dropout)
        self.drop_input = drop_input
        if drop_input:
            self.input_dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()
        self.batch_norm = batch_norm
        self.residual = residual
        self.num_layers = num_layers
        self.alpha, self.theta = alpha, theta

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            if theta is None:
                conv = GCN2Conv(hidden_channels, alpha=alpha, theta=None,
                                layer=None, shared_weights=shared_weights,
                                normalize=False, add_self_loops=False)
            else:
                conv = GCN2Conv(hidden_channels, alpha=alpha, theta=theta,
                                layer=i+1, shared_weights=shared_weights,
                                normalize=False, add_self_loops=False)
            self.convs.append(conv)

        if self.batch_norm:
            self.bns = ModuleList()
            for i in range(num_layers):
                bn = BatchNorm1d(hidden_channels)
                self.bns.append(bn)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()


    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        if self.drop_input:
            x = self.input_dropout(x)
        x = x0 = self.activation(self.lins[0](x))
        x = self.dropout(x)
        for idx, conv in enumerate(self.convs[:-1]):
            h = conv(x, x0, adj_t)
            if self.batch_norm:
                h = self.bns[idx](h)
            if self.residual:
                h += x[:h.size(0)]
            x = self.activation(h)
            x = self.dropout(x)

        h = self.convs[-1](x, x0, adj_t)
        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual:
            h += x[:h.size(0)]
        x = self.activation(h)
        x = self.dropout(x)
        x = self.lins[1](x)
        return x


    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        if layer == 0:
            if self.drop_input:
                x = self.input_dropout(x)
            x = x_0 = self.activation(self.lins[0](x))
            state['x_0'] = x_0[:adj_t.size(0)]
        x = self.dropout(x)
        h = self.convs[layer](x, state['x_0'], adj_t)
        if self.batch_norm:
            h = self.bns[layer](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        x = self.activation(h)
        if layer == self.num_layers - 1:
            x = self.dropout(x)
            x = self.lins[1](x)  
        return h


    @torch.no_grad()
    def mini_inference(self, x_all, loader):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')
        state = {}
        for i in range(len(self.convs)):
            xs = []
            for batch_size, n_id, adj in loader:
                edge_index, _, size = adj.to('cuda')
                x = x_all[n_id].to('cuda')
                xs.append(self.forward_layer(i, x, edge_index, state).cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all    