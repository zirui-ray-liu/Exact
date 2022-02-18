from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import ModuleList, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv



class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 drop_input: bool = False, batch_norm: bool = False, residual: bool = False):
        super(GCN, self).__init__()

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
        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
            conv = GCNConv(in_dim, out_dim, normalize=False)
            self.convs.append(conv)

        if self.batch_norm:
            self.bns = ModuleList()
            for i in range(num_layers - 1):
                bn = BatchNorm1d(hidden_channels)
                self.bns.append(bn)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()


    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        if self.drop_input:
            x = self.input_dropout(x)
        for idx, conv in enumerate(self.convs[:-1]):
            h = conv(x, adj_t)
            if self.batch_norm:
                h = self.bns[idx](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = self.activation(h)
            x = self.dropout(x)

        x = self.convs[-1](x, adj_t)
        return x


    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t):
        if layer != 0:
            if self.drop_input:
                x = self.input_dropout(x)
            x = self.dropout(x)
        h = self.convs[layer](x, adj_t)
        if layer < self.num_layers - 1:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = self.activation(h)
        return h


    @torch.no_grad()
    def mini_inference(self, x_all, loader):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')
        for i in range(len(self.convs)):
            xs = []
            for batch_size, n_id, adj in loader:
                edge_index, _, size = adj.to('cuda')
                x = x_all[n_id].to('cuda')
                xs.append(self.forward_layer(i, x, edge_index).cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all    