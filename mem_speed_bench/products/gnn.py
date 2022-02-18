import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, GCNConv, GENConv
from torch.utils.checkpoint import checkpoint

from exact import config


class SharedModule(torch.nn.Module):

    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to('cuda')
                x = x_all[n_id].to('cuda')
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)
            del x_all
            x_all = torch.cat(xs, dim=0)
            del xs
        pbar.close()
        return x_all    

    
class GCN(SharedModule):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = torch.nn.ReLU()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = self.act(x)
            if not config.debug_remove_dropout:
                x = self.dropout(x)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


        
class SAGE(SharedModule):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, batch_norm=False):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        use_bn = batch_norm and not config.debug_remove_bn
        self.use_bn = use_bn
        if use_bn:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if use_bn:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = torch.nn.ReLU()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()


    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.act(x)
            if not config.debug_remove_dropout:
                x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1) 


class DeeperGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, mlp_layers, dropout=0.5, 
                 block='res+', gcn_aggr='softmax_sg', norm='batch', t=1., p=1., learn_t=False, learn_p=False,
                 msg_norm=False, learn_msg_scale=False):
        super(DeeperGCN, self).__init__()
        self.checkpoint_grad = False
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.block = block
        self.aggr = gcn_aggr
        self.num_layers = num_layers
        self.mlp_layers = mlp_layers
        self.t, self.p, self.learn_t, self.learn_p = t, p, learn_t, learn_p
        self.msg_norm, self.learn_msg_scale = msg_norm, learn_msg_scale
        self.norm = norm
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = torch.nn.ReLU()

        if self.aggr in ['softmax_sg', 'softmax', 'power'] and self.num_layers > 3:
            self.checkpoint_grad = True
            self.ckp_k = self.num_layers // 2

        if self.block == 'res+':
            print('BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->BN->ReLU->Res')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')
        if self.norm != 'batch':
            raise NotImplementedError()

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.node_features_encoder = torch.nn.Linear(in_channels, hidden_channels)
        self.node_pred_linear = torch.nn.Linear(hidden_channels, out_channels)

        for _ in range(self.num_layers):
            gcn = GENConv(hidden_channels, hidden_channels,
                            aggr=self.aggr,
                            t=self.t, learn_t=self.learn_t,
                            p=self.p, learn_p=self.learn_p,
                            msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                            norm=self.norm, num_layers=self.mlp_layers)

            self.gcns.append(gcn)
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
    
    def reset_parameters(self):
        for gcn in self.gcns:
            gcn.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        
    def forward(self,  x, edge_index):
        h = self.node_features_encoder(x)
        if self.block == 'res+':
            h = self.gcns[0](h, edge_index)
            if self.checkpoint_grad:

                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = self.act(h1)
                    h2 = self.dropout(h2)

                    if layer % self.ckp_k != 0:
                        res = checkpoint(self.gcns[layer], h2, edge_index)
                        h = res + h
                    else:
                        h = self.gcns[layer](h2, edge_index) + h
            else:
                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = self.act(h1)
                    h2 = self.dropout(h2)
                    h = self.gcns[layer](h2, edge_index) + h

            h = self.act(self.norms[self.num_layers - 1](h))
            h = self.dropout(h)

        elif self.block == 'res':

            h = self.act(self.norms[0](self.gcns[0](h, edge_index)))
            h = self.dropout(h)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = self.act(h2) + h
                h = self.dropout(h)

        elif self.block == 'plain':

            h = self.act(self.norms[0](self.gcns[0](h, edge_index)))
            h = self.dropout(h)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = self.act(h2)
                h = self.dropout(h)
        else:
            raise Exception('Unknown block Type')
        h = self.node_pred_linear(h)
        return torch.log_softmax(h, dim=-1)


    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * len(self.gcns))
        pbar.set_description('Evaluating')
        if self.block != 'res+':
            raise NotImplementedError
        for i, gcn in enumerate(self.gcns):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to('cuda')
                x = x_all[n_id].to('cuda')
                if i == 0:
                    x = self.node_features_encoder(x)
                    h = gcn(x, edge_index)
                else:
                    h = x
                    h1 = self.norms[i - 1](h)
                    h2 = self.act(h1)
                    h2 = self.dropout(h2)
                    h = gcn(h2, edge_index) + h
                    if i == self.num_layers - 1:
                        h = self.norms[self.num_layers - 1](h)
                        h = F.relu(h)
                        h = self.dropout(h)
                        h = self.node_pred_linear(h)
                xs.append(h.cpu())
                pbar.update(batch_size)
            del x_all
            x_all = torch.cat(xs, dim=0)
            del xs
        pbar.close()
        return x_all    