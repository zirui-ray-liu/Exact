from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict
from collections import OrderedDict
from torch_geometric.nn.conv.gen_conv import GENConv
from torch_geometric.nn.conv.graph_conv import GraphConv
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from torch_geometric.nn.conv import GCNConv, SAGEConv, GCN2Conv, SGConv
from .layers import QBatchNorm1d, QLinear, QReLU, QGCNConv, QDropout, QDropout2, QSAGEConv, QGENConv, QGraphConv, QGCN2Conv
from .conf import config
from .gatconv import CustomGATConv, QCustomGATConv


class QModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        QModule.convert_layers(model)

    @staticmethod
    def convert_layers(module):
        for name, child in module.named_children():
            # Do not convert layers that are already quantized
            if isinstance(child, (QBatchNorm1d, QLinear, QReLU, QGCNConv, QGCN2Conv, QDropout, QSAGEConv, QGENConv, QCustomGATConv)):
                continue
            if isinstance(child, nn.BatchNorm1d) and config.enable_quantized_bn:
                setattr(module, name, QBatchNorm1d(child.num_features, child.eps, child.momentum,
                    child.affine, child.track_running_stats))
            elif isinstance(child, nn.Linear):
                setattr(module, name, QLinear(child.in_features, child.out_features,
                    child.bias is not None))
            elif isinstance(child, nn.ReLU):
                setattr(module, name, QReLU())
            elif isinstance(child, nn.Dropout):
                if config.dropout2:
                    setattr(module, name, QDropout2(child.p))
                else:
                    setattr(module, name, QDropout(child.p))
            elif isinstance(child, GCNConv):
                setattr(module, name, QGCNConv(child.in_channels, child.out_channels, child.improved, child.cached,
                                               child.add_self_loops, child.normalize, child.bias is not None,
                                               aggr=child.aggr))
            elif isinstance(child, GCN2Conv):
                beta = child.beta
                shared_weights = child.weight2 is None
                setattr(module, name, QGCN2Conv(child.channels, alpha=child.alpha, theta=None, layer=None, shared_weights=shared_weights,
                                                cached=child.cached, add_self_loops=child.add_self_loops, normalize=child.normalize))
                curconv = getattr(module, name)
                curconv.beta = child.beta
            elif isinstance(child, SAGEConv):
                setattr(module, name, QSAGEConv(child.in_channels, child.out_channels, child.normalize, child.root_weight,
                                               child.lin_l.bias is not None))
            elif isinstance(child, GraphConv):
                setattr(module, name, QGraphConv(child.in_channels, child.out_channels, child.aggr, child.lin_l.bias is not None))
            elif isinstance(child, CustomGATConv):
                setattr(module, name, QCustomGATConv(child.in_channels, child.out_channels, child.heads, child.concat, 
                                                     child.negative_slope, child.dropout, child.add_self_loops, child.bias is not None, 
                                                     child.residual, child.use_attn_dst))
            elif isinstance(child, GENConv):
                msg_norm = child.msg_norm is not None
                learn_msg_scale = True if (msg_norm and child.msg_norm.scale.requires_grad) else False
                learn_p = isinstance(child.p, torch.nn.Parameter)
                is_softmax = child.aggr == 'softmax'
                if is_softmax and isinstance(child.t, torch.nn.Parameter):
                    learn_t = True
                else:
                    learn_t = False
                num_layers = 0
                norm = 'batch'
                for m in child.mlp:
                    if isinstance(m, torch.nn.Linear):
                        num_layers += 1
                    if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.InstanceNorm1d)):
                        if isinstance(m, nn.BatchNorm1d):
                            pass
                        elif isinstance(m, nn.LayerNorm):
                            norm = 'layer'
                        elif isinstance(m, nn.InstanceNorm1d):
                            norm = 'instance'

                setattr(module, name, QGENConv(child.in_channels, child.out_channels, child.aggr, child.initial_t, learn_t, child.initial_p, learn_p, 
                                               msg_norm, learn_msg_scale, norm, num_layers, child.eps))
            else:
                QModule.convert_layers(child)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                        strict: bool = True):
        # remove the prefix "model." added by this wrapper
        new_state_dict = OrderedDict([("model." + k,  v) for k, v in state_dict.items()])
        return super().load_state_dict(new_state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super().state_dict(destination, prefix, keep_vars)

        # remove the prefix "model." added by this wrapper
        ret = OrderedDict([(k[6:], v) for k, v in ret.items()])
        return ret

    def reset_parameters(self):
        self.model.reset_parameters()


    @torch.no_grad()
    def mini_inference(self, x_all, loader):
        return self.model.mini_inference(x_all, loader)