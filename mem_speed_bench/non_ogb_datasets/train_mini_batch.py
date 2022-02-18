import sys
import os

from torch.autograd import grad
sys.path.append(os.getcwd())
import argparse
import random
import time
import warnings
from torch_geometric.data.cluster import ClusterLoader
import yaml
import pdb

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.utils import subgraph
from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler, ClusterData
from torch_geometric.utils import degree

from exact import get_memory_usage, compute_tensor_bytes, exp_recorder, config, QModule
import models
from data import get_data
from logger import Logger
from sklearn.metrics import f1_score
import torch_geometric.transforms as T

MB = 1024**2
GB = 1024**3

SAMPLER = {'graph_saint': GraphSAINTRandomWalkSampler, 'cluster_gcn': ClusterData}

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, required=True, 
                    help='the path to the configuration file')
parser.add_argument('--dataset', type=str, required=True, 
                    help='the name of the applied dataset')
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--n_bits', type=int, default=8)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--act_fp', action='store_true')
parser.add_argument('--kept_frac', type=float, default=1.0)
parser.add_argument('--grad_norm', type=float, default=None)
parser.add_argument('--inductive', action='store_true')
parser.add_argument('--test_speed', action='store_true')
parser.add_argument('--debug_mem', action='store_true')



def get_optimizer(model_config, model):
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
    else:
        raise NotImplementedError
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
    else:
        raise NotImplementedError
    return optimizer


def to_inductive(data):
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def train(model, loader, optimizer, loss_op, grad_norm):
    model.train()
    total_loss = 0
    for data in loader:
        # s_time = time.time()
        data = T.ToSparseTensor()(data.to('cuda'))
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)
        loss = loss_op(out[data.train_mask], data.y[data.train_mask])
        del data
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        total_loss += loss.item()
        # print(f'used time: {time.time() - s_time}')
    return total_loss / len(loader)


def compute_micro_f1(logits, y, mask=None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
        
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.adj_t)
    y_true = data.y
    train_acc = compute_micro_f1(out, y_true, data.train_mask)
    valid_acc = compute_micro_f1(out, y_true, data.val_mask)
    test_acc = compute_micro_f1(out, y_true, data.test_mask)
    return train_acc, valid_acc, test_acc


def main():
    global args 
    args = parser.parse_args()
    with open(args.conf, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
        name = model_config['name']
        if args.dataset == 'reddit2':
            model_config = model_config['params']['reddit']
        else:
            model_config = model_config['params'][args.dataset]
        model_config['name'] = name
    print(f'model config: {model_config}')
    if args.dataset == 'yelp':
        multi_label = True
    else:
        multi_label = False
    print(f'clipping grad norm: {args.grad_norm}')
    args.model = model_config['arch_name']
    assert model_config['name'] in ['graph_saint', 'cluster_gcn']
    if args.act_fp and args.kept_frac == 1.0:
        use_qmodule = False
        print('=' * 30 + 'Do not compress activation' + '=' * 30)
        config.compress_activation = False
    else:
        use_qmodule = True
        config.compress_activation = not args.act_fp
        config.activation_compression_bits = [args.n_bits]
        precision = 'FP32' if not config.compress_activation\
                     else f'INT{config.activation_compression_bits[0]}'
        print('=' * 30 + f'Store activation in {precision}' + '=' * 30)
    print(f'use qmodule: {use_qmodule}')
    if args.kept_frac < 1.0:
        config.kept_frac = args.kept_frac
    print(f'kept frac: {args.kept_frac}')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print("Use GPU {} for training".format(args.gpu))

    torch.cuda.set_device(args.gpu)
    data, num_features, num_classes = get_data(args.root, args.dataset)
    sampler_data = data
    print('converting data form...')
    s_time = time.time()
    data = T.ToSparseTensor()(data.clone())
    data = data.to('cuda')
    print(f'done. used {time.time() - s_time} sec')

    if args.inductive:
        print('inductive learning mode')
        sampler_data = to_inductive(sampler_data)

    sampler_cls = SAMPLER[model_config['name']]
    s_time = time.time()
    print("=" * 30 + f'Prepare {sampler_cls.__name__} for training' + '=' * 30)
    if model_config['name'] == 'graph_saint':
        num_steps = int(sampler_data.num_nodes / model_config['sampler']['batch_size'] / model_config['sampler']['walk_length'])
        print(f'num steps: {num_steps}')
        loader = sampler_cls(sampler_data,
                            num_workers=args.num_workers,
                            num_steps=num_steps,
                            **model_config['sampler'])

    elif model_config['name'] == 'cluster_gcn':
        batch_size = model_config['sampler']['batch_size']
        del model_config['sampler']['batch_size']
        cluster_data = sampler_cls(sampler_data, 
                                   **model_config['sampler'])
        loader = ClusterLoader(cluster_data, batch_size=batch_size, 
                               shuffle=True, num_workers=args.num_workers)
    else:
        raise NotImplementedError
    print("=" * 30 + 
          f'Finished Building {sampler_cls.__name__}, used {time.time() - s_time} sec' + 
          '=' * 30)

    GNN = getattr(models, model_config['arch_name'])
    model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    if use_qmodule:
        print('convert the model')
        model = QModule(model)
    loss_op = F.binary_cross_entropy_with_logits if multi_label else F.cross_entropy
    print(model)
    model.cuda(args.gpu)

    if args.debug_mem:
        print("========== Model, optimizer, and init data===========")
        model.reset_parameters()
        model.train()
        optimizer = get_optimizer(model_config, model)
        optimizer.zero_grad()
        usage = get_memory_usage(args.gpu, True)
        exp_recorder.record("network", 'SAGE')
        exp_recorder.record("model_only", usage / MB, 2)    
        print("========== Load data to GPU ===========")
        for _data in loader:     
            optimizer.zero_grad()
            _data = T.ToSparseTensor()(_data.to('cuda'))
            _data.adj_t.fill_cache_()
            num_nodes, num_edges = _data.num_nodes, _data.num_edges
            print(f'sampled sub-graph: #nodes: {num_nodes}, #edges: {num_edges}')
            init_mem = get_memory_usage(args.gpu, True)
            data_mem = init_mem / MB - exp_recorder.val_dict['model_only']
            exp_recorder.record("data", init_mem / MB - exp_recorder.val_dict['model_only'], 2)
            out = model(_data.x, _data.adj_t)
            y = _data.y
            loss = loss_op(out[_data.train_mask], y[_data.train_mask])
            print("========== Before Backward ===========")
            before_backward = get_memory_usage(args.gpu, True)
            act_mem = get_memory_usage(args.gpu, False) - init_mem - compute_tensor_bytes([loss, out])
            res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (before_backward / MB,
                                                                            data_mem,
                                                                            act_mem / MB)
            print(res)
            loss.backward()
            optimizer.step()
            del loss
            print("========== After Backward ===========")
            after_backward = get_memory_usage(args.gpu, True)
            total_mem = before_backward + (after_backward - init_mem)
            res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (total_mem / MB,
                                                                            data_mem,
                                                                            act_mem / MB)
            print(res)
            exp_recorder.record("total", total_mem / MB, 2)
            exp_recorder.record("activation", act_mem / MB, 2)
            exit()

    logger = Logger(args.runs, args)
    eval_start_epoch = model_config['eval_start_epoch']
    eval_steps = model_config['eval_steps']
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        for epoch in range(1, 1 + model_config['epochs']):
            loss = train(model, loader, optimizer, loss_op, args.grad_norm)
            if model_config['log_steps'] > 0 and epoch % model_config['log_steps'] == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}')
        
            if epoch > eval_start_epoch and epoch % eval_steps == 0:
                result = test(model, data)
                logger.add_result(run, result)
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train f1: {100 * train_acc:.2f}%, '
                      f'Valid f1: {100 * valid_acc:.2f}% '
                      f'Test f1: {100 * test_acc:.2f}%')

        logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == '__main__':
    main()