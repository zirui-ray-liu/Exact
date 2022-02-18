import sys
import os
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

from exact import get_memory_usage, compute_tensor_bytes, exp_recorder, config, QModule, cast_adj
import models
from data import get_data
from logger import Logger

import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator

MB = 1024**2
GB = 1024**3

SAMPLER = {'graph_saint': GraphSAINTRandomWalkSampler, 'cluster_gcn': ClusterData}

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, required=True, 
                    help='the path to the configuration file')
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--debug_mem', action='store_true',
                    help='whether to debug the memory usage')
parser.add_argument('--n_bits', type=int, default=8)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--simulate', action='store_true')
parser.add_argument('--act_fp', action='store_true')
parser.add_argument('--kept_frac', type=float, default=1.0)
parser.add_argument('--amp', help='whether to enable apx mode', action='store_true')
parser.add_argument('--dropout2', help='whether to use dropout2', action='store_true')



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


def train(model, loader, optimizer, scaler, amp_mode, grad_norm):
    model.train()

    total_loss = 0
    for data in loader:
        data = T.ToSparseTensor()(data.to('cuda'))
        optimizer.zero_grad()
        with autocast(enabled=amp_mode):
            out = model(data.x, data.adj_t)
            y = data.y.squeeze(1)
            loss = F.cross_entropy(out[data.train_mask], y[data.train_mask])
        del data
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def test(model, data, evaluator, subgraph_loader, amp_mode):
    model.eval()
    with autocast(enabled=amp_mode):
        out = model.mini_inference(data.x, subgraph_loader)

    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.val_mask],
        'y_pred': y_pred[data.val_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    args = parser.parse_args()
    with open(args.conf, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
    args.model = model_config['arch_name']
    assert model_config['name'] in ['graph_saint', 'cluster_gcn']
    if args.simulate:
        print('=' * 30 + 'Simulation' + '=' * 30 )
        config.simulate = True
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
    print(f'amp mode: {config.amp}')
    if args.amp:
        print('activate amp mode...')
        config.amp = True
        scaler = GradScaler()
    else:
        config.amp = False
        scaler = None

    if args.dropout2:
        print('using dropout2')
        config.dropout2 = True

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
    grad_norm = model_config.get('grad_norm', None)
    print(f'clipping grad norm: {grad_norm}')
    torch.cuda.set_device(args.gpu)
    data, num_features, num_classes = get_data(args.root, 'products')
    sampler_cls = SAMPLER[model_config['name']]
    s_time = time.time()
    print("=" * 30 + f'Prepare {sampler_cls.__name__} for training' + '=' * 30)
    if model_config['name'] == 'graph_saint':
        loader = sampler_cls(data,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            **model_config['sampler'])

    elif model_config['name'] == 'cluster_gcn':
        cluster_data = sampler_cls(data, 
                                   **model_config['sampler'])
        loader = ClusterLoader(cluster_data, batch_size=args.batch_size, 
                               shuffle=True, num_workers=args.num_workers)
    else:
        raise NotImplementedError
    print("=" * 30 + 
          f'Finished Building {sampler_cls.__name__}, used {time.time() - s_time} sec' + 
          '=' * 30)
    GNN = getattr(models, args.model)
    model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    if use_qmodule:
        print('convert the model')
        model = QModule(model)
    print(model)
    model.cuda(args.gpu)
    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)
    if args.debug_mem:
        print("========== Model Only ===========")
        usage = get_memory_usage(args.gpu, True)
        exp_recorder.record("network", 'SAGE')
        exp_recorder.record("model_only", usage / MB, 2)    
        model.reset_parameters()
        model.train()
        optimizer = get_optimizer(model_config, model)
        print("========== Load data to GPU ===========")
        for data in loader:     
            optimizer.zero_grad()
            data = T.ToSparseTensor()(data.to('cuda'))
            data.adj_t.fill_cache_()
            num_nodes, num_edges = data.num_nodes, data.num_edges
            print(f'sampled sub-graph: #nodes: {num_nodes}, #edges: {num_edges}')
            init_mem = get_memory_usage(args.gpu, True)
            data_mem = init_mem / MB - exp_recorder.val_dict['model_only']
            exp_recorder.record("data", init_mem / MB - exp_recorder.val_dict['model_only'], 2)
            with autocast(enabled=args.amp):
                out = model(data.x, data.adj_t)
                y = data.y.squeeze(1)
                loss = F.cross_entropy(out[data.train_mask], y[data.train_mask])
            print("========== Before Backward ===========")
            before_backward = get_memory_usage(args.gpu, True)
            act_mem = get_memory_usage(args.gpu, False) - init_mem - compute_tensor_bytes([loss, out])
            res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (before_backward / MB,
                                                                            data_mem,
                                                                            act_mem / MB)
            print(res)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
            exp_recorder.dump('mem_results.json')
            exit()

    print("=" * 30 + 'Prepare NeighborSampler for evaluation' + '=' * 30)
    s_time = time.time()
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=4096, shuffle=False)
    print("=" * 30 + 
          f'Finished Building NeighborSampler, used {time.time() - s_time} sec' + 
          '=' * 30)
    data.edge_index = None
    eval_start_epoch = model_config['eval_start_epoch']
    eval_steps = model_config['eval_steps']

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        for epoch in range(1, 1 + model_config['epochs']):
            loss = train(model, loader, optimizer, scaler, amp_mode=config.amp, grad_norm=grad_norm)
            if model_config['log_steps'] > 0 and epoch % model_config['log_steps'] == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}')

            if epoch > eval_start_epoch and epoch % eval_steps == 0:
                result = test(model, data, evaluator, subgraph_loader, amp_mode=args.amp)
                logger.add_result(run, result)
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == '__main__':
    main()