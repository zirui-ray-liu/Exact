import sys
import os

from torch.optim import optimizer
sys.path.append(os.getcwd())
import argparse
import random
import time
import yaml
import math
import warnings

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.nodeproppred import Evaluator

from exact import get_memory_usage, compute_tensor_bytes, exp_recorder, config, QModule
from utils import cast_int32
import models
from data import get_data
from logger import Logger


EPSILON = 1 - math.log(2)
MB = 1024**2
GB = 1024**3


parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, required=True)
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--debug_mem', action='store_true',
                    help='whether to debug the memory usage')
parser.add_argument('--n_bits', type=int, default=8)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--simulate', action='store_true')
parser.add_argument('--act_fp', action='store_true')
parser.add_argument('--kept_frac', type=float, default=1.0)
parser.add_argument('--amp', help='whether to enable apx mode', action='store_true')
parser.add_argument('--dropout2', help='whether to use dropout2', action='store_true')


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def drop_edge(adj_t, edge_drop):
    if edge_drop == 0:
        return adj_t
    nnz = adj_t.nnz()
    bound = int(nnz * edge_drop)
    perm = torch.randperm(nnz).cuda()
    eids = perm[bound:]
    row, col, value = adj_t.storage._row, adj_t.storage._col, adj_t.storage._value
    row, col= row[eids], col[eids]
    if value is not None:
        value = value[eids]
    adj_t = SparseTensor(row=row, col=col, value=value, 
                            sparse_sizes=(adj_t.size(0), adj_t.size(1)), is_sorted=False)
    return adj_t

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


@torch.no_grad()
def test(model, data, evaluator, amp_mode=False):
    model.eval()
    with autocast(enabled=amp_mode):
        out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[data.train_mask],
        'y_pred': y_pred[data.train_mask],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[data.val_mask],
        'y_pred': y_pred[data.val_mask],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_mask],
        'y_pred': y_pred[data.test_mask],
    })['acc']
    return train_acc, valid_acc, test_acc


def main():
    args = parser.parse_args()
    with open(args.conf, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
    args.model = model_config['name']
    assert args.model.lower() in ['gcn', 'sage', 'gat']
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
    
    if args.amp:
        config.amp = True
        print(f'amp mode: {config.amp}')
        
    if args.dropout2:
        print('using dropout2')
        config.dropout2 = True

    grad_norm = model_config.get('grad_norm', None)
    print(f'clipping grad norm: {grad_norm}')
    torch.cuda.set_device(args.gpu)

    data, num_features, num_classes = get_data(args.root, 'products')
    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)
    GNN = getattr(models, args.model)
    model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    if model_config['loop']:
        t = time.perf_counter()
        print('Adding self-loops...', end=' ', flush=True)
        data.adj_t = data.adj_t.set_diag()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    if model_config['normalize']:
        t = time.perf_counter()
        print('Normalizing data...', end=' ', flush=True)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    if use_qmodule:
        print('convert the model')
        model = QModule(model)
    print(model)
    model.cuda(args.gpu)
    if args.debug_mem:
        model.train()
        print("========== Model Only ===========")
        usage = get_memory_usage(args.gpu, True)
        exp_recorder.record("network", 'GCN')
        exp_recorder.record("model_only", usage / MB, 2)
        print("========== Load data to GPU ===========")
        data = T.ToSparseTensor()(data).to('cuda')
        data.adj_t.fill_cache_()
        init_mem = get_memory_usage(args.gpu, True)
        data_mem = init_mem / MB - exp_recorder.val_dict['model_only']
        exp_recorder.record("data", init_mem / MB - exp_recorder.val_dict['model_only'], 2)
        model.reset_parameters()
        model.train()
        optimizer = get_optimizer(model_config, model)
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)[data.train_mask]
        loss = F.nll_loss(out, data.y.squeeze(1)[data.train_mask])
        print("========== Before Backward ===========")
        del out
        before_backward = get_memory_usage(args.gpu, True)
        act_mem = get_memory_usage(args.gpu, False) - init_mem - compute_tensor_bytes([loss])
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
        exp_recorder.dump('mem_results.json')
        exit()
    data = T.ToSparseTensor()(data)
    data = data.to('cuda')
    # ======================================================================
    # Note: if you want to train the model with 32-bit graph structure data,
    # uncomment the below code.

    # cast_int32(data)

    # ======================================================================
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        if args.amp:
            print('activate amp mode')
            scaler = GradScaler()
        else:
            scaler = None
        for epoch in range(1, 1 + model_config['epochs']):
            # ==== train the model ====
            if model_config.get('adjust_lr', False):
                adjust_learning_rate(optimizer, model_config['lr'], epoch)
            model.train()
            optimizer.zero_grad()
            with autocast(enabled=args.amp):
                out = model(data.x, data.adj_t)
                loss = F.cross_entropy(out[data.train_mask], data.y.squeeze(1)[data.train_mask])
            torch.cuda.empty_cache()
            if args.amp:
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
            # ===========================
            result = test(model, data, evaluator, args.amp)
            logger.add_result(run, result)
            if  model_config['log_steps'] > 0 and epoch % model_config['log_steps'] == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == '__main__':
    main()