import sys
import os
import numpy as np
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

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.nodeproppred import Evaluator

from exact import get_memory_usage, compute_tensor_bytes, exp_recorder, config, QModule

import models
import torch_geometric.transforms as T
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
parser.add_argument('--test_speed', action='store_true', help='whether to test the speed and throughout')

 
def add_labels(feat, labels, n_classes, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device='cuda')
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(EPSILON + y) - math.log(EPSILON)
    return torch.mean(y)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


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
def test(model, data, evaluator, n_classes, use_labels=False, amp_mode=False):
    model.eval()
    if use_labels:
        feat = add_labels(data.x, data.y, n_classes, data.train_mask)
    else:
        feat = data.x
    with autocast(enabled=amp_mode):
        out = model(feat, data.adj_t)
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
    assert args.model.lower() in ['gcn', 'sage', 'gat', 'gcn2']
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
    if args.amp:
        config.amp = True
        scaler = GradScaler()
    else:
        config.amp = False
        scaler = None
    print(f'amp mode: {config.amp}')
    use_custom_loss = model_config.get('use_custom_loss', True)
    if use_custom_loss:
        print('use the custom loss function')
    else:
        print('use crossentropy as the loss function')
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
        print('activate amp mode')
        config.amp = True
        scaler = GradScaler()
    else:
        scaler = None

    use_labels, n_label_iters, mask_rate = model_config.get('use_labels', False), \
                                           model_config.get('n_label_iters', 0.), \
                                           model_config.get('mask_rate', 0.5)
    grad_norm = model_config.get('grad_norm', None)
    print(f'clipping grad norm: {grad_norm}')
    if not use_labels and n_label_iters > 0:
        raise ValueError("'use_labels' must be enabled when n_label_iters > 0")
    torch.cuda.set_device(args.gpu)

    data, num_features, num_classes = get_data(args.root, 'arxiv')
    data = T.ToSparseTensor()(data)
    num_features = num_features + num_classes if use_labels else num_features
    evaluator = Evaluator(name='ogbn-arxiv')
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
        print("========== Model and Optimizer Only ===========")
        optimizer = get_optimizer(model_config, model)
        optimizer.zero_grad()
        usage = get_memory_usage(args.gpu, True)
        exp_recorder.record("network", args.model)
        exp_recorder.record("model_only", usage / MB, 4)
        print("========== Load data to GPU ===========")
        data.adj_t.fill_cache_()
        data = data.to('cuda')
        init_mem = get_memory_usage(args.gpu, True)
        data_mem = init_mem / MB - exp_recorder.val_dict['model_only']
        exp_recorder.record("data", init_mem / MB - exp_recorder.val_dict['model_only'], 4)
        model.reset_parameters()
        model.train()
        with autocast(enabled=args.amp):
            out = model(data.x, data.adj_t)[data.train_idx]
            loss = F.nll_loss(out, data.y.squeeze(1)[data.train_idx])
        print("========== Before Backward ===========")
        before_backward = get_memory_usage(args.gpu, True)
        act_mem = get_memory_usage(args.gpu, False) - init_mem - compute_tensor_bytes([loss, out])

        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (before_backward / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(f'max allocated mem (MB): {torch.cuda.max_memory_allocated(0) / MB}')
        print(res)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        del loss, out
        print("========== After Backward ===========")
        after_backward = get_memory_usage(args.gpu, True)
        total_mem = before_backward + (after_backward - init_mem)
        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (total_mem / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(res)
        print(f'max allocated mem (MB): {torch.cuda.max_memory_allocated(0) / MB}')
        exp_recorder.record("total", total_mem / MB, 2)
        exp_recorder.record("activation", act_mem / MB, 2)
        exp_recorder.dump('mem_results.json')
        torch.cuda.synchronize()
        s_time = time.time()
        if args.test_speed:
            model.reset_parameters()
            optimizer.zero_grad()
            epoch_per_sec = []

            for i in range(100):
                t = time.time()
                optimizer.zero_grad()
                out = model(data.x, data.adj_t)[data.train_idx]
                loss = F.nll_loss(out, data.y.squeeze(1)[data.train_idx])
                loss.backward()
                optimizer.step()
                duration = time.time() - t
                epoch_per_sec.append(duration)
                print(f'epoch {i}, duration: {duration} sec')
            torch.cuda.synchronize()
            print(f'epoch/s: {100/(time.time() - s_time) }')
        exit()

    data = data.to('cuda')
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        for epoch in range(1, 1 + model_config['epochs']):
            # ==== train the model ====
            if model_config.get('adjust_lr', False):
                adjust_learning_rate(optimizer, model_config['lr'], epoch)
            model.train()
            optimizer.zero_grad()
            with autocast(enabled=args.amp):
                if use_labels:
                    mask = torch.rand(data.train_idx.shape) < mask_rate
                    train_labels_idx = data.train_idx[mask]
                    train_pred_idx = data.train_idx[~mask]
                    feat = add_labels(data.x, data.y, num_classes, train_labels_idx)
                else:
                    train_pred_idx = data.train_idx
                    feat = data.x
                out = model(feat, data.adj_t)
                if use_custom_loss:
                    loss = custom_loss_function(out[train_pred_idx], data.y[train_pred_idx])
                else:
                    loss = F.cross_entropy(out[train_pred_idx], data.y.squeeze(1)[train_pred_idx])
            del feat
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
            result = test(model, data, evaluator, num_classes, use_labels, args.amp)
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