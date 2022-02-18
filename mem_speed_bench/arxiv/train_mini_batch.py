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
from torch_geometric.data import GraphSAINTRandomWalkSampler, ClusterData, ClusterLoader
import torch_geometric.transforms as T


from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.nodeproppred import Evaluator

from exact import get_memory_usage, compute_tensor_bytes, exp_recorder, config, QModule

import models
from data import get_data
from logger import Logger


EPSILON = 1 - math.log(2)
MB = 1024**2
GB = 1024**3

SAMPLER = {'graph_saint': GraphSAINTRandomWalkSampler, 'cluster_gcn': ClusterData}

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
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--simulate', action='store_true')
parser.add_argument('--act_fp', action='store_true')
parser.add_argument('--kept_frac', type=float, default=1.0)
parser.add_argument('--test_speed', action='store_true', help='whether to test the speed and throughout')


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels, reduction="none")
    y = torch.log(EPSILON + y) - math.log(EPSILON)
    return torch.mean(y)


def train(model, loader, optimizer, loss_op, grad_norm):
    model.train()
    total_loss = 0
    for data in loader:
        # s_time = time.time()
        data = T.ToSparseTensor()(data.to('cuda'))
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)
        loss = loss_op(out[data.train_mask], data.y.squeeze(1)[data.train_mask])
        del data
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        total_loss += loss.item()
        # print(f'used time: {time.time() - s_time}')
    return total_loss / len(loader)


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
def test(model, data, evaluator):
    model.eval()
    feat = data.x
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
    assert args.model.lower() in ['graph_saint', 'cluster_gcn']
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

    grad_norm = model_config.get('grad_norm', None)
    print(f'clipping grad norm: {grad_norm}')
    torch.cuda.set_device(args.gpu)

    data, num_features, num_classes = get_data(args.root, 'arxiv')
    sampler_data = data
    print('converting data form...')
    s_time = time.time()
    data = T.ToSparseTensor()(data.clone())
    data = data.to('cuda')
    print(f'done. used {time.time() - s_time} sec')

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

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)
    GNN = getattr(models, model_config['arch_name'])
    model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])

    if use_qmodule:
        print('convert the model')
        model = QModule(model)
    print(model)
    model.cuda(args.gpu)
    loss_op = custom_loss_function if use_custom_loss else F.cross_entropy
    if args.debug_mem:
        print("========== Model and Optimizer Only ===========")
        optimizer = get_optimizer(model_config, model)
        optimizer.zero_grad()
        usage = get_memory_usage(args.gpu, True)
        exp_recorder.record("network", args.model)
        exp_recorder.record("model_only", usage / MB, 4)
        print("========== Load data to GPU ===========")
        for _data in loader:     
            optimizer.zero_grad()
            _data = T.ToSparseTensor()(_data.to('cuda'))
            _data.adj_t.fill_cache_()
            init_mem = get_memory_usage(args.gpu, True)
            data_mem = init_mem / MB - exp_recorder.val_dict['model_only']
            exp_recorder.record("data", init_mem / MB - exp_recorder.val_dict['model_only'], 4)
            model.reset_parameters()
            model.train()
            out = model(_data.x, _data.adj_t)[_data.train_mask]
            loss = F.nll_loss(out, _data.y.squeeze(1)[_data.train_mask])
            print("========== Before Backward ===========")
            before_backward = get_memory_usage(args.gpu, True)
            act_mem = get_memory_usage(args.gpu, False) - init_mem - compute_tensor_bytes([loss, out])

            res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (before_backward / MB,
                                                                            data_mem,
                                                                            act_mem / MB)
            print(res)
            del out
            loss.backward()
            optimizer.step()
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
                print(f's/epoch: {np.mean(epoch_per_sec)}')
                torch.cuda.synchronize()
                print(f'epoch/s: {100/(time.time() - s_time) }')
            exit()

    data = data.to('cuda')
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        for epoch in range(1, 1 + model_config['epochs']):
            loss = train(model, loader, optimizer, loss_op, grad_norm)
            result = test(model, data, evaluator)
            logger.add_result(run, result)
            if  model_config['log_steps'] > 0 and epoch % model_config['log_steps'] == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == '__main__':
    main()