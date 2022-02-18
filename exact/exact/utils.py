import os
from collections import OrderedDict
import json

import torch
import numpy as np


def cast_low_bit_int(idx_vec):
    if idx_vec is None:
        return idx_vec
    assert idx_vec.dtype == torch.int64
    max_v = torch.max(idx_vec)
    if max_v <= np.iinfo(np.int16).max:
        return idx_vec.type(torch.int16)
    elif max_v < np.iinfo(np.int32).max:
        return idx_vec.type(torch.int32)
    else:
        return idx_vec


def cast_adj(data):
    data.adj_t.storage._col = cast_low_bit_int(data.adj_t.storage._col)
    data.adj_t.storage._row = cast_low_bit_int(data.adj_t.storage._row)
    data.adj_t.storage._rowptr = cast_low_bit_int(data.adj_t.storage._rowptr)
    data.adj_t.storage._rowcount = cast_low_bit_int(data.adj_t.storage._rowcount)
    data.adj_t.storage._colptr = cast_low_bit_int(data.adj_t.storage._colptr)
    data.adj_t.storage._csr2csc = cast_low_bit_int(data.adj_t.storage._csr2csc)


def swap_to_cpu(tensor):
    tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu', pin_memory=True)
    tensor_cpu.copy_(tensor, non_blocking=True)
    return tensor_cpu


def get_memory_usage(gpu, print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(gpu)
    reserved = torch.cuda.memory_reserved(gpu)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    ret = 0
    for x in tensors:
        if x.dtype in [torch.int64, torch.long]:
            ret += np.prod(x.size()) * 8
        if x.dtype in [torch.float32, torch.int, torch.int32]:
            ret += np.prod(x.size()) * 4
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8]:
            ret += np.prod(x.size())
        else:
            print(x.dtype)
            raise ValueError()
    return ret


def empty_cache(ratio):
    if ratio is None:
        return
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if reserved > 0 and allocated / reserved < ratio:
        torch.cuda.empty_cache()


def disable_cache_allocator():
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'


def enable_cache_allocator():
    del os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING']


class GlobalExpRecorder:
    def __init__(self):
        self.val_dict = OrderedDict()

    def record(self, key, value, float_round=6):
        if isinstance(value, (np.int32, np.int64)):
            value = int(value)
        if isinstance(value, (float, np.float32, np.float64)):
            value = round(value, float_round)

        self.val_dict[key] = value

    def dump(self, filename):
        with open(filename, "a") as fout:
            fout.write(json.dumps(self.val_dict) + '\n')
        print("Save exp results to %s" % filename)

    def clear(self):
        pass

exp_recorder = GlobalExpRecorder()
