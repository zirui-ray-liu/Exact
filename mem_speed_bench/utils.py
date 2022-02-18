
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from exact import get_memory_usage
from torch_sparse import SparseTensor
from ogb.nodeproppred import Evaluator

def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask

# from torch_sparse.storage import SparseStorage
def cast_int32(data):
    data.adj_t.storage.rowptr()
    data.adj_t.storage.col()
    data.adj_t.storage.row()
    data.adj_t.storage.rowcount() 
    data.adj_t.storage.colptr()
    data.adj_t.storage._colptr = data.adj_t.storage._colptr.int()
    data.adj_t.storage._rowptr = data.adj_t.storage._rowptr.int()
    data.adj_t.storage._col = data.adj_t.storage._col.int()
    data.adj_t.storage._row = data.adj_t.storage._row.int()
    data.adj_t.storage._rowcount = data.adj_t.storage._rowcount.int()