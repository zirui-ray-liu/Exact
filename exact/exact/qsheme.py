import pdb

import torch
from .conf import config
import exact.cpp_extension.calc_precision as ext_calc_precision


class QScheme(object):
    num_samples = 1
    num_matmul_layer = 0
    num_lin_layer = 0
    num_batchnorm_layer = 0
    num_sage_layer = 0
    batch = None
    update_scale = True
    layers = []
    prev_layer = None

    def __init__(self, name):
        self.initial_bits = config.initial_bits
        self.bits = config.activation_compression_bits[0]
        QScheme.layers.append(self)

        assert name in ['linear', 'matmul', 'batchnorm', 'sage']
        if name == 'linear':
            self.name = f'linear_{QScheme.num_lin_layer}'
            QScheme.num_lin_layer += 1
        elif name == 'matmul':
            self.name = f'matmul_{QScheme.num_matmul_layer}'
            QScheme.num_matmul_layer += 1
        elif name == 'batchnorm':
            self.name = f'batchnorm_{QScheme.num_batchnorm_layer}'
            QScheme.num_batchnorm_layer += 1
        elif name == 'sage':
            self.name = f'sage_{QScheme.num_sage_layer}'
            QScheme.num_sage_layer += 1
        else:
            raise ValueError


    def compute_quantization_bits(self, input):
        QScheme.prev_layer = self
        N = input.shape[0]
        input_flatten = input.view(N, -1)
        mn, mx = torch.min(input_flatten, 1)[0], torch.max(input_flatten, 1)[0]
        # range = mx - mn
        b = config.activation_compression_bits[0]
        return input_flatten, b, mn, mx