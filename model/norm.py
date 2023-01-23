import torch
from torch import nn


CONV2D_NORMS = {
    'batch': (nn.BatchNorm2d, True),
    'instance': (nn.InstanceNorm2d, False),
}

def build_conv2d_norm(norm, nc):
    if norm is None:
        return None, False
    cls, norm_has_bias = CONV2D_NORMS[norm]
    return cls(nc), norm_has_bias
