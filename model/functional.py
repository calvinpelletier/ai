'''Functional operations.'''

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union


ALIGN_CORNERS_MODES = ['linear', 'bilinear', 'trilinear', 'bicubic']

def resample(x: Tensor, scale: Union[int, float], mode: str = 'bilinear'):
    '''Resample x by scale factor.'''

    return F.interpolate(
        x,
        scale_factor=scale,
        mode=mode,
        align_corners=True if mode in ALIGN_CORNERS_MODES else None,
    )


def flatten(x: Tensor, keep_batch_dim: bool = True):
    if keep_batch_dim:
        return torch.flatten(x, 1)
    return torch.flatten(x)


def normalize_2nd_moment(x: Tensor, dim: int = 1, eps: float = 1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# aliases
def one_hot(*a, **kw):
    return F.one_hot(*a, **kw)
def cat(*a, **kw):
    return torch.cat(*a, **kw)
