'''Functional operations.'''

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union
import einops


ALIGN_CORNERS_MODES = ['linear', 'bilinear', 'trilinear', 'bicubic']

def resample(x: Tensor, scale: Union[int, float], mode: str = 'bilinear'):
    '''Resample x by scale factor.'''

    return F.interpolate(
        x,
        scale_factor=scale,
        mode=mode,
        align_corners=True if mode in ALIGN_CORNERS_MODES else None,
    )


def flatten(x: Tensor, start_dim: int = 1):
    return torch.flatten(x, start_dim)


def normalize_2nd_moment(x: Tensor, dim: int = 1, eps: float = 1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# aliases
one_hot = F.one_hot
cat = torch.cat
rearrange = einops.rearrange
pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence
