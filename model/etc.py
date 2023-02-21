import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List
from einops.layers.torch import Rearrange


def resample(stride: Union[int, float]):
    '''Resize a feature map by scale factor of 1/stride.

    INPUT
        tensor[b, c, h, w]
    OUTPUT
        tensor[b, c, h / <stride>, w / <stride>]

    ARGS
        stride : int or float
            inverse of the scale factor
    '''

    if stride > 1:
        return nn.AvgPool2d(stride)
    elif stride < 1:
        return nn.Upsample(scale_factor=int(1 / stride), mode='nearest')
    return nn.Identity()


def global_avg():
    '''Global spatial average.

    INPUT
        tensor[b, c, h, w]
    OUTPUT
        tensor[b, c, 1, 1]
    '''

    return nn.AdaptiveAvgPool2d(1)


class Residual(nn.Module):
    '''Main operation plus a shortcut around it.

    ARGS
        main : module
        shortcut : module or null
            if null, the residual is simply the input
    '''

    def __init__(s, main: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        s._main = main
        s._shortcut = shortcut

    def forward(s, x):
        res = x if s._shortcut is None else s._shortcut(x)
        return s._main(x) + res

def res(*a, **kw):
    return Residual(*a, **kw)


class Clamp(nn.Module):
    '''Restrict input tensor values within a range.

    ARGS
        val : float or null
            clamp all output values between [-val, val]
    '''

    def __init__(s, val: Optional[float]):
        super().__init__()
        s._val = val

    def forward(s, x):
        if s._val is None:
            return x
        return x.clamp(-s._val, s._val)

def clamp(*a, **kw):
    return Clamp(*a, **kw)


class Flatten(nn.Module):
    '''Flatten the input tensor.

    ARGS
        keep_batch_dim : bool
            if true (default), [bs, ...] to [bs, n]
            else, [...] to [n]
    '''

    def __init__(s, keep_batch_dim: bool = True):
        super().__init__()
        s._keep_batch_dim = keep_batch_dim

    def forward(s, x):
        if s._keep_batch_dim:
            return torch.flatten(x, 1)
        return torch.flatten(x)

def flatten(*a, **kw):
    return Flatten(*a, **kw)


class Blur(nn.Module):
    def __init__(s, up: int, pad: List[int], gain: float):
        super().__init__()
        s._up = up
        s._pad = pad
        s._gain = gain

        f = torch.as_tensor([1, 3, 3, 1], dtype=torch.float32)
        f = f.ger(f)
        assert f.ndim == 2
        f /= f.sum()
        s.register_buffer('_filter', f)

    def forward(s, x):
        bs, nc, h, w = x.shape

        x = x.reshape([bs, nc, h, 1, w, 1])
        x = F.pad(x, [0, s._up - 1, 0, 0, 0, s._up - 1])
        x = x.reshape([bs, nc, h * s._up, w * s._up])
        x = F.pad(x, s._pad)

        f = s._filter * (s._gain ** (s._filter.ndim / 2))
        f = f.to(x.dtype)
        f = f.flip(list(range(f.ndim)))
        f = f[np.newaxis, np.newaxis].repeat([nc, 1] + [1] * f.ndim)

        return F.conv2d(input=x, weight=f, groups=nc)

def blur(*a, **kw):
    return Blur(*a, **kw)


# aliases
rearrange = Rearrange
