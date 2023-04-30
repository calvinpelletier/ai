import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List


def resample(scale: Union[int, float]):
    '''Resize a feature map by scale factor.

    INPUT
        tensor[b, c, h, w]
    OUTPUT
        tensor[b, c, h * <scale>, w * <scale>]

    ARGS
        scale
            The scale factor.
    '''

    if scale < 1:
        return nn.AvgPool2d(int(1 / scale))
    elif scale > 1:
        return nn.Upsample(scale_factor=scale, mode='nearest')
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
        start_dim : int
            default 1 to preserve batch dim: [bs, ...] to [bs, n]
    '''

    def __init__(s, start_dim: int = 1):
        super().__init__()
        s._start_dim = start_dim

    def forward(s, x):
        return torch.flatten(x, s._start_dim)

def flatten(*a, **kw):
    return Flatten(*a, **kw)


class Blur(nn.Module):
    def __init__(s, pad: List[int] = [2,2,2,2], gain: float = 1., up: int = 1):
        super().__init__()
        s._pad = pad
        s._gain = gain
        s._up = up

        f = torch.as_tensor([1, 3, 3, 1], dtype=torch.float32)
        f = f.ger(f)
        s.register_buffer('_filter', f / f.sum())

    def forward(s, x):
        bs, nc, h, w = x.shape

        x = x.reshape([bs, nc, h, 1, w, 1])
        x = F.pad(x, [0, s._up - 1, 0, 0, 0, s._up - 1])
        x = x.reshape([bs, nc, h * s._up, w * s._up])
        x = F.pad(x, s._pad)

        f = s._filter * (s._gain ** (s._filter.ndim / 2)) # type: ignore
        f = f.to(x.dtype)
        f = f.flip(list(range(f.ndim)))
        f = f[np.newaxis, np.newaxis].repeat([nc, 1] + [1] * f.ndim)

        return F.conv2d(input=x, weight=f, groups=nc)

def blur(*a, **kw):
    return Blur(*a, **kw)


class Gain(nn.Module):
    def __init__(s, val):
        super().__init__()
        s._val = val

    def forward(s, x):
        return x * s._val

def gain(*a, **kw):
    return Gain(*a, **kw)
