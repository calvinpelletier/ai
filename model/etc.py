import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def resample(stride):
    '''resize a feature map by scale factor of 1/stride

    stride : int or float
        inverse of the scale factor
    '''

    if stride > 1:
        return nn.AvgPool2d(stride)
    elif stride < 1:
        return nn.Upsample(scale_factor=int(1 / stride), mode='nearest')
    return nn.Identity()


def global_avg():
    '''feature map [b, c, h, w] -> global spatial average [b, c, 1, 1]'''

    return nn.AdaptiveAvgPool2d(1)


class Residual(nn.Module):
    '''main operation plus a shortcut around it'''

    def __init__(s, main, shortcut=None):
        '''
        main : module
        shortcut : module or null
            if null, the residual is simply the input
        '''

        super().__init__()
        s._main = main
        s._shortcut = shortcut

    def forward(s, x):
        res = x if s._shortcut is None else s._shortcut(x)
        return s._main(x) + res

def res(*a, **kw):
    return Residual(*a, **kw)


class Clamp(nn.Module):
    def __init__(s, val):
        '''
        val : float or null
            clamp all output values between [-val, val]
        '''

        super().__init__()
        s._val = val

    def forward(s, x):
        return x.clamp(-s._val, s._val)

def clamp(*a, **kw):
    return Clamp(*a, **kw)


class Flatten(nn.Module):
    def __init__(s, keep_batch_dim=True):
        '''
        keep_batch_dim : bool
            if true (default), [bs, ...] to [bs, n]
            else, [...] to [n]
        '''

        super().__init__()
        s._keep_batch_dim = keep_batch_dim

    def forward(s, x):
        if s._keep_batch_dim:
            return torch.flatten(x, 1)
        return torch.flatten(x)

def flatten(*a, **kw):
    return Flatten(*a, **kw)


class Blur(nn.Module):
    def __init__(s, up, pad, gain):
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
