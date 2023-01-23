import torch
from torch import nn


def resample(stride):
    if stride > 1:
        return nn.AvgPool2d(stride)
    elif stride < 1:
        return nn.Upsample(scale_factor=int(1 / stride), mode='nearest')
    return nn.Identity()


def global_avg():
    return nn.AdaptiveAvgPool2d(1)


class Residual(nn.Module):
    def __init__(s, main, shortcut=None):
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
        super().__init__()
        s._val = val

    def forward(s, x):
        return x.clamp(-s._val, s._val)

def clamp(*a, **kw):
    return Clamp(*a, **kw)


class Flatten(nn.Module):
    def __init__(s, keep_batch_dim=True):
        super().__init__()
        s._keep_batch_dim = keep_batch_dim

    def forward(s, x):
        if s._keep_batch_dim:
            return torch.flatten(x, 1)
        return torch.flatten(x)

def flatten(*a, **kw):
    return Flatten(*a, **kw)
