from torch import nn
from copy import deepcopy

from ai.util import log2_diff


def seq(*a):
    if not a:
        return nn.Identity()
    return nn.Sequential(*a)


def repeat(n, obj):
    assert n >= 0
    if n == 0:
        return nn.Identity()
    if n == 1:
        return obj

    objs = [obj]
    while len(objs) < n:
        objs.append(deepcopy(obj))

    return nn.Sequential(*objs)


class Pyramid(nn.Module):
    def __init__(s, size_start, size_end, nc_min, nc_max, block):
        super().__init__()
        assert nc_min < nc_max

        n = log2_diff(size_start, size_end)
        nc = [min(nc_min*2**i, nc_max) for i in range(n+1)]
        sizes = [min(size_start, size_end) * 2**i for i in range(n+1)]
        if size_start < size_end:
            nc = nc[::-1]
        else:
            sizes = sizes[::-1]

        blocks = [
            block(sizes[i], nc[i], nc[i+1])
            for i in range(n)
        ]
        s._net = blocks[0] if n == 1 else nn.Sequential(*blocks)

        s.nc_in = nc[0]
        s.nc_out = nc[-1]

    def forward(s, x):
        return s._net(x)

def pyramid(*a, **kw):
    return Pyramid(*a, **kw)
