'''sequences of modules'''

from torch import nn
from copy import deepcopy

from ai.util import log2_diff


def seq(*a):
    if not a:
        return nn.Identity()
    return nn.Sequential(*a)


def repeat(n, obj):
    '''sequence of <obj> deepcopied <n> times'''

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
    '''sequence of blocks that results in a pyramid of feature maps

    input
        tensor[b, *nc_in, <size_in>, <size_in>]
    output
        tensor[b, *nc_out, <size_out>, <size_out>]

    * nc_in and nc_out are exposed as attributes of the Pyramid object

    if down (size_in > size_out)
        input depth = <nc_min>
        each block halves the size and doubles the depth (max <nc_max>)

    if up (size_in < size_out)
        inverse of the above sequence

    see model/ae.py for an example

    args
        size_in : int
            input size
        size_out : int
            output size
        nc_min : int
            minimum number of channels (input channels if down, output if up)
        nc_max : int
            maximum number of channels
        block : callable
            factory that produces blocks: block(size, nc1, nc2)
                size : int
                    size of the input feature map to the block
                nc1 : int
                    number of input channels
                nc2 : int
                    number of output channels
    '''

    def __init__(s, size_in, size_out, nc_min, nc_max, block):
        super().__init__()
        assert nc_min < nc_max

        n = log2_diff(size_in, size_out)
        assert n > 0

        nc = [min(nc_min*2**i, nc_max) for i in range(n+1)]
        sizes = [min(size_in, size_out) * 2**i for i in range(n+1)]
        if size_in < size_out:
            nc = nc[::-1]
        else:
            sizes = sizes[::-1]
        s.nc_in = nc[0]
        s.nc_out = nc[-1]

        blocks = [
            block(sizes[i], nc[i], nc[i+1])
            for i in range(n)
        ]
        s._net = blocks[0] if n == 1 else nn.Sequential(*blocks)

    def forward(s, x):
        return s._net(x)

def pyramid(*a, **kw):
    return Pyramid(*a, **kw)
