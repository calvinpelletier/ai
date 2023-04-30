'''sequences of modules'''

from torch import nn
from torch.nn import Module
from copy import deepcopy
from typing import Callable, List

from ai.util import log2_diff


def seq(*a) -> Module:
    '''Like torch.nn.Sequential.'''

    if not a:
        return nn.Identity()
    return nn.Sequential(*a)


def repeat(n: int, obj: Module) -> Module:
    '''Sequence of <obj> deepcopied <n> times.

    NOTE: if repeating a custom module with params manually initialized in the
    constructor, each instance of the module will have the same initial param
    values unless you define an init_params() method, which will be called when
    the param init fn is applied recursively (i.e. when Model.init() is called).
    '''

    assert n >= 0
    if n == 0:
        return nn.Identity()
    if n == 1:
        return obj

    objs = [obj]
    while len(objs) < n:
        objs.append(deepcopy(obj))

    return nn.Sequential(*objs)


class Pyramid(Module):
    '''Sequence of blocks that results in a pyramid of feature maps.

    INPUT
        tensor[b, *nc_in, <size_in>, <size_in>]
    OUTPUT
        tensor[b, *nc_out, <size_out>, <size_out>]

    *nc_in/nc_out are calculated in the constructor then exposed as
    attributes of the Pyramid object.

    if down (size_in > size_out)
        input depth = <nc_min>
        each block halves the size and doubles the depth (max <nc_max>)

    if up (size_in < size_out)
        inverse of the above sequence (output depth = <nc_min>)

    see model/ae.py for an example

    ARGS
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

    ATTRIBUTES
        nc_in : int
        nc_out : int
    '''

    def __init__(s,
        size_in: int,
        size_out: int,
        nc_min: int,
        nc_max: int,
        block: Callable[[int, int, int], Module],
    ):
        super().__init__()
        assert nc_min < nc_max

        # number of blocks
        n = log2_diff(size_in, size_out)
        assert n > 0

        # intermediate channels and sizes
        nc = [min(nc_min*2**i, nc_max) for i in range(n+1)]
        sizes = [min(size_in, size_out) * 2**i for i in range(n+1)]
        if size_in < size_out:
            nc = nc[::-1]
        else:
            sizes = sizes[::-1]

        # expose expected input/output channels
        s.nc_in = nc[0]
        s.nc_out = nc[-1]

        # use provided factory to build blocks in a sequence
        blocks = [
            block(sizes[i], nc[i], nc[i+1])
            for i in range(n)
        ]
        s._net = blocks[0] if n == 1 else nn.Sequential(*blocks)

    def forward(s, x):
        return s._net(x)

def pyramid(*a, **kw) -> Module:
    return Pyramid(*a, **kw)


class ModSeq(Module):
    '''Sequence of moduled blocks.'''

    def __init__(s, blocks: List[Module]):
        super().__init__()
        s._blocks = nn.ModuleList(blocks)

    def forward(s, x, z):
        for block in s._blocks:
            x = block(x, z)
        return x

def modseq(*a, **kw) -> Module:
    return ModSeq(*a, **kw)
