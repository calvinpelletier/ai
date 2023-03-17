'''components of ResNets'''

from torch import nn
from typing import Optional, Union

from ai.model.sequence import seq
from ai.model.conv2d import conv
from ai.model.etc import resample, res, global_avg


def resblk(
    nc1: int,
    nc2: int,
    stride: Union[int, float] = 1,
    actv: str = 'mish',
    norm: Optional[str] = 'batch',
    se: bool = True,
) -> nn.Module:
    '''ResNet block.

    INPUT
        tensor[b, <nc1>, h, w]
    OUTPUT
        tensor[b, <nc2>, h*, w*] where h* = h / <stride>

    ARGS
        nc1 : int
            number of input channels
        nc2 : int
            number of output channels
        stride : int or float
            for strides < 1 (i.e. an up convolution), the input is first resized
            by a scale factor of 1/stride before using a conv of stride=1
        actv : str or null
            activation (see model/actv.py)
        norm : str or null
            normalization (see model/norm.py)
        se : bool
            add squeeze-excitation op (self-modulate using global information)
    '''

    # main path
    main = [
        conv(nc1, nc2, stride=stride, norm=norm, actv=actv),
        conv(nc2, nc2, norm=norm),
    ]
    if se:
        main.append(SqueezeExcite(nc2, actv=actv))
    main = seq(*main)

    # shortcut
    if nc1 != nc2 or stride != 1:
        shortcut = conv(nc1, nc2, k=1, stride=stride)
    else:
        shortcut = None # simple addition residual

    return res(main, shortcut)


def resblk_group(
    n: int,
    nc1: int,
    nc2: int,
    stride: Union[int, float] = 1,
    **kw,
) -> nn.Module:
    '''Sequence of resnet blocks (but only striding once).

    INPUT
        tensor[b, <nc1>, h, w]
    OUTPUT
        tensor[b, <nc2>, h*, w*] where h* = h / <stride>

    ARGS
        n : int
            number of blocks
        nc1 : int
            number of input channels
        nc2 : int
            number of output channels
        stride : int or float
            stride of the first block (all others use stride=1)
        **kw
            any additional resblk kwargs
    '''

    assert n > 0
    blocks = [resblk(nc1, nc2, stride, **kw)]
    for _ in range(n - 1):
        blocks.append(resblk(nc2, nc2, 1, **kw))
    return seq(*blocks)


class SqueezeExcite(nn.Module):
    '''Squeeze-Excitation operation (self-modulate using global information).

    INPUT
        tensor[b, <nc>, h, w]
    OUTPUT
        tensor[b, <nc>, h, w]

    ARGS
        nc : int
            number of channels
        reduction : int
            intermediate number of channels during squeeze = nc / reduction
        actv : str
            intermediate activation function
    '''

    def __init__(s, nc: int, reduction: int = 16, actv: str = 'mish'):
        super().__init__()
        reduced = max(1, nc // reduction)
        s._net = seq(
            global_avg(),
            conv(nc, reduced, k=1, actv=actv),
            conv(reduced, nc, k=1, actv='sigmoid'),
        )

    def forward(s, x):
        return x * s._net(x)

def se(*a, **kw) -> nn.Module:
    return SqueezeExcite(*a, **kw)
