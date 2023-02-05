'''components of ResNets'''

from torch import nn

from ai.model.sequence import seq
from ai.model.conv2d import conv
from ai.model.etc import resample, res, global_avg


def resblk(nc1, nc2, stride=1, actv='mish', norm='batch', se=True):
    '''resnet block

    input
        tensor[b, <nc1>, h, w]
    output
        tensor[b, <nc2>, h*, w*] where h* = h / <stride>

    args
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

    main = [
        conv(nc1, nc2, stride=stride, norm=norm, actv=actv),
        conv(nc2, nc2, norm=norm),
    ]
    if se:
        main.append(SqueezeExcite(nc2, actv=actv))
    main = seq(*main)

    if nc1 != nc2 or stride != 1:
        shortcut = seq(resample(stride), conv(nc1, nc2, k=1))
    else:
        shortcut = None

    return res(main, shortcut)


def resblk_group(n, nc1, nc2, stride=1, **kw):
    '''sequence of resnet blocks (but only striding once)

    input
        tensor[b, <nc1>, h, w]
    output
        tensor[b, <nc2>, h*, w*] where h* = h / <stride>

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
    '''self-modulate using global information

    input
        tensor[b, <nc>, h, w]
    output
        tensor[b, <nc>, h, w]
    '''

    def __init__(s, nc, reduction=16, actv='mish'):
        '''
        nc : int
            number of channels
        reduction : int
            intermediate number of channels during squeeze = nc / reduction
        actv : str
            intermediate activation function
        '''

        super().__init__()
        reduction = min(reduction, nc)

        s._net = seq(
            global_avg(),
            conv(nc, nc // reduction, k=1, actv=actv),
            conv(nc // reduction, nc, k=1, actv='sigmoid'),
        )

    def forward(s, x):
        return x * s._net(x)

def se(*a, **kw):
    return SqueezeExcite(*a, **kw)
