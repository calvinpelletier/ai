from torch import nn

from ai.model.sequence import seq
from ai.model.conv2d import conv
from ai.model.etc import resample, res, global_avg


def resblk(nc1, nc2, stride=1, norm='batch', actv='mish', se=True):
    main = [
        conv(nc1, nc2, stride=stride, norm=norm, actv=actv),
        conv(nc2, nc2, norm=norm),
    ]
    if se:
        main.append(SqueezeExcite(nc2))
    main = seq(*main)

    if nc1 != nc2 or stride != 1:
        shortcut = seq(resample(stride), conv(nc1, nc2, k=1))
    else:
        shortcut = None

    return res(main, shortcut)


def resblk_group(n, nc1, nc2, stride=1, norm='batch', actv='mish', se=True):
    assert n > 0
    blocks = [resblk(nc1, nc2, stride, norm, actv, se)]
    for _ in range(n - 1):
        blocks.append(resblk(nc2, nc2, 1, norm, actv, se))
    return seq(*blocks)


class SqueezeExcite(nn.Module):
    def __init__(s, nc, reduction=16, actv='mish'):
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
