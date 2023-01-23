import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.model.conv2d import conv, Noise
from ai.model.linear import fc
from ai.model.norm import AdaLIN
from ai.model.actv import build_actv
from ai.model.sequence import seq
from ai.model.etc import Clamp


def modconv(
    nc1,
    nc2,
    z_dim,
    modtype='adalin',
    k=3,
    stride=1,
    actv='mish',
    clamp=None,
    noise=False,
    padtype='zeros',
    scale_w=False,
    lr_mult=None,
):
    if modtype == 'adalin':
        return NormModConv(AdaLIN, nc1, nc2, z_dim, k, stride, actv, clamp,
            noise, padtype, scale_w, lr_mult)

    elif modtype == 'weight':
        assert k == 3, 'TODO'
        assert padtype == 'zeros', 'TODO'
        assert not scale_w, 'TODO'
        assert lr_mult is None, 'TODO'
        return WeightModConv(nc1, nc2, z_dim, stride, actv, clamp, noise)

    raise NotImplementedError(f'modtype={modtype}')


class NormModConv(nn.Module):
    def __init__(s,
        Norm,
        nc1,
        nc2,
        z_dim,
        k=3,
        stride=1,
        actv='mish',
        clamp=None,
        noise=False,
        padtype='zeros',
        scale_w=False,
        lr_mult=None,
    ):
        super().__init__()

        s._conv = conv(
            nc1,
            nc2,
            k=k,
            stride=stride,
            bias=False,
            padtype=padtype,
            scale_w=scale_w,
            lr_mult=lr_mult,
        )

        s._norm = Norm(nc2, z_dim)

        post = []
        if noise:
            post.append(Noise())
        if actv is not None:
            post.append(build_actv(actv))
        if clamp is not None:
            post.append(Clamp(clamp))
        s._post = seq(*post)

    def forward(s, x, z):
        x = s._conv(x)
        x = s._norm(x, z)
        x = s._post(x)
        return x


class WeightModConv(nn.Module):
    def __init__(s,
        nc1,
        nc2,
        z_dim,
        stride=1,
        actv='mish',
        clamp=None,
        noise=False,
    ):
        super().__init__()
        assert stride >= 1 or stride == .5
        s._stride = stride

        s._fc = fc(z_dim, nc1, bias_init=1., scale_w=True)

        s._weight = nn.Parameter(torch.randn([nc2, nc1, 3, 3]))
        s._bias = nn.Parameter(torch.zeros([nc2]))

        post = []
        if noise:
            post.append(Noise())
        if actv is not None:
            post.append(build_actv(actv))
        if clamp is not None:
            post.append(Clamp(clamp))
        s._post = seq(*post)

    def init_params(s):
        nn.init.normal_(s._weight)

    def forward(s, x, z):
        bs = x.shape[0]

        z = s._fc(z)
        y = x * z.reshape(bs, -1, 1, 1)

        if s._stride >= 1:
            y = F.conv2d(
                input=y,
                weight=s._weight,
                bias=None,
                stride=s._stride,
                padding=1,
                dilation=1,
                groups=1,
            )
        else:
            y = F.conv_transpose2d(
                input=y,
                weight=s._weight.transpose(0, 1),
                bias=None,
                stride=2,
                padding=1,
                output_padding=1,
                groups=1,
                dilation=1,
            )

        w = s._weight.unsqueeze(0) * z.reshape(bs, 1, -1, 1, 1)
        dcoefs = ((w * w).sum(dim=[2,3,4]) + 1e-8).rsqrt()
        y *= dcoefs.reshape(bs, -1, 1, 1)

        return y + s._bias.reshape([1, -1, 1, 1])
