import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ai.model.actv import build_actv
from ai.model.norm import build_conv2d_norm
from ai.model.etc import resample, Clamp


def conv(
    nc1, nc2,
    k=3,
    stride=1,
    actv=None,
    norm=None,
    bias=True,
    padtype='zeros',
    scale_w=False,
    lr_mult=None,
    clamp=None,
):
    actv = build_actv(actv)
    norm, norm_has_bias = build_conv2d_norm(norm, nc2)

    seq = []

    if stride < 1:
        seq.append(resample(stride))

    if scale_w or lr_mult is not None:
        seq.append(_Conv2d(
            nc1, nc2,
            k=k,
            stride=max(1, stride),
            bias=bias and not norm_has_bias,
            scale_w=scale_w,
            lr_mult=lr_mult,
        ))
    else:
        seq.append(nn.Conv2d(
            nc1, nc2,
            kernel_size=k,
            stride=max(1, stride),
            padding=(k - 1) // 2,
            padding_mode=padtype,
            bias=bias and not norm_has_bias,
        ))

    if norm is not None:
        seq.append(norm)

    if actv is not None:
        seq.append(actv)

    if clamp is not None:
        seq.append(Clamp(clamp))

    if len(seq) == 1:
        return seq[0]
    return nn.Sequential(*seq)


class _Conv2d(nn.Module):
    def __init__(s,
        nc1,
        nc2,
        k=3,
        stride=1,
        bias=True,
        scale_w=False,
        lr_mult=None,
    ):
        super().__init__()
        if lr_mult is None:
            lr_mult = 1.

        s._weight = nn.Parameter(torch.randn([nc2, nc1, k, k]))
        s._bias = nn.Parameter(torch.randn([nc2])) if bias else None

        s._weight_mult = lr_mult
        if scale_w:
            s._weight_mult /= np.sqrt(nc1 * k**2)
        s._bias_mult = lr_mult

        s._stride = stride
        s._pad = (k - 1) // 2

    def init_params(s):
        nn.init.normal_(s._weight)
        if s._bias is not None:
            nn.init.normal_(s._bias)

    def forward(s, x):
        w = s._weight * s._weight_mult
        b = s._bias * s._bias_mult if s._bias is not None else None
        return F.conv2d(
            x,
            w,
            bias=b,
            stride=s._stride,
            padding=s._pad,
        )
