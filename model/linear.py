import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ai.model.actv import build_actv


def fc(
    n1,
    n2,
    actv=None,
    bias=True,
    bias_init=None,
    scale_w=False,
    lr_mult=None,
):
    if scale_w or lr_mult is not None or bias_init is not None:
        linear = Linear(n1, n2, bias, bias_init, scale_w, lr_mult)
    else:
        linear = nn.Linear(n1, n2, bias=bias)

    if actv is None:
        return linear
    return nn.Sequential(linear, build_actv(actv))


class Linear(nn.Module):
    def __init__(s,
        n1,
        n2,
        bias=True,
        bias_init=None,
        scale_w=False,
        lr_mult=None,
    ):
        super().__init__()
        if lr_mult is None:
            lr_mult = 1.

        s._weight = nn.Parameter(torch.randn([n2, n1]) / lr_mult)
        s._bias = nn.Parameter(torch.randn([n2])) if bias else None
        s._bias_init = bias_init

        s._weight_mult = lr_mult
        if scale_w:
            s._weight_mult /= np.sqrt(n1)
        s._bias_mult = lr_mult

    def init_params(s):
        nn.init.normal_(s._weight)
        if s._bias is not None:
            if s._bias_init is None:
                nn.init.normal_(s._bias)
            else:
                nn.init.constant_(s._bias, s._bias_init)

    def forward(s, x):
        w = s._weight * s._weight_mult
        if s._bias is None:
            return F.linear(x, w)
        return F.linear(x, w, bias=s._bias * s._bias_mult)