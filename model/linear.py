'''fully connected layers'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from ai.model.actv import build_actv


def fc(
    n1: int,
    n2: int,
    actv: Optional[str] = None,
    bias: bool = True,
    bias_init: Optional[float] = None,
    scale_w: bool = False,
    lr_mult: Optional[float] = None,
    dropout: Optional[float] = None,
):
    '''Fully connected layer.

    INPUT
        tensor[b, <n1>]
    OUTPUT
        tensor[b, <n2>]

    operations in order:
        linear
            either:
                torch.nn.Linear
            or:
                a custom implementation that can handle learning rate
                multiplying, scaling weights, and initializing bias
        activation function
        dropout

    ARGS
        n1 : int
            input size
        n2 : int
            output size
        actv : str or null
            activation (see model/actv.py)
        bias : bool
            enable bias in linear op (default true)
        bias_init : float or null
            optional initial value for bias
        scale_w : bool
            if enabled, scale weights by 1/sqrt(n1)
        lr_mult : float or null
            learning rate multiplier (scale weights and bias)
        dropout : float or null
    '''

    if scale_w or lr_mult is not None or bias_init is not None:
        linear = Linear(n1, n2, bias, bias_init, scale_w, lr_mult)
    else:
        linear = nn.Linear(n1, n2, bias=bias)
    ops = [linear]

    if actv is not None:
        ops.append(build_actv(actv))

    if dropout is not None and dropout != 0.:
        ops.append(nn.Dropout(dropout))

    if len(ops) > 1:
        return nn.Sequential(*ops)
    return ops[0]


class Linear(nn.Module):
    def __init__(s,
        n1: int,
        n2: int,
        bias: bool = True,
        bias_init: Optional[float] = None,
        scale_w: bool = False,
        lr_mult: Optional[float] = None,
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
