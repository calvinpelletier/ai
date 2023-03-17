'''2d convolutions'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union

from ai.model.actv import build_actv
from ai.model.norm import build_conv2d_norm
from ai.model.etc import resample, Clamp, Blur, Gain


def conv(
    nc1: int,
    nc2: int,
    k: int = 3,
    stride: Union[int, float] = 1,
    actv: Optional[str] = None,
    norm: Optional[str] = None,
    gain: Optional[float] = None,
    clamp: Optional[float] = None,
    noise: bool = False,
    blur: bool = False,
    bias: bool = True,
    padtype: str = 'zeros',
    scale_w: bool = False,
    lr_mult: Optional[float] = None,
):
    '''A Conv2d operation and optional additional ops in sequence.

    INPUT
        tensor[b, <nc1>, h, w]
    OUTPUT
        tensor[b, <nc2>, h / <stride>, w / <stride>]

    operations in order:
        blur
        resample
            if stride < 1 or (k == 1 and stride != 1)
            NOTE: if resampling, stride for conv will be 1
        convolution
            either:
                torch.nn.Conv2d
            or:
                a custom implementation that can handle learning rate
                multipliers and scaling weights
        normalization
        noise
        activation function
        gain
        clamp

    ARGS
        nc1 : int
            number of input channels
        nc2 : int
            number of output channels
        k : int
            kernel size
        stride : int or float
            if stride < 1 or (k == 1 and stride != 1), the input is first
            resized by a scale factor of 1/stride before using a conv of
            stride=1.
            TODO: allow option for transposed convs instead
        actv : str or null
            activation (see model/actv.py)
        norm : str or null
            normalization (see model/norm.py)
        gain : float or null
            multiply by constant value
        clamp : float or null
            clamp all output values between [-clamp, clamp]
        noise : bool
            add noise [b, 1, h, w] multiplied by a learnable magnitude
            NOTE: this is not for preventing overfitting
            TODO: add option for a fixed magnitude
        blur : bool
            blur (similar to upfirn2d used by stylegan2) before all other ops
        bias : bool
            enable bias in conv (default true)
            NOTE: if norm has a learnable bias, the conv bias is disabled
            regardless
        padtype : str
            "zeros", "reflect", etc.
        scale_w : bool
            if enabled, scale conv weights by 1/sqrt(nc1 * k**2)
        lr_mult : float or None
            learning rate multiplier (scale conv weights and bias)
    '''

    actv = build_actv(actv)
    norm, norm_has_bias = build_conv2d_norm(norm, nc2)

    seq = []

    if blur:
        seq.append(Blur())

    if stride < 1 or (k == 1 and stride != 1):
        seq.append(resample(1 / stride))
        stride = 1

    if scale_w or lr_mult is not None:
        seq.append(Conv2d(
            nc1, nc2,
            k=k,
            stride=stride,
            padding=0 if blur else (k - 1) // 2,
            bias=bias and not norm_has_bias,
            scale_w=scale_w,
            lr_mult=lr_mult,
        ))
    else:
        seq.append(nn.Conv2d(
            nc1, nc2,
            kernel_size=k,
            stride=stride, # type: ignore
            padding=0 if blur else (k - 1) // 2,
            padding_mode=padtype,
            bias=bias and not norm_has_bias,
        ))

    if norm is not None:
        seq.append(norm)

    if noise:
        seq.append(Noise())

    if actv is not None:
        seq.append(actv)

    if gain is not None:
        seq.append(Gain(gain))

    if clamp is not None:
        seq.append(Clamp(clamp))

    if len(seq) == 1:
        return seq[0]
    return nn.Sequential(*seq)


class Noise(nn.Module):
    '''Add random noise to a feature map.

    ARGS
        mag : float or null
            noise multiplier. if null, it is learnable (initalized at 0)
    '''

    def __init__(s, mag=None):
        super().__init__()
        s._mag = nn.Parameter(torch.zeros([])) if mag is None else mag

    def forward(s, x):
        noise = torch.randn(
            [x.shape[0], 1, x.shape[2], x.shape[3]],
            device=x.device,
        )
        return x + s._mag * noise


class Conv2d(nn.Module):
    '''2d conv operation that can scale weights and bias.

    ARGS
        nc1 : int
            number of input channels
        nc2 : int
            number of output channels
        k : int
            kernel size
        stride : int
        bias : bool
            enable bias
        scale_w : bool
            if enabled, scale conv weights by 1/sqrt(nc1 * k**2)
        lr_mult : float or null
            learning rate multiplier
    '''

    def __init__(s,
        nc1: int,
        nc2: int,
        k: int = 3,
        stride: Union[int, float] = 1,
        padding: int = 0,
        bias: bool = True,
        scale_w: bool = False,
        lr_mult: Optional[float] = None,
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
        s._padding = padding

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
            stride=s._stride, # type: ignore
            padding=s._padding, # type: ignore
        )
