'''modulated convolutions'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt

from ai.model.conv2d import conv, Noise
from ai.model.linear import fc
from ai.model.norm import AdaLIN
from ai.model.actv import build_actv
from ai.model.sequence import seq
from ai.model.etc import Clamp, blur


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
    '''modulated convolution

    input
        x : tensor[b, <nc1>, h, w]
        z : tensor[b, <z_dim>]
    output
        tensor[b, <nc2>, h*, w*] where h* = h / <stride>

    nc1 : int
        number of input channels
    nc2 : int
        number of output channels
    z_dim : int
        size of modulating vector
    modtype : str
        "adalin" (default)
            adaptive layer-instance normalization (see paper "U-GAT-IT:
            Unsupervised Generative Attentional Networks with Adaptive
            Layer-Instance Normalization for Image-to-Image Translation")
        "weight"
            weight modulation (see paper "Analyzing and Improving the Image
            Quality of StyleGAN")
    k : int
        kernel size
    stride : int or float
        for strides < 1 (i.e. an up convolution), the input is first resized
        by a scale factor of 1/stride before using a conv of stride=1
    actv : str or null
        activation (see model/actv.py)
    clamp : float or null
        clamp all output values between [-clamp, clamp]
    noise : bool
        add random noise [b, 1, h, w] of learnable magnitude
    padtype : str
        "zeros", "reflect", etc.
        TODO: support padtype!="zeros" for "weight" modtype
    scale_w : bool
        if enabled, scale conv weights by 1/sqrt(nc1 * k**2)
    lr_mult : float or None
        learning rate multiplier (scale conv weights and bias)
        TODO: support lr_mult for "weight" modtype
    '''

    if modtype == 'adalin':
        return NormModConv(AdaLIN, nc1, nc2, z_dim, k, stride, actv, clamp,
            noise, padtype, scale_w, lr_mult)

    elif modtype == 'weight':
        assert padtype == 'zeros', 'TODO'
        assert lr_mult is None, 'TODO'
        return WeightModConv(nc1, nc2, z_dim, k, stride, actv, clamp, noise,
            scale_w)

    raise NotImplementedError(f'modtype={modtype}')


class NormModConv(nn.Module):
    '''modulate a convolution by modulating the normalization that follows it

    input
        x : tensor[b, <nc1>, h, w]
        z : tensor[b, <z_dim>]
    output
        tensor[b, <nc2>, h*, w*] where h* = h / <stride>
    '''

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
        '''
        Norm : class
            norm = Norm(<nc2>, <z_dim>)
        (see modconv)
        '''

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
    '''modulated convolution from stylegan2

    input
        x : tensor[b, <nc1>, h, w]
        z : tensor[b, <z_dim>]
    output
        tensor[b, <nc2>, h*, w*] where h* = h / <stride>
    '''

    def __init__(s,
        nc1,
        nc2,
        z_dim,
        k=3,
        stride=1,
        actv='mish',
        clamp=None,
        noise=False,
        scale_w=True,
    ):
        '''
        nc1 : int
            number of input channels
        nc2 : int
            number of output channels
        z_dim : int
            size of modulating vector
        stride : int or float
            stride must be >=1 or ==0.5
            TODO: option for resampling instead of transposed conv
        actv : str or null
            activation (see model/actv.py)
        clamp : float or null
            clamp all output values between [-clamp, clamp]
        noise : bool
            add random noise [b, 1, h, w] of learnable magnitude
        scale_w : bool
            if enabled, scale conv weights by 1/sqrt(nc1 * k**2)
        '''

        super().__init__()
        assert stride >= 1 or stride == .5
        s._stride = stride

        s._fc = fc(z_dim, nc1, bias_init=1., scale_w=True)

        s._weight = nn.Parameter(torch.randn([nc2, nc1, k, k]))
        s._weight_mult = 1. / sqrt(nc1 * k**2) if scale_w else 1.
        s._bias = nn.Parameter(torch.zeros([nc2]))

        if s._stride < 1:
            s._blur = blur(up=1, pad=[1,1,1,1], gain=4.)

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

        # affine
        z = s._fc(z)

        # normalize
        weight = s._weight * s._weight_mult / s._weight.norm(
            float('inf'), dim=[1,2,3], keepdim=True)
        z = z / z.norm(float('inf'), dim=1, keepdim=True)

        # calc weights and demod coefficients
        w = weight.unsqueeze(0)
        w = w * z.reshape(bs, 1, -1, 1, 1)
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt()

        # pre conv scale
        x = x * z.reshape(bs, -1, 1, 1)

        # conv
        if s._stride < 1:
            x = F.conv_transpose2d(
                input=x,
                weight=weight.transpose(0, 1),
                bias=None,
                stride=2,
                padding=0,
            )
            x = s._blur(x)
        else:
            x = F.conv2d(
                input=x,
                weight=weight,
                bias=None,
                stride=s._stride,
                padding=1,
            )

        # post conv scale
        x = x * dcoefs.reshape(bs, -1, 1, 1)

        # bias
        x = x + s._bias.reshape(1, -1, 1, 1)

        # noise, actv, clamp
        return s._post(x)
