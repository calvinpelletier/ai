'''normalizations'''

import torch
from torch import nn

from ai.model.linear import fc
from ai.model.sequence import seq


CONV2D_NORMS = {
    'batch': (nn.BatchNorm2d, True),
    'instance': (nn.InstanceNorm2d, False),
}

def build_conv2d_norm(norm, nc):
    '''string-to-module for conv2d normalizations

    input
        tensor[b, <nc>, h, w]
    output
        tensor[b, <nc>, h, w]

    args
        norm : string
            see CONV2D_NORMS for possible values
        nc : int
            number of input channels
    '''
    if norm is None:
        return None, False
    cls, norm_has_bias = CONV2D_NORMS[norm]
    return cls(nc), norm_has_bias


class AdaLIN(nn.Module):
    '''adaptive layer-instance normalization

    see paper "U-GAT-IT: Unsupervised Generative Attentional Networks with
    Adaptive Layer-Instance Normalization for Image-to-Image Translation"

    input
        x : tensor[b, <nc>, h, w]
        z : tensor[b, <z_dim>]
    output
        tensor[b, <nc>, h, w]
    '''

    def __init__(s, nc, z_dim, eps=1e-5):
        '''
        nc : int
            number of input channels
        z_dim : int
            size of modulating vector
        eps : float
            epsilon
        '''

        super().__init__()
        s._eps = eps
        s._rho = nn.Parameter(torch.full([1, nc, 1, 1], 0.9))
        s._style_std = seq(fc(z_dim, nc, scale_w=True), PixelNorm())
        s._style_mean = seq(fc(z_dim, nc, scale_w=True), PixelNorm())

    def forward(s, input, style):
        bs = input.shape[0]

        gamma = 1 + s._style_std(style)
        beta = s._style_mean(style)

        in_mean = torch.mean(input, dim=[2, 3], keepdim=True)
        in_var = torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + s._eps)

        ln_mean = torch.mean(input, dim=[1, 2, 3], keepdim=True)
        ln_var = torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + s._eps)

        out = s._rho.expand(input.shape[0], -1, -1, -1) * out_in + \
            (1 - s._rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + \
            beta.unsqueeze(2).unsqueeze(3)
        return out

class PixelNorm(nn.Module):
    def forward(s, input):
        return input * torch.rsqrt(torch.mean(
            input ** 2,
            dim=1,
            keepdim=True,
        ) + 1e-8)
