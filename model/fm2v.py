'''converting features maps to vectors'''

import torch

from ai.model.linear import fc
from ai.model.conv2d import conv
from ai.model.sequence import seq
from ai.model.etc import flatten, global_avg
from ai.model.norm import MinibatchStd


def simple(nc_in, n_out, actv=None):
    '''feature map to vector via: global avg -> flatten -> fully connected

    input
        tensor[b, <nc_in>, h, w]
    output
        tensor[b, <n_out>]

    args
        nc_in : int
            number of input channels
        n_out : int
            output size
        actv : str or null
            activation (see model/actv.py)
    '''

    return seq(
        global_avg(),
        flatten(),
        fc(nc_in, n_out, actv=actv),
    )


def mbstd(
    res,
    nc,
    n_out=1,
    mbstd_group_size=4,
    mbstd_nc=1,
    actv='lrelu',
    final_actv=None,
    clamp=256,
    scale_w=True,
):
    '''feature map to vector commonly used by discriminators

    input
        tensor[b, <nc>, <res>, <res>]
    output
        tensor[b, <n_out>]

    args
        res : int
            input resolution [b, c, <res>, <res>]
        nc : int
            input channels [b, <nc>, h, w]
        n_out : int
            output size [b, <n_out>]
        mbstd_group_size : int
        mbstd_nc : int
        actv : str or null
            activation function used for intermediate steps
        final_actv : str or null
    '''

    return seq(
        MinibatchStd(group_size=mbstd_group_size, nc=mbstd_nc),
        conv(nc + mbstd_nc, nc, actv=actv, clamp=clamp, scale_w=scale_w),
        flatten(),
        fc(nc * res**2, nc, actv=actv, scale_w=scale_w),
        fc(nc, n_out, actv=final_actv, scale_w=scale_w),
    )
