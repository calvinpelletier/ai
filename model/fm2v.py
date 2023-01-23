import torch

from ai.model.linear import fc
from ai.model.conv2d import conv
from ai.model.sequence import seq
from ai.model.etc import flatten, global_avg


def simple(n_in, n_out, actv=None):
    return seq(
        global_avg(),
        flatten(),
        fc(n_in, n_out, actv=actv),
    )


def mbstd(
    res,
    nc,
    n_out,
    mbstd_group_size=4,
    mbstd_nc=1,
    actv='lrelu',
    final_actv=None,
):
    return seq(
        MinibatchStd(group_size=mbstd_group_size, nc=mbstd_nc),
        conv(nc + mbstd_nc, nc, actv=actv),
        flatten(),
        fc(nc * res**2, nc, actv=actv),
        fc(nc, n_out, actv=final_actv),
    )

class MinibatchStd(torch.nn.Module):
    def __init__(s, group_size=4, nc=1):
        super().__init__()
        s._group_size = group_size
        s._nc = nc

    def forward(s, x):
        N, C, H, W = x.shape
        G = torch.min(
            torch.as_tensor(s._group_size),
            torch.as_tensor(N),
        )
        F = s._nc
        c = C // F
        y = x.reshape(G, -1, F, c, H, W)
        y = y - y.mean(dim=0)
        y = y.square().mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2,3,4])
        y = y.reshape(-1, F, 1, 1)
        y = y.repeat(G, 1, H, W)
        return torch.cat([x, y], dim=1)
