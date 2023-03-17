# type: ignore
import torch
from numpy import sqrt

import ai.model as m
from ai.util import log2_diff


# generator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Generator(m.Model):
    def __init__(s,
        imsize,
        z_dim=512,
        nc_min=32,
        nc_max=512,
        f_n_layers=8,
        clamp=256,
    ):
        super().__init__()
        s.imsize = imsize
        s.z_dim = z_dim

        # disentangled latent vector (w) -> img
        s.g = SynthesisNetwork(imsize, z_dim, nc_min, nc_max, clamp)

        # latent vector (z) -> disentangled latent vector (w)
        s.f = MappingNetwork(z_dim, len(s.g.blocks) * 2, n_layers=f_n_layers)

    def forward(s, z, trunc=1):
        w = s.f(z, trunc=trunc)
        return s.g(w)


class SynthesisNetwork(m.Module):
    def __init__(s, imsize, z_dim, nc_min=32, nc_max=512, clamp=256):
        super().__init__()
        n = log2_diff(4, imsize)
        nc = [min(nc_min*2**i, nc_max) for i in range(n, -1, -1)]
        s.blocks = m.modules([
            SynthesisBlock(nc[i], nc[i+1], z_dim, clamp)
            for i in range(n)
        ])
        s.initial = torch.nn.Parameter(torch.randn([nc[0], 4, 4]))

    def forward(s, ws):
        x = s.initial.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        img = None
        for i, block in enumerate(s.blocks):
            x, img = block(x, img, ws.narrow(1, 2*i, 2))
        return img


class SynthesisBlock(m.Module):
    def __init__(s, nc1, nc2, z_dim, clamp):
        super().__init__()
        kw = {
            'modtype': 'weight',
            'actv': 'lrelu',
            'clamp': clamp,
            'scale_w': True,
            'gain': sqrt(2),
        }
        s.conv0 = m.modconv(nc1, nc2, z_dim, stride=.5, noise=True, **kw)
        s.conv1 = m.modconv(nc2, nc2, z_dim, noise=True, **kw)
        s.to_rgb = m.modconv(nc2, 3, z_dim, **kw)

    def forward(s, x, img, ws):
        x = s.conv0(x, ws[:, 0, :])
        x = s.conv1(x, ws[:, 1, :])
        y = s.to_rgb(x, ws[:, 1, :])
        img = y if img is None else y + m.f.resample(img, 2)
        return x, img


class MappingNetwork(m.Module):
    def __init__(s, z_dim, num_ws, n_layers=8, actv='lrelu', lr_mult=0.01):
        super().__init__()
        s.num_ws = num_ws
        s.register_buffer('w_ema', torch.zeros([z_dim]))
        s.net = m.repeat(
            n_layers,
            m.fc(z_dim, z_dim, actv=actv, lr_mult=lr_mult, scale_w=True),
        )

    def forward(s, z, trunc=1, update_w_ema=True):
        w = s.net(m.f.normalize_2nd_moment(z))

        if s.training and update_w_ema:
            s.w_ema.copy_(w.detach().mean(dim=0).lerp(s.w_ema, 0.995))

        ws = w.unsqueeze(1).repeat([1, s.num_ws, 1])

        if trunc != 1:
            ws = s.w_ema.lerp(ws, trunc)

        return ws
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# discriminator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Discriminator(m.Model):
    def __init__(s, imsize, nc_min=32, nc_max=512, clamp=256, smallest=4):
        initial = m.conv(3, nc_min, k=1, actv='lrelu', clamp=clamp,
            scale_w=True, gain=sqrt(2))

        main = m.pyramid(
            imsize, smallest,
            nc_min, nc_max,
            lambda _, nc1, nc2: discrim_block(nc1, nc2, clamp),
        )

        final = m.fm2v.mbstd(smallest, main.nc_out, clamp=clamp)

        super().__init__(m.seq(initial, main, final))


def discrim_block(nc1, nc2, clamp=256):
    all = {'scale_w': True}
    main = {'actv': 'lrelu'}
    shortcut = {'k': 1, 'bias': False}
    down = {'stride': 2, 'blur': True, 'clamp': clamp * sqrt(0.5)}
    flat = {'clamp': clamp}
    return m.res(
        m.seq(
            m.conv(nc1, nc1, **all, **main, **flat, gain=sqrt(2)),
            m.conv(nc1, nc2, **all, **main, **down), # gain=sqrt(2)*sqrt(.5)=1
        ),
        m.conv(nc1, nc2, **all, **shortcut, **down, gain=sqrt(0.5)),
    )
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
