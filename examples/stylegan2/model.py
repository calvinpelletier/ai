import torch

import ai.model as m
from ai.util import log2_diff


class Generator(m.Model):
    def __init__(s, imsize, z_dim=512, nc_min=32, nc_max=512, f_n_layers=8):
        super().__init__()

        # expose z_dim so external code can sample the latent space
        s.z_dim = z_dim

        # disentangled latent vector (w) -> img
        s.g = SynthesisNetwork(imsize, z_dim, nc_min, nc_max)

        # latent vector (z) -> disentangled latent vector (w)
        s.f = MappingNetwork(z_dim, len(s.g.blocks) * 2, n_layers=f_n_layers)

    def forward(s, z, trunc=1):
        w = s.f(z, trunc=trunc)
        return s.g(w)


class Discriminator(m.Model):
    def __init__(s, imsize, nc_min=32, nc_max=512):
        initial = m.conv(3, nc_min, actv='lrelu')

        main = m.pyramid(
            imsize, 4,
            nc_min, nc_max,
            lambda _, a, b: m.resblk(a, b, stride=2, norm=None, actv='lrelu'),
        )

        final = m.fm2v.mbstd(4, main.nc_out, 1)

        super().__init__(m.seq(initial, main, final))


class SynthesisNetwork(m.Module):
    def __init__(s, imsize, z_dim, nc_min=32, nc_max=512):
        super().__init__()
        n = log2_diff(4, imsize)
        nc = [min(nc_min*2**i, nc_max) for i in range(n, -1, -1)]
        s.blocks = m.modules([
            SynthesisBlock(nc[i], nc[i+1], z_dim)
            for i in range(n)
        ])
        s.initial = torch.nn.Parameter(torch.randn([nc[0], 4, 4]))

    def forward(s, ws):
        x = s.initial.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        img = None
        for i, block in enumerate(s.blocks):
            x, img = block(x, img, ws.narrow(1, 2*i, 2))
        return img


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


class SynthesisBlock(m.Module):
    def __init__(s, nc1, nc2, z_dim):
        super().__init__()
        s.conv0 = m.modconv(
            nc1, nc2, z_dim, stride=.5, noise=True, modtype='weight')
        s.conv1 = m.modconv(nc2, nc2, z_dim, noise=True, modtype='weight')
        s.to_rgb = m.modconv(nc2, 3, z_dim, modtype='weight')

    def forward(s, x, img, ws):
        x = s.conv0(x, ws[:, 0, :])
        x = s.conv1(x, ws[:, 1, :])
        y = s.to_rgb(x, ws[:, 1, :])
        img = y if img is None else y + m.f.resample(img, 0.5)
        return x, img
