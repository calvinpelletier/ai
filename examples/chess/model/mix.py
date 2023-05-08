import ai.model as m

from .convert import to_grid
from .modulate import ModResnet, ModTx, ModHybrid


class VectorsMixer(m.Module):
    def __init__(s, cfg, c):
        super().__init__()
        s._mix = c.mix
        if c.mix == 'add':
            s._net = m.repeat(c.n_layers, m.fc(cfg.dim, cfg.dim, actv='mish'))
        elif c.mix == 'cat':
            s._net = m.seq(
                m.fc(2 * cfg.dim, cfg.dim, actv='mish'),
                m.repeat(c.n_layers - 1, m.fc(cfg.dim, cfg.dim, actv='mish')),
            )
            raise ValueError(c.mix)

    def forward(s, z1, z2):
        if s._mix == 'add':
            z = z1 + z2
        else:
            z = s._joint(m.f.cat([z1, z2], dim=1))
        return s._main(z)


class GridVectorMixer(m.Module):
    def __init__(s, cfg, c):
        if c.type == 'simple':
            net = _SimpleGridVectorMixer(cfg, c)
        elif c.type == 'conv':
            net = ModResnet(cfg, c)
        elif c.type == 'tx':
            net = ModTx(cfg, c)
        elif c.type == 'hybrid':
            net = ModHybrid(cfg, c)
        else:
            raise ValueError(c.type)
        super().__init__(net)


class _SimpleGridVectorMixer(m.Module):
    def __init__(s, cfg, c):
        super().__init__()
        s._mix = c.mix
        kw = {'k': 1, 'actv': 'mish'}
        if c.mix == 'add':
            s._net = m.repeat(c.n_layers, m.conv(cfg.dim, cfg.dim, **kw))
        elif c.mix == 'cat':
            s._net = m.seq(
                m.conv(2 * cfg.dim, cfg.dim, **kw),
                m.repeat(c.n_layers - 1, m.conv(cfg.dim, cfg.dim, **kw)),
            )
        else:
            raise ValueError(c.mix)

    def forward(s, x, z):
        x = to_grid(x)
        if s._mix == 'add':
            x = x + m.f.rearrange(z, 'b d -> b d 1 1')
        else:
            x = m.f.cat([
                x,
                m.f.repeat(z, f'b d -> b d {BOARD_H} {BOARD_W}'),
            ], dim=1)
        return s._net(x)
