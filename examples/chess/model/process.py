import ai.model as m

from .convert import to_seq, to_grid
from .pos import PosifyGrid, PosifySeq


class Processor(m.Module):
    def __init__(s, cfg, c):
        if c.type == 'none':
            net = m.null()
        elif c.type == 'conv':
            net = Resnet(cfg, c)
        elif c.type == 'tx':
            net = Tx(cfg, c)
        elif c.type == 'hybrid':
            if c.first == 'conv':
                net = m.seq(Resnet(cfg, c.conv), Tx(cfg, c.tx))
            elif c.first == 'tx':
                net = m.seq(Tx(cfg, c.tx), Resnet(cfg, c.conv))
            else:
                raise ValueError(c.first)
        else:
            raise ValueError(c.type)
        super().__init__(net)


class Resnet(m.Module):
    def __init__(s, cfg, c):
        super().__init__()
        net = m.repeat(c.n_blocks, m.resblk(cfg.dim, cfg.dim))
        if hasattr(c, 'posify') and c.posify:
            s._net = m.seq(PosifyGrid(cfg), net)
        else:
            s._net = net

    def forward(s, x):
        return s._net(to_grid(x))


class Tx(m.Module):
    def __init__(s, cfg, c):
        super().__init__()
        net = m.tx_enc(c.n_blocks, cfg.dim, c.n_heads, cfg.dim * 2)
        if hasattr(c, 'posify') and c.posify:
            s._net = m.seq(PosifySeq(cfg), net)
        else:
            s._net = net

    def forward(s, x):
        return s._net(to_seq(x))
