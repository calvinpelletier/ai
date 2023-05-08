import ai.model as m
from ai.game.chess import BOARD_AREA, BOARD_H, BOARD_W

from .convert import to_seq, to_grid
from .pos import Posify


class ModTx(m.Module):
    def __init__(s, cfg, c):
        super().__init__()
        net = m.tx_enc(c.n_blocks, cfg.dim, c.n_heads, cfg.dim * 2)
        if hasattr(c, 'posify') and c.posify:
            s._net = m.seq(Posify((1, BOARD_AREA + 1, cfg.dim)), net)
        else:
            s._net = net

    def forward(s, x, z):
        x = to_seq(x)
        x = m.f.cat([x, m.f.rearrange(z, 'b d -> b 1 d')], dim=1)
        return s._net(x)[:, :BOARD_AREA, :]


class ModResnet(m.Module):
    def __init__(s, cfg, c):
        super().__init__()

        s._posify = None
        if hasattr(c, 'posify') and c.posify:
            s._posify = Posify((1, cfg.dim, BOARD_H, BOARD_W))

        s._net = m.repeat(
            c.n_blocks,
            m.modresblk(cfg.dim, cfg.dim, cfg.dim, c.modtype),
            modulated=True,
        )

    def forward(s, x, z):
        x = to_grid(x)
        if s._posify:
            x = s._posify(x)
        return s._net(x, z)


class ModHybrid(m.Module):
    def __init__(s, cfg, c):
        assert c.first in ['conv', 'tx']
        if c.first == 'conv':
            net = m.modseq(ModResnet(cfg, c.conv), ModTx(cfg, c.tx))
        else:
            net = m.modseq(ModTx(cfg, c.tx), ModResnet(cfg, c.conv))
        super().__init__(net)
