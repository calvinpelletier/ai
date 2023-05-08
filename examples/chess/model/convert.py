import ai.model as m
from ai.game.chess import BOARD_H, BOARD_W

from .pos import PosifyGrid


class VectorToGrid(m.Module):
    def __init__(s, cfg, c):
        super().__init__()
        s._net = m.seq(
            PosifyGrid(cfg),
            m.repeat(c.n_layers, m.conv(cfg.dim, cfg.dim, k=1, actv='mish')),
        )

    def forward(s, x):
        return s._net(m.f.repeat(x, f'b c -> b c {BOARD_H} {BOARD_W}'))


def to_vector(x):
    if len(x.shape) == 2:
        return x
    elif len(x.shape) == 3:
        return torch.mean(x, 1)
    elif len(x.shape) == 4:
        return torch.mean(x, (2, 3))
    raise ValueError(len(x.shape))


def to_seq(x):
    if len(x.shape) == 3:
        return x
    elif len(x.shape) == 4:
        return m.f.rearrange(m.f.flatten(x, 2), 'b c l -> b l c')
    raise ValueError(len(x.shape))


def to_grid(x):
    if len(x.shape) == 3:
        return m.f.rearrange(x, 'b (h w) c -> b c h w', h=BOARD_H, w=BOARD_W)
    elif len(x.shape) == 4:
        return x
    raise ValueError(len(x.shape))
