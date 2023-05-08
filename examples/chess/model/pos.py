import ai.model as m
from ai.game.chess import BOARD_AREA, BOARD_H, BOARD_W


class Posify(m.Module):
    def __init__(s, shape):
        super().__init__()
        s._shape = shape
        s.init_params()

    def init_params(s):
        s._pos_emb = m.param(*s._shape)

    def forward(s, x):
        return x + s._pos_emb

class PosifySeq(Posify):
    def __init__(s, cfg):
        super().__init__((1, BOARD_AREA, cfg.dim))

class PosifyGrid(Posify):
    def __init__(s, cfg):
        super().__init__((1, cfg.dim, BOARD_H, BOARD_W))
