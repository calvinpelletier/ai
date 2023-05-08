import ai.model as m
from ai.game.chess import REL_ACTION_SIZE, BOARD_DEPTH

from ..data import META_DIM
from .convert import to_grid, to_vector, VectorToGrid
from .process import Processor


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DECODER
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ValueDecoder(m.Module):
    def __init__(s, cfg, c):
        super().__init__(m.seq(
            Processor(cfg, c.main),
            ValuePost(cfg, c.post),
        ))


class PolicyDecoder(m.Module):
    def __init__(s, cfg, c):
        super().__init__(m.seq(
            Processor(cfg, c.main),
            PolicyPost(cfg, c.post),
        ))


class ValuePolicyDecoder(m.Module):
    def __init__(s, cfg, c):
        super().__init__()
        s._value_decoder = ValueDecoder(cfg, c.value)
        s._policy_decoder = PolicyDecoder(cfg, c.policy)

    def forward(s, x):
        return s._value_decoder(x), s._policy_decoder(x)


class BoardDecoder(m.Module):
    def __init__(s, cfg, c):
        super().__init__(m.seq(
            VectorToGrid(cfg, c),
            BoardPost(cfg),
        ))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# POST
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ValuePost(m.Module):
    def __init__(s, cfg):
        super().__init__()
        s._net = m.fc(cfg.dim, 1, actv='tanh')

    def forward(s, x):
        return s._net(to_vector(x)).squeeze(1)


class _ActionPost(m.Module):
    def __init__(s, dim, actv):
        super().__init__()
        s._net = m.seq(
            m.conv(dim, REL_ACTION_SIZE, k=1, actv=actv),
            m.flatten(),
        )

    def forward(s, x):
        return s._net(to_grid(x))

class PolicyPost(_ActionPost):
    def __init__(s, cfg):
        super().__init__(cfg.dim, None)

class LegalPost(_ActionPost):
    def __init__(s, cfg):
        super().__init__(cfg.dim, 'sigmoid')


class MetaPost(m.Module):
    def __init__(s, cfg):
        super().__init__(m.fc(cfg.dim, META_DIM))


class BoardPost(m.Module):
    def __init__(s, cfg):
        super().__init__(m.conv(cfg.dim, BOARD_DEPTH, k=1))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
