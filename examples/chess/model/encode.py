import torch

import ai.model as m
from ai.game.chess import (
    PIECES_PER_SIDE,
    N_PIECE_VALUES,
    PIECE_IDX,
    CASTLE_EP_IDX,
    BOARD_DEPTH,
    ACTION_SIZE,
)

from ..data import META_DIM
from .mix import GridVectorMixer
from .pos import PosifyGrid
from .process import Processor
from .util import build


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOARD/META ENCODER
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BoardEncoder(m.Module):
    def __init__(s, cfg, c=None):
        if c is None: c = cfg
        super().__init__(m.seq(
            BoardPre(cfg, c.pre),
            Processor(cfg, c.main),
        ))


class MetaEncoder(m.Module):
    def __init__(s, cfg, c):
        super().__init__()
        s._net = m.seq(
            m.fc(META_DIM, cfg.dim, actv='mish'),
            m.repeat(c.n_layers - 1, m.fc(cfg.dim, cfg.dim, actv='mish')),
        )

    def forward(s, meta):
        return s._net(meta.to(torch.float32))


class BoardMetaEncoder(m.Module):
    def __init__(s, cfg, c):
        super().__init__()
        s._board_encoder = BoardEncoder(cfg, c.board)
        s._meta_encoder = MetaEncoder(cfg, c.meta)
        s._mixer = GridVectorMixer(cfg, c.mixer)

    def forward(s, board, meta):
        board = s._board_encoder(board)
        meta = s._meta_encoder(meta)
        return s._mixer(board, meta)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HISTORY ENCODER
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class HistoryEncoder(m.Module):
    def __init__(s, cfg, c=None):
        super().__init__()
        if c is None: c = cfg
        assert c.type == 'lstm', f'TODO: implement type: {c.type}'

        s._emb = m.embed(ACTION_SIZE + 1, cfg.dim)

        s._lstm = m.lstm(cfg.dim, cfg.dim, c.n_layers, batch_first=True)

        s._hidden = m.fc(c.n_layers * cfg.dim, cfg.dim, actv='mish')
        s._cell = m.fc(c.n_layers * cfg.dim, cfg.dim, actv='mish')
        s._final = m.fc(2 * cfg.dim, cfg.dim, actv='mish')

        s._state_shape = (c.n_layers, cfg.dim)
        s.init_params()

    def init_params(s):
        s._h0 = m.param(*s._state_shape)
        s._c0 = m.param(*s._state_shape)

    def forward(s, history, history_len):
        bs = history.shape[0]
        history = s._emb(history.to(torch.int32))
        packed = m.f.pack_padded_sequence(history, history_len)

        _, (hidden, cell) = s._lstm(packed, (
            m.f.repeat(s._h0, f'd h -> d {bs} h'),
            m.f.repeat(s._c0, f'd h -> d {bs} h'),
        ))

        hidden = s._hidden(m.f.rearrange(hidden, 'd n h -> n (d h)'))
        cell = s._cell(m.f.rearrange(cell, 'd n h -> n (d h)'))
        return s._final(m.f.cat([hidden, cell], dim=1))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOARD PRE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BoardPre(m.Module):
    def __init__(s, cfg, c):
        net = build({
            'emb': _EmbBoardPre,
            'fc': _FcBoardPre,
        }, cfg, c)

        if c.posify:
            net = m.seq(net, PosifyGrid(cfg))

        super().__init__(net)


class _EmbBoardPre(m.Module):
    def __init__(s, cfg, c):
        super().__init__()
        s._emb = m.embed(N_PIECE_VALUES, cfg.dim - 1)

    def forward(s, board):
        pieces = board[:, PIECE_IDX, :, :]
        pieces = s._emb((pieces + PIECES_PER_SIDE).to(torch.int32))
        pieces = m.f.rearrange(pieces, 'b h w c -> b c h w')

        castle_ep = board[:, CASTLE_EP_IDX, :, :]
        castle_ep = castle_ep.unsqueeze(dim=1).to(torch.float32)

        return m.f.cat([pieces, castle_ep], dim=1)


class _FcBoardPre(m.Module):
    def __init__(s, cfg, c):
        super().__init__()
        kw = {'k': 1, 'actv': 'mish'}
        s._net = m.seq(
            m.conv(BOARD_DEPTH, cfg.dim, **kw),
            m.repeat(c.n_layers - 1, m.conv(cfg.dim, cfg.dim, **kw)),
        )

    def forward(s, board):
        return s._net(board.to(torch.float32))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
