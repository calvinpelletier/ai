import chess
import numpy as np

from ai.game.chess.util import (square_to_coords, coords_to_square,
    perspective, color_to_side, side_to_color)
from ai.game.chess.constants import BOARD_W, BOARD_H


EP_RANKS = [2, 5]

PIECE_ENCS = [
    [0, 0, 0], # empty
    [0, 0, 1], # pawn
    [0, 1, 0], # knight
    [0, 1, 1], # bishop
    [1, 0, 0], # rook
    [1, 0, 1], # queen
    [1, 1, 0], # king
]
ENC_TO_PIECE = {
    np.asarray(enc, dtype=np.float32).data.tobytes(): i
    for i, enc in enumerate(PIECE_ENCS)
}

PIECE_ENC_DIM = 3
BOARD_DIM = PIECE_ENC_DIM + 1 + 1 # piece type + piece side + castle/ep status
PIECE_SIDE_IDX = PIECE_ENC_DIM
CASTLE_EP_IDX = PIECE_SIDE_IDX + 1


def board_to_state(board, player):
    state = np.zeros((BOARD_DIM, BOARD_H, BOARD_W), dtype=np.float32)

    for square, piece in board.piece_map().items():
        side = color_to_side(piece.color, player)
        piece = PIECE_ENCS[piece.piece_type]
        x, y = square_to_coords(square, player)
        state[:, y, x] = piece + [side, 0]

    for color in [True, False]:
        y = perspective(0, player) if color else perspective(7, player)
        state[CASTLE_EP_IDX, y, 7] = int(board.has_kingside_castling_rights(
            color))
        state[CASTLE_EP_IDX, y, 0] = int(board.has_queenside_castling_rights(
            color))

    if board.has_legal_en_passant():
        x, y = square_to_coords(board.ep_square, player)
        state[CASTLE_EP_IDX, y, x] = 1

    return state


def state_to_board(state, player):
    board = chess.Board()
    board.clear()

    board.turn = player == 1

    for y in range(BOARD_H):
        for x in range(BOARD_W):
            enc = state[:, y, x]
            piece = ENC_TO_PIECE[enc[:PIECE_ENC_DIM].data.tobytes()]
            if piece:
                board.set_piece_at(
                    coords_to_square(x, y, player),
                    chess.Piece(
                        piece,
                        side_to_color(enc[PIECE_SIDE_IDX], player),
                    ),
                )

    castle_fen = ''
    if state[CASTLE_EP_IDX, perspective(0, player), 7]:
        castle_fen += 'K'
    if state[CASTLE_EP_IDX, perspective(0, player), 0]:
        castle_fen += 'Q'
    if state[CASTLE_EP_IDX, perspective(7, player), 7]:
        castle_fen += 'k'
    if state[CASTLE_EP_IDX, perspective(7, player), 0]:
        castle_fen += 'q'
    board.set_castling_fen(castle_fen if castle_fen else '-')

    for y in EP_RANKS:
        for x in range(BOARD_W):
            if state[CASTLE_EP_IDX, y, x]:
                assert board.ep_square is None
                board.ep_square = coords_to_square(x, y, player)

    return board
