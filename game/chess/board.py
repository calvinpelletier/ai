import chess
import numpy as np

from ai.game.chess.util import (square_to_coords, coords_to_square,
    perspective, color_to_side, side_to_color)


BOARD_W = 8
BOARD_H = 8
BOARD_AREA = BOARD_W * BOARD_H

BOARD_DEPTH = 2
PIECE_IDX = 0
CASTLE_EP_IDX = 1

BOARD_SHAPE = (BOARD_DEPTH, BOARD_H, BOARD_W)

EP_RANKS = [2, 5]


def board_to_neural(board, player):
    neural = np.zeros(BOARD_SHAPE, dtype=np.int8)

    for square, piece in board.piece_map().items():
        side = color_to_side(piece.color, player)
        x, y = square_to_coords(square, player)
        neural[PIECE_IDX, y, x] = piece.piece_type * side

    for color in [True, False]:
        y = perspective(0, player) if color else perspective(7, player)
        neural[CASTLE_EP_IDX, y, 7] = int(board.has_kingside_castling_rights(
            color))
        neural[CASTLE_EP_IDX, y, 0] = int(board.has_queenside_castling_rights(
            color))

    if board.has_legal_en_passant():
        x, y = square_to_coords(board.ep_square, player)
        neural[CASTLE_EP_IDX, y, x] = 1

    return neural


def neural_to_board(neural, player):
    board = chess.Board()
    board.clear()

    board.turn = player == 1

    for y in range(BOARD_H):
        for x in range(BOARD_W):
            piece = neural[PIECE_IDX, y, x]
            if piece:
                board.set_piece_at(
                    coords_to_square(x, y, player),
                    chess.Piece(
                        abs(piece),
                        side_to_color(1 if piece > 0 else -1, player),
                    ),
                )

    castle_fen = ''
    if neural[CASTLE_EP_IDX, perspective(0, player), 7]:
        castle_fen += 'K'
    if neural[CASTLE_EP_IDX, perspective(0, player), 0]:
        castle_fen += 'Q'
    if neural[CASTLE_EP_IDX, perspective(7, player), 7]:
        castle_fen += 'k'
    if neural[CASTLE_EP_IDX, perspective(7, player), 0]:
        castle_fen += 'q'
    board.set_castling_fen(castle_fen if castle_fen else '-')

    for y in EP_RANKS:
        for x in range(BOARD_W):
            if neural[CASTLE_EP_IDX, y, x]:
                assert board.ep_square is None
                board.ep_square = coords_to_square(x, y, player)

    return board
