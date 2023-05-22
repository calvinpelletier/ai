import chess
import numpy as np

from ai.game.chess.action_map import RelActionMap
from ai.game.chess.util import move_to_coords, coords_to_move
from ai.game.chess.board import BOARD_AREA, BOARD_W


REL_ACTION_MAP = RelActionMap()
REL_ACTION_SIZE = len(REL_ACTION_MAP)
ACTION_SIZE = BOARD_AREA * REL_ACTION_SIZE


def move_to_action(move, player):
    x1, y1, x2, y2 = move_to_coords(move, player)
    dx, dy = x2 - x1, y2 - y1
    rel_action = REL_ACTION_MAP.to_action(dx, dy, move.promotion)
    return relative_to_absolute_action(x1, y1, rel_action)

def action_to_move(action, board):
    player = 1 if board.turn else -1
    x1, y1, rel_action = absolute_to_relative_action(action)
    dx, dy, underpromo = REL_ACTION_MAP.from_action(rel_action)
    x2, y2 = x1 + dx, y1 + dy
    return coords_to_move(board, player, x1, y1, x2, y2, underpromo)


def legal_mask(board):
    player = 1 if board.turn else -1
    ret = np.zeros(ACTION_SIZE, dtype=np.uint8)
    for move in board.legal_moves:
        ret[move_to_action(move, player)] = 1
    return ret


def relative_to_absolute_action(x1, y1, rel_action):
    return rel_action * BOARD_AREA + y1 * BOARD_W + x1

def absolute_to_relative_action(action):
    x1 = action % BOARD_W
    y1 = (action // BOARD_W) % BOARD_W
    rel_action = action // BOARD_AREA
    return x1, y1, rel_action
