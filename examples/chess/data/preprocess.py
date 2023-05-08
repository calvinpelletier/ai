import numpy as np
import chess

from ai.game.chess import (
    action_to_move,
    move_to_action,
    board_to_neural,
    ACTION_SIZE,
    BOARD_SHAPE,
)
from ai.util.path import dataset_path

from .meta import META_DIM


def preprocess_board(dir):
    games = np.load(dir / 'game.npy')
    actions = np.load(dir / 'action.npy')

    c, h, w = BOARD_SHAPE
    data = np.empty((len(actions), c, h, w), dtype=np.int8)
    for game in games:
        start, end = game[:2]
        board = chess.Board()
        for i in range(start, end):
            player = 1 if board.turn else -1
            data[i] = board_to_neural(board, player)
            board.push(action_to_move(actions[i], player, board))

    np.save(dir / 'board.npy', data)


def preprocess_legal(dir):
    games = np.load(dir / 'game.npy')
    actions = np.load(dir / 'action.npy')

    legal = np.zeros((len(actions), ACTION_SIZE), dtype=np.uint8)
    for game in games:
        start, end = game[:2]
        board = chess.Board()
        for i in range(start, end):
            player = 1 if board.turn else -1
            for move in board.legal_moves:
                legal[i, move_to_action(move, player)] = 1
            assert legal[i, actions[i]] == 1
            board.push(action_to_move(actions[i], player, board))

    np.save(dir / 'legal.npy', legal)


def preprocess_history(dir, max_length=127):
    games = np.load(dir / 'game.npy')
    actions = np.load(dir / 'action.npy')

    history = np.zeros((len(actions), max_length), dtype=np.uint16)
    history_len = np.empty(len(actions), dtype=np.uint8)
    for game in games:
        start, end = game[:2]
        for i in range(start, end):
            history_len[i] = i - start
            assert history_len[i] <= max_length
            if history_len[i] > 0:
                for j in range(history_len[i]):
                    history[i, j] = actions[start + j]
            else:
                history_len[i] = 1
                history[i, 0] = ACTION_SIZE

    np.save(dir / 'history.npy', history)
    np.save(dir / 'history_len.npy', history_len)


def preprocess_meta(dir):
    games = np.load(dir / 'game.npy')
    actions = np.load(dir / 'action.npy')

    data = np.empty((len(actions), META_DIM), dtype=np.int32)
    for game in games:
        start, end = game[:2]
        meta = game[3:7]
        for i in range(start, end):
            data[i, :] = meta

    np.save(dir / 'meta.npy', data)


def preprocess_value(dir):
    games = np.load(dir / 'game.npy')
    actions = np.load(dir / 'action.npy')

    value = np.empty(len(actions), dtype=np.int8)
    for game in games:
        start, end, outcome = game[:3]
        for i in range(start, end):
            value[i] = outcome * (1 if i % 2 == 0 else -1)

    np.save(dir / 'value.npy', value)


if __name__ == '__main__':
    dir = dataset_path('chess/test/10k')
    # preprocess_board(dir)
    # preprocess_legal(dir)
    # preprocess_history(dir)
    # preprocess_meta(dir)
    # preprocess_value(dir)
