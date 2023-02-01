import numpy as np

from ai.game import Game2p


class TicTacToe(Game2p):
    def __init__(s):
        super().__init__(n_actions=9, ob_shape=[1, 3, 3])
        s.reset()

    def reset(s):
        super().reset()
        s.board = np.zeros((3, 3), dtype=np.int8)

    def step(s, action):
        assert s.outcome is None

        # update board
        x, y = action_to_coords(action)
        assert s.board[y, x] == 0
        s.board[y, x] = s.to_play

        # check for win
        for slice in slices(s.board):
            if slice[0] != 0 and np.all(slice == slice[0]):
                s.outcome = slice[0]
                break

        # check for tie
        if s.outcome is None and 0 not in s.board:
            s.outcome = 0

        # next turn
        super().step(action)

        return s.outcome

    def observe(s):
        return np.expand_dims(np.copy(s.board), 0)

    def get_legal_actions(s):
        legal_actions = []
        for a in range(s.n_actions):
            x, y = action_to_coords(a)
            if s.board[y, x] == 0:
                legal_actions.append(a)
        return legal_actions

def action_to_coords(a):
    return a % 3, a // 3

def slices(board):
    for x in range(3):
        yield board[:, x]
    for y in range(3):
        yield board[y, :]
    yield board.diagonal()
    yield np.fliplr(board).diagonal()
