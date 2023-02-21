import numpy as np

from ai.game import Game2p


class Connect2(Game2p):
    def __init__(s):
        super().__init__(n_actions=4, ob_shape=[4])
        s.reset()

    def reset(s):
        super().reset()
        s.board = np.zeros(4, dtype=np.int8)

    def step(s, action):
        assert s.outcome is None

        # update board
        assert s.board[action] == 0
        s.board[action] = s.to_play

        # check for win
        for pair in pairs(s.board):
            if pair[0] != 0 and pair[0] == pair[1]:
                s.outcome = pair[0]
                break

        # check for tie
        if s.outcome is None and 0 not in s.board:
            s.outcome = 0

        # next turn
        super().step(action)

        return s.outcome

    def observe(s, perspective=True):
        ob = np.expand_dims(np.copy(s.board), 0)
        if perspective:
            ob *= s.to_play
        return ob

    def get_legal_actions(s):
        legal_actions = []
        for a in range(s.n_actions):
            if s.board[a] == 0:
                legal_actions.append(a)
        return legal_actions

def pairs(board):
    yield board[0:2]
    yield board[1:3]
    yield board[2:]
