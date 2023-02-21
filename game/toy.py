import numpy as np

from ai.game import Game2p


BOARD_TO_OUTCOME = {}
for board, outcome in [
    ([1, -1, 1], -1),
    ([1, -1, 2], 0),
    ([1, -2, 1], 1),
    ([1, -2, 2], 0),
    ([2, -1, 1], -1),
    ([2, -1, 2], 1),
    ([2, -2, 1], 0),
    ([2, -2, 2], 0),
]:
    BOARD_TO_OUTCOME[np.asarray(board, dtype=np.int8).tobytes()] = outcome


class ToyGame(Game2p):
    def __init__(s):
        super().__init__(n_actions=2, ob_shape=[3])
        s.reset()

    def reset(s):
        super().reset()
        s.board = np.zeros(3, dtype=np.int8)

    def step(s, action):
        assert s.outcome is None
        assert s.board[s.ply] == 0
        s.board[s.ply] = (action + 1) * s.to_play
        if s.ply == 2:
            s.outcome = BOARD_TO_OUTCOME[s.board.tobytes()]
        super().step(action)
        return s.outcome

    def observe(s, perspective=True):
        ob = np.copy(s.board)
        if perspective:
            ob *= s.to_play
        return ob

    def get_legal_actions(s):
        if s.ply > 2:
            return []
        return [0, 1]
