from typing import List, Union
import numpy as np


class Game:
    def __init__(s,
        n_players: int,
        n_actions: int,
        ob_shape: List[int],
        outcome_bounds: Union[list, tuple] = (-1, 1),
    ):
        assert len(outcome_bounds) == 2
        s.n_players = n_players
        s.n_actions = n_actions
        s.ob_shape = ob_shape
        s.outcome_bounds = outcome_bounds

        s.history = []
        s.outcome = None
        s.to_play = 1

    @property
    def ply(s) -> int:
        return len(s.history)

    def reset(s):
        s.history = []
        s.outcome = None
        s.to_play = 1

    def step(s, action: int):
        s.history.append(action)

    def observe(s) -> np.ndarray:
        raise NotImplementedError()

    def get_legal_actions(s) -> List[int]:
        raise NotImplementedError()

    def as_gameinfo(s) -> 'GameInfo':
        return GameInfo(s.n_actions, s.history, s.to_play)


class Game1p(Game):
    def __init__(s,
        n_actions: int,
        ob_shape: List[int],
        outcome_bounds: Union[list, tuple] = (-1, 1),
    ):
        super().__init__(1, n_actions, ob_shape, outcome_bounds)

    def as_gameinfo(s) -> 'Game1pInfo':
        return Game1pInfo(s.n_actions, s.history)


class Game2p(Game):
    PLAYER1 = 1
    PLAYER2 = -1

    def __init__(s,
        n_actions: int,
        ob_shape: List[int],
        outcome_bounds: Union[list, tuple] = (-1, 1),
    ):
        super().__init__(2, n_actions, ob_shape, outcome_bounds)

    def step(s, action: int):
        super().step(action)
        s.to_play *= -1

    def observe(s, perspective: bool = True) -> np.ndarray:
        raise NotImplementedError()

    def as_gameinfo(s) -> 'Game2pInfo':
        return Game2pInfo(s.n_actions, s.history, s.to_play)


class GameInfo:
    def __init__(s, n_actions: int, history: List[int], to_play: int):
        s.n_actions = n_actions
        s.history = history
        s.to_play = to_play

    @property
    def ply(s) -> int:
        return len(s.history)

    def step(s, action: int):
        s.history.append(action)


class Game1pInfo(GameInfo):
    def __init__(s, n_actions: int, history: List[int]):
        super().__init__(n_actions, history, 1)


class Game2pInfo(GameInfo):
    def step(s, action: int):
        super().step(action)
        s.to_play *= -1
