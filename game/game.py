

class Game:
    def __init__(s, n_players, n_actions, ob_shape, outcome_bounds=(-1, 1)):
        s.n_players = n_players
        s.n_actions = n_actions
        s.ob_shape = ob_shape
        s.outcome_bounds = outcome_bounds

        s.history = []
        s.outcome = None
        s.to_play = 1

    @property
    def ply(s):
        return len(s.history)

    def reset(s):
        s.history = []
        s.outcome = None
        s.to_play = 1

    def step(s, action):
        s.history.append(action)

    def observe(s):
        raise NotImplementedError()

    def get_legal_actions(s):
        raise NotImplementedError()

    def as_gameinfo(s):
        return GameInfo(s.n_actions, s.history, s.to_play, s.ply)


class Game1p(Game):
    def __init__(s, n_actions, ob_shape, outcome_bounds=(-1, 1)):
        super().__init__(1, n_actions, ob_shape, outcome_bounds)

    def as_gameinfo(s):
        return Game1pInfo(s.n_actions, s.history)


class Game2p(Game):
    def __init__(s, n_actions, ob_shape, outcome_bounds=(-1, 1)):
        super().__init__(2, n_actions, ob_shape, outcome_bounds)

    def step(s, action):
        super().step(action)
        s.to_play *= -1

    def as_gameinfo(s):
        return Game2pInfo(s.n_actions, s.history, s.to_play)


class GameInfo:
    def __init__(s, n_actions, history, to_play):
        s.n_actions = n_actions
        s.history = history
        s.to_play = to_play

    @property
    def ply(s):
        return len(s.history)

    def step(s, action):
        s.history.append(action)


class Game1pInfo(GameInfo):
    def __init__(s, n_actions, history):
        super().__init__(n_actions, history, 1)


class Game2pInfo(GameInfo):
    def step(s, action):
        super().step(action)
        s.to_play *= -1
