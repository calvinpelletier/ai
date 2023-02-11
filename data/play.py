import numpy as np

from ai.data.datagen import RLDataGenerator


class SelfPlay(RLDataGenerator):
    def __init__(s, game, agent=None):
        super().__init__(agent)
        s._game = game

    def __next__(s):
        return play_self(s._game, s._agent)

def play_self(game, player):
    game.reset()
    replay = {'ob': [], 'pi': [], 'outcome': None}
    while game.outcome is None:
        replay['ob'].append(game.observe())
        action, pi = player.act(game, return_pi=True)
        replay['pi'].append(pi)
        game.step(action)
    replay['ob'] = np.asarray(replay['ob'])
    replay['pi'] = np.asarray(replay['pi'])
    replay['outcome'] = game.outcome
    return replay
