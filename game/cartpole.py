import gymnasium as gym
import numpy as np

from ai.game import Game1p


class CartPole(Game1p):
    def __init__(s):
        super().__init__(
            n_actions=2,
            ob_shape=[1, 1, 4],
            outcome_bounds=(0, 475),
        )
        s.env = gym.make('CartPole-v1')
        s.reset()

    def reset(s):
        super().reset()
        s._observation, _ = s.env.reset()

    def step(s, action):
        s._observation, reward, done, _, _ = s.env.step(action)
        if done:
            s.outcome = reward
        super().step(action)
        return s.outcome

    def observe(s):
        print(s._observation)
        return s._observation.reshape(1, 1, 4)

    def get_legal_actions(s):
        return list(range(s.n_actions))
