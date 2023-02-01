import torch
from torch import nn
import torch.nn.functional as F
import math

from ai.model import Model
from ai.game import TicTacToe, CartPole
from ai.search.mcts import MctsConfig, MonteCarloTreeSearch
from ai.examples.alphazero import AlphaZeroMLP


def test_alphazero_tictactoe_mcts():
    game = TicTacToe()
    model = AlphaZeroMLP(game).init()

    cfg = MctsConfig(
        modeled_env=False,
        n_sims=8,
        intermediate_rewards=False,
        discount=None,
    )
    mcts = MonteCarloTreeSearch(cfg, model, game.outcome_bounds)

    _run(mcts, game)


def test_muzero_tictactoe_mcts():
    game = TicTacToe()
    model = MuZeroMLP(game).init()

    cfg = MctsConfig(
        modeled_env=True,
        n_sims=8,
        intermediate_rewards=False,
        discount=None,
    )
    mcts = MonteCarloTreeSearch(cfg, model, game.outcome_bounds)

    _run(mcts, game)


def test_muzero_cartpole_mcts():
    game = CartPole()
    model = MuZeroMLP(game).init()

    cfg = MctsConfig(
        modeled_env=True,
        n_sims=8,
        intermediate_rewards=True,
        discount=.997,
    )
    mcts = MonteCarloTreeSearch(cfg, model)

    _run(mcts, game)


def _run(mcts, game):
    with torch.no_grad():
        actions, counts = mcts.run(game)

    assert len(actions) > 0
    assert len(actions) == len(counts)


class MuZeroMLP(Model):
    def __init__(s, game):
        super().__init__()
        s.representation_net = RepresentationNet(game.ob_shape)
        s.prediction_net = PredictionNet(game.n_actions)
        s.dynamics_net = DynamicsNet(game.n_actions)

    def forward(s, ob=None, state=None, action=None):
        if ob is not None:
            assert state is None and action is None
            return s.infer_initial(ob)
        else:
            assert state is not None and action is not None
            return s.infer_recurrent(state, action)

    def infer_initial(s, ob):
        state = s.representation_net(ob)
        pi, v = s.prediction_net(state)
        return {
            'pi': pi,
            'v': v,
            'r': torch.zeros_like(v),
            'state': state,
        }

    def infer_recurrent(s, state, action):
        new_state, r = s.dynamics_net(state, action)
        pi, v = s.prediction_net(new_state)
        return {
            'pi': pi,
            'v': v,
            'r': r,
            'state': new_state,
        }

class RepresentationNet(nn.Module):
    def __init__(s, ob_shape):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(math.prod(ob_shape), 64),
            nn.Sigmoid(),
        )

    def forward(s, ob):
        return s.net(torch.flatten(ob, 1))

class PredictionNet(nn.Module):
    def __init__(s, n_actions):
        super().__init__()
        s.policy_net = nn.Linear(64, n_actions)
        s.value_net = nn.Sequential(nn.Linear(64, 1), nn.Tanh()) # TODO

    def forward(s, state):
        return s.policy_net(state), s.value_net(state)

class DynamicsNet(nn.Module):
    def __init__(s, n_actions):
        super().__init__()
        s.n_actions = n_actions
        s.state_net = nn.Sequential(
            nn.Linear(64 + n_actions, 64),
            nn.Sigmoid(),
        )
        s.reward_net = nn.Sequential(nn.Linear(64, 1), nn.Tanh()) # TODO

    def forward(s, state, action):
        x = torch.cat((state, F.one_hot(action, s.n_actions)), dim=1)
        new_state = s.state_net(x)
        return new_state, s.reward_net(new_state)
