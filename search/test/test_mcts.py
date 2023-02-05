import torch
from torch import nn
import torch.nn.functional as F
import math

from ai.model import Model
from ai.game import TicTacToe, CartPole
from ai.search.mcts import MctsConfig, MonteCarloTreeSearch
from ai.examples.alphazero import AlphaZeroMLP
from ai.examples.muzero import MuZeroMLP


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
