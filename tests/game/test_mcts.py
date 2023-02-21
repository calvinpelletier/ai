import torch

from ai.game import TicTacToe, CartPole, MctsPlayer
from ai.game.mcts import MctsConfig, MonteCarloTreeSearch
from ai.examples.alphazero import AlphaZeroMLP, ExampleConfig
from ai.examples.muzero import MuZeroMLP
from ai.task import GameTask
from ai.config import Config


def test_mcts():
    game = TicTacToe()

    cfg = ExampleConfig().player
    cfg.mcts.n_sims = 32
    player = MctsPlayer(cfg, game, _Model())

    task = GameTask(game, player, 128)
    win_rate_vs_random, _ = task()
    assert win_rate_vs_random > .7


def test_alphazero_tictactoe_mcts():
    cfg = ExampleConfig().player
    cfg.mcts.n_sims = 8

    game = TicTacToe()
    model = AlphaZeroMLP(game).init()
    player = MctsPlayer(cfg, game, model)
    mcts = MonteCarloTreeSearch(cfg.mcts, player, game.outcome_bounds)

    _run(mcts, game)


def test_muzero_tictactoe_mcts():
    cfg = ExampleConfig().player
    cfg.mcts.modeled_env = True
    cfg.mcts.n_sims = 8

    game = TicTacToe()
    model = MuZeroMLP(game).init()
    player = MctsPlayer(cfg, game, model)
    mcts = MonteCarloTreeSearch(cfg.mcts, player, game.outcome_bounds)

    _run(mcts, game)


def test_muzero_cartpole_mcts():
    cfg = ExampleConfig().player
    cfg.mcts.modeled_env = True
    cfg.mcts.n_sims = 8
    cfg.mcts.intermediate_rewards = True
    cfg.mcts.discount = .997

    game = CartPole()
    model = MuZeroMLP(game).init()
    player = MctsPlayer(cfg, game, model)
    mcts = MonteCarloTreeSearch(cfg.mcts, player)

    _run(mcts, game)


class _Model:
    def __call__(s, _):
        return {
            'pi': torch.ones(1, 9),
            'v': torch.zeros(1),
        }

    def get_device(s):
        return 'cpu'


def _run(mcts, game):
    with torch.no_grad():
        actions, counts = mcts.run(game)

    assert len(actions) > 0
    assert len(actions) == len(counts)


def _log(step, k, v):
    print(f'{k}: {v:.2f}')
