from ai.util.testing import *
from ai.examples.alphazero.main import run, ExampleConfig
from ai.examples.alphazero.model import AlphaZeroMLP
from ai.game import TicTacToe


def test_alphazero():
    cfg = ExampleConfig()
    cfg.outpath = '/tmp/testing/a0'
    cfg.device = DEVICE
    cfg.train.steplimit = 10
    cfg.player.mcts.n_sims = 4
    cfg.batch_size = 8
    cfg.task.n_matches = 8
    
    game = TicTacToe()
    run(cfg, game, AlphaZeroMLP(game))
