import ai
from ai.util.testing import *
from ai.examples.alphazero import ExampleConfig, AlphaZeroMLP


BS = 8


def test_self_play():
    cfg = ExampleConfig()
    cfg.device = DEVICE
    cfg.batch_size = BS
    cfg.data.n_workers = None
    cfg.buf_size = BS
    cfg.n_replay_times = 1
    cfg.infer.device = DEVICE
    cfg.player.mcts.n_sims = 2

    game = ai.game.TicTacToe()
    model = AlphaZeroMLP(game).init().to(DEVICE)
    player = ai.game.MctsPlayer(cfg.player, game, model)
    data = ai.data.SelfPlay.from_cfg(cfg, game, player)

    for i, batch in enumerate(data):
        assert_shape(batch['ob'], [BS, 1, 3, 3])
        assert_shape(batch['pi'], [BS, 9])
        assert_shape(batch['v'], [BS])
        assert_bounds(batch['ob'], [-1., 1.])
        assert_bounds(batch['pi'], [0., 1.])
        assert_bounds(batch['v'], [-1., 1.])
        if i > 8:
            break
