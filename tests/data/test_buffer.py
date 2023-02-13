from ai.util.testing import *
from ai.game import TicTacToe
from ai.data.buffer import ReplayBuffer
from ai.infer import Inferencer
from ai.data.play import SelfPlay
from ai.examples.alphazero import AlphaZeroPlayer, AlphaZeroMLP


def test_replay_buf():
    game = TicTacToe()

    model = AlphaZeroMLP(game).init()
    inferencer = Inferencer(model, device=DEVICE, batch_size=8)
    player = AlphaZeroPlayer(
        AlphaZeroPlayer.TEST, game, inferencer.create_client())

    buf = ReplayBuffer(
        SelfPlay(game, player),
        device=DEVICE,
        batch_size=4,
        n_data_workers=8,
        buf_size=8,
        n_replay_times=2,
    )

    for i, batch in enumerate(buf):
        assert_shape(batch['ob'], [4, 1, 3, 3])
        assert_shape(batch['pi'], [4, 9])
        assert_shape(batch['v'], [4])
        assert_bounds(batch['v'], [-1., 1.])
        if i > 8:
            break
