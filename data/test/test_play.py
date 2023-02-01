from ai.data.util import create_data_loader
from ai.data.play import SelfPlay
from ai.util import assert_shape, assert_bounds
from ai.game import TicTacToe
from ai.examples.alphazero import AlphaZeroPlayer, AlphaZeroMLP
from ai.config import Config


def test_self_play():
    game = TicTacToe()

    model = AlphaZeroMLP(game).init()
    player = AlphaZeroPlayer(AlphaZeroPlayer.TEST, game, model)

    data = create_data_loader(
        SelfPlay(game, player),
        batch_size=None,
        device='cpu',
        n_workers=1,
    )

    for i, batch in enumerate(data):
        game_len = batch['ob'].shape[0]
        assert_shape(batch['ob'], [game_len, 1, 3, 3])
        assert_shape(batch['pi'], [game_len, 9])
        assert_shape(batch['outcome'], [])
        assert_bounds(batch['outcome'], [-1., 1.])
        if i > 8:
            break
