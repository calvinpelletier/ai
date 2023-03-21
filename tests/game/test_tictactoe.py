from ai.game import TicTacToe


def test_tictactoe():
    game = TicTacToe()
    assert game.n_actions == 9
    assert game.ob_shape == [1, 3, 3]
    
    assert (game.observe() == 0.).all()
    assert game.outcome is None
    legal_actions = game.get_legal_actions()
    assert len(legal_actions) == 9
    legal_actions = set(legal_actions)
    for i in range(9):
        assert i in legal_actions
    assert game.ply == 0
    assert game.to_play == 1
    
    game.step(0)

    assert game.observe()[0, 0, 0] == -1.
    assert game.observe(False)[0, 0, 0] == 1.
    assert game.outcome is None
    assert game.ply == 1
    assert game.to_play == -1
    assert len(game.get_legal_actions()) == 8
    assert 0 not in game.get_legal_actions()
    
    game.step(3)
    assert game.outcome is None
    game.step(1)
    assert game.outcome is None
    game.step(4)
    assert game.outcome is None
    game.step(2)
    assert game.outcome == 1

    game.reset()
    assert game.ply == 0
    assert game.to_play == 1
    assert len(game.get_legal_actions()) == 9
    assert game.outcome is None

    game.step(0)
    game.step(3)
    game.step(1)
    game.step(4)
    game.step(8)
    game.step(5)
    assert game.outcome == -1

    game.reset()
    game.step(0)
    game.step(2)
    game.step(1)
    game.step(3)
    game.step(6)
    game.step(4)
    game.step(5)
    game.step(7)
    game.step(8)
    assert game.outcome == 0

