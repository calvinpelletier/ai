from pathlib import Path
import sys

from ai.examples.chess.data.pgn import pgn_splitter, pgn_to_game
from ai.game.chess.action import move_to_action, action_to_move
from ai.game.chess.board import neural_to_board, board_to_neural
from ai.game.chess.util import eq_boards


TEST_DATA = Path(__file__).parent / 'data'


def test_move_action_conversion():
    pgns = []
    with open(TEST_DATA / 'underpromo.pgn', 'r') as f:
        pgns.append(f.read())
    with open(TEST_DATA / 'lichess_medium.pgn', 'r') as f:
        for pgn in pgn_splitter(f):
            pgns.append(pgn)

    for pgn in pgns:
        game = pgn_to_game(pgn)
        for position in game.mainline():
            if position.parent is None:
                continue
            gt_move = position.move
            board = position.parent.board()
            player = 1 if position.parent.turn() else -1
            action = move_to_action(gt_move, player)
            move = action_to_move(action, player, board)
            assert move == gt_move


def test_board_state_conversion():
    pgns = []
    with open(TEST_DATA / 'lichess_medium.pgn', 'r') as f:
        for pgn in pgn_splitter(f):
            pgns.append(pgn)

    for pgn in pgns:
        game = pgn_to_game(pgn)
        for position in game.mainline():
            gt_board = position.board()
            player = 1 if position.turn() else -1
            neural = board_to_neural(gt_board, player)
            board = neural_to_board(neural, player)
            assert eq_boards(board, gt_board)


def test_mirror():
    with open(TEST_DATA / 'mirror/a.pgn', 'r') as f:
        a = pgn_to_game(f.read())
    with open(TEST_DATA / 'mirror/b.pgn', 'r') as f:
        b = pgn_to_game(f.read())

    a = a.next().next().next()
    b = b.next().next().next().next()
    p1 = 1 if a.parent.turn() else -1
    p2 = 1 if b.parent.turn() else -1
    assert p1 == 1
    assert p2 == -1

    while a is not None and b is not None:
        action1 = move_to_action(a.move, p1)
        action2 = move_to_action(b.move, p2)
        assert action1 == action2

        enc1 = board_to_neural(a.parent.board(), p1)
        enc2 = board_to_neural(b.parent.board(), p2)
        assert (enc1 == enc2).all()

        a = a.next()
        b = b.next()
        p1 = -p1
        p2 = -p2
    assert a is None and b is None
