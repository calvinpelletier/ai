from pathlib import Path
import sys

from ai.data.chess.pgn import pgn_splitter, pgn_to_game


TEST_DATA = Path(__file__).parent / 'data'


def test_pgn_splitter():
    with open(TEST_DATA / 'lichess_small.pgn', 'rb') as f:
        _assert_pgns([pgn for pgn in pgn_splitter(f)])


def _assert_pgns(pgns):
    gts = []
    for i in range(3):
        with open(TEST_DATA / f'lichess_small_pgns/{i}.pgn', 'r') as f:
            gts.append(f.read())

    for pgn, gt in zip(pgns, gts):
        assert pgn.rstrip('\n') == gt.rstrip('\n')

    for pgn in pgns:
        game = pgn_to_game(pgn)


if __name__ == '__main__':
    _assert_pgns([pgn for pgn in pgn_splitter(sys.stdin.buffer)])
