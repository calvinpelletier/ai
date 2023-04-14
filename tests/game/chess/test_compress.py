from pathlib import Path
import sys

from ai.data.chess.pgn import pgn_splitter, pgn_to_game
from ai.data.chess.compress import CompressedGame
from ai.data.chess.game import Game


TEST_DATA = Path(__file__).parent / 'data'


def test_underpromo_compress():
    with open(TEST_DATA / 'underpromo.pgn', 'r') as f:
        pgn = f.read()
    game = pgn_to_game(pgn)
    game.headers['Result'] = '1-0'
    game.headers['WhiteElo'] = '1234'
    game.headers['BlackElo'] = '1234'
    game.headers['TimeControl'] = '60+0'
    gt_game = Game.from_lichess(game)

    compressor = LichessCompressor(include_times=False)
    compressor.compress(gt_game)
    assert compressor.n_games() == 1

    game = compressor.decompress(0)
    assert game == gt_game


def test_lichess_compressor():
    pgns = []
    with open(TEST_DATA / 'lichess_medium.pgn', 'rb') as f:
        for pgn in pgn_splitter(f):
            pgns.append(pgn)

    compressor = LichessCompressor()
    for pgn in pgns:
        compressor.compress(Game.from_lichess(pgn_to_game(pgn)))
    assert compressor.n_games() == len(pgns)

    for i in range(len(pgns)):
        gt_game = Game.from_lichess(pgn_to_game(pgns[i]))
        game = compressor.decompress(i)
        assert game == gt_game

    compressor_trunc = LichessCompressor(max_game_len=64)
    for pgn in pgns:
        compressor_trunc.compress(Game.from_lichess(pgn_to_game(pgn)))
    assert compressor_trunc.n_games() == len(pgns)
    assert compressor_trunc.n_moves() < compressor.n_moves()

    has_truncated = False
    for i in range(len(pgns)):
        game = compressor_trunc.decompress(i)
        if game.truncated:
            has_truncated = True
            break
    assert has_truncated


def test_compress_lichess_data():
    counts = compress_lichess_data(
        TEST_DATA / 'lichess_small.pgn',
        TEST_DATA / 'lichess_small',
        use_filter=False,
    )
    assert counts['total'] == 3
    assert counts['valid'] == 3

    counts = import_lichess(
        TEST_DATA / 'lichess_medium.pgn',
        TEST_DATA / 'lichess_medium',
    )
    assert counts['total'] > 0
    assert counts['valid'] > 0
    assert counts['termination'] > 0
    assert counts['time control'] > 0
    assert counts['min game len'] > 0

def _equal(a, b):
    a = a.next()
    b = b.next()
    while a is not None and b is not None:
        if a.move != b.move:
            return False
        a, b = a.next(), b.next()
    return a is None and b is None
