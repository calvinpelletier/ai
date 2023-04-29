from pathlib import Path

from ai.data.chess.pgn import pgn_splitter, pgn_to_game
from ai.data.chess.compress import CompressedGame
from ai.data.chess.lichess import compress_lichess_data
from ai.data.chess.chunk import GameChunk


TEST_DATA = Path(__file__).parent / 'data'
OUTPUT_DIR = Path('/tmp/ai/test/chess/compress/lichess')
CHUNK_SIZE = 4


def test_lichess_compressor():
    pgns = []
    with open(TEST_DATA / 'lichess_medium.pgn', 'r') as f:
        for pgn in pgn_splitter(f):
            pgns.append(pgn)

    counts = compress_lichess_data(
        OUTPUT_DIR,
        TEST_DATA / 'lichess_medium.pgn',
        valid_terminations=['Normal', 'Abandoned', 'Time forfeit'],
        min_time_control=0,
        max_time_control=float('inf'),
        min_game_len=0,
        max_game_len=None,
        chunk_size=CHUNK_SIZE,
    )
    assert len(pgns) == counts['total'] == counts['valid']

    chunks = sorted(OUTPUT_DIR.iterdir(), key=lambda x: int(x.name))
    n_chunks = len(chunks)
    assert n_chunks == 13
    for i, chunk in enumerate(chunks):
        chunk = GameChunk.from_dir(chunk)
        if i == n_chunks - 1:
            assert len(chunk.games) == len(pgns) % CHUNK_SIZE
        else:
            assert len(chunk.games) == CHUNK_SIZE

        for j in range(len(chunk.games)):
            assert _equal_games(
                pgn_to_game(pgns[i * CHUNK_SIZE + j]),
                chunk[j].decompress(),
            )


def test_lichess_filter():
    counts = compress_lichess_data(
        OUTPUT_DIR,
        TEST_DATA / 'lichess_medium.pgn',
        chunk_size=CHUNK_SIZE,
    )
    assert counts['min game len'] == 5
    assert counts['time control'] == 19
    assert counts['termination'] == 0
    assert counts['valid'] == 26
    assert counts['total'] == 50


def test_underpromo_compress():
    with open(TEST_DATA / 'underpromo.pgn', 'r') as f:
        pgn = f.read()
    a = pgn_to_game(pgn)
    a.headers['Result'] = '1-0'
    a.headers['WhiteElo'] = '1234'
    a.headers['BlackElo'] = '2468'
    a.headers['TimeControl'] = '60+0'
    a.headers['Termination'] = 'Normal'

    compressed = CompressedGame.from_lichess(a)
    b = compressed.decompress()

    assert _equal_games(a, b)
    assert compressed.meta[0] == 1
    assert compressed.meta[1] == 1234
    assert compressed.meta[2] == 2468
    assert compressed.meta[3] == 60
    assert compressed.meta[4] == 0
    assert compressed.meta[5] == 0


def _equal_games(a, b):
    a = a.next()
    b = b.next()
    while a is not None and b is not None:
        if a.move != b.move:
            return False
        a, b = a.next(), b.next()
    return a is None and b is None
