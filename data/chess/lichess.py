from fire import Fire
from collections import defaultdict
from pathlib import Path
import sys

from ai.data.chess.pgn import pgn_to_game, pgn_splitter
from ai.data.chess.chunk import GameChunkWriter
from ai.data.chess.compress import CompressedGame


# zstdcat lichess_db.pgn.zst | python lichess.py ...
def compress_lichess_data(
    output_path,
    input_path=None,
    n_games=None,
    valid_terminations=['Normal', 'Abandoned'],
    min_time_control=180,
    max_time_control=60 * 30,
    min_game_len=4,
    max_game_len=128,
    truncate=True,
    chunk_size=1_000_000,
):
    writer = GameChunkWriter(Path(output_path), chunk_size)

    game_filter = Filter(
        valid_terminations,
        min_time_control,
        max_time_control,
        min_game_len,
        max_game_len if not truncate else float('inf'),
    )

    f = sys.stdin.buffer if input_path is None else open(input_path, 'rb')

    counts = defaultdict(int)
    for pgn in pgn_splitter(f):
        counts['total'] += 1

        # convert pgn to game
        try:
            game = pgn_to_game(pgn)
        except Exception as e:
            counts[str(e)] += 1
            continue

        # check if we should skip
        reason = game_filter(game)
        if reason:
            counts[reason] += 1
            continue

        # compress game and add to chunk
        writer.add(CompressedGame.from_lichess(
            game,
            max_game_len if truncate else None,
        ))

        counts['valid'] += 1

    writer.flush()

    for k, v in sorted(counts.items(), key=lambda x: x[1]):
        print(f'{k}: {v}')

    return counts


class Filter:
    def __init__(s,
        valid_terminations=['Normal', 'Abandoned'],
        min_time_control=180,
        max_time_control=60 * 30,
        min_game_len=4,
        max_game_len=96,
    ):
        s._valid_terminations = set(valid_terminations)
        s._min_time_control = min_time_control
        s._max_time_control = max_time_control
        s._min_game_len = min_game_len
        s._max_game_len = max_game_len

    def __call__(s, game):
        if game.headers['Termination'] not in s._valid_terminations:
            return 'termination'

        tc = game.headers['TimeControl']
        if tc == '-': # correspondence
            return 'time control'
        base, increment = map(lambda x: int(x), tc.split('+'))
        if base < s._min_time_control or base > s._max_time_control:
            return 'time control'

        game_len = game.end().ply()
        if game_len < s._min_game_len:
            return 'min game len'
        if game_len > s._max_game_len:
            return 'max game len'

        return None


if __name__ == '__main__':
    Fire(compress_lichess_data)
