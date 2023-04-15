import numpy as np
from shutil import rmtree

from ai.data.chess.compress import CompressedGame


class GameChunkWriter:
    def __init__(s, output_dir, chunk_size):
        if output_dir.exists():
            rmtree(output_dir)
        output_dir.mkdir(parents=True)
        s._output_dir = output_dir

        s._chunk_size = chunk_size
        s._cur_id = 0
        s._games = []
        s._moves = []
        s._times = []

    def add(s, compressed):
        start = len(s._moves)
        s._moves.extend(compressed.moves)
        s._times.extend(compressed.times)
        end = len(s._moves)
        s._games.append([start, end] + compressed.meta)

        if len(s._games) >= s._chunk_size:
            s.flush()

    def flush(s):
        if not len(s._games):
            return

        GameChunk(
            np.asarray(s._games, dtype=np.int32),
            np.asarray(s._moves, dtype=np.uint16),
            np.asarray(s._times, dtype=np.uint16),
        ).write(s._output_dir / str(s._cur_id))

        s._cur_id += 1
        s._games = []
        s._moves = []
        s._times = []


class GameChunk:
    def __init__(s, games, moves, times):
        s.games = games
        s.moves = moves
        s.times = times

    @classmethod
    def from_dir(cls, dir):
        return cls(
            games=np.load(dir / 'games.npy'),
            moves=np.load(dir / 'moves.npy'),
            times=np.load(dir / 'times.npy'),
        )

    def __getitem__(s, idx):
        game = s.games[idx]
        start, end, meta = game[0], game[1], game[2:]
        return CompressedGame(s.moves[start:end], s.times[start:end], meta)

    def write(s, dir):
        dir.mkdir(exist_ok=True)
        np.save(dir / 'games.npy', s.games)
        np.save(dir / 'moves.npy', s.moves)
        np.save(dir / 'times.npy', s.times)
