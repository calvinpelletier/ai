import numpy as np
from shutil import rmtree

from .compress import CompressedGame


class GameChunkWriter:
    def __init__(s, output_dir, chunk_size):
        if output_dir.exists():
            rmtree(output_dir)
        output_dir.mkdir(parents=True)
        s._output_dir = output_dir

        s._chunk_size = chunk_size
        s._cur_id = 0
        s._games = []
        s._actions = []
        s._times = []

    def add(s, compressed):
        start = len(s._actions)
        s._actions.extend(compressed.actions)
        s._times.extend(compressed.times)
        end = len(s._actions)
        s._games.append([start, end] + compressed.meta)

        if len(s._games) >= s._chunk_size:
            s.flush()

    def flush(s):
        if not len(s._games):
            return

        GameChunk(
            np.asarray(s._games, dtype=np.int32),
            np.asarray(s._actions, dtype=np.uint16),
            np.asarray(s._times, dtype=np.uint16),
        ).write(s._output_dir / str(s._cur_id))

        s._cur_id += 1
        s._games = []
        s._actions = []
        s._times = []


class GameChunk:
    def __init__(s, games, actions, times):
        s.games = games
        s.actions = actions
        s.times = times

    @classmethod
    def from_dir(cls, dir):
        return cls(
            games=np.load(dir / 'games.npy'),
            actions=np.load(dir / 'actions.npy'),
            times=np.load(dir / 'times.npy'),
        )

    def __getitem__(s, idx):
        game = s.games[idx]
        start, end, meta = game[0], game[1], game[2:]
        return CompressedGame(s.actions[start:end], s.times[start:end], meta)

    def write(s, dir):
        dir.mkdir(exist_ok=True)
        np.save(dir / 'games.npy', s.games)
        np.save(dir / 'actions.npy', s.actions)
        np.save(dir / 'times.npy', s.times)
