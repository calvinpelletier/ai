import numpy as np


class GameChunk:
    def __init__(s, games, moves, times):
        s._games = games
        s._moves = moves
        s._times = times

    @classmethod
    def from_dir(cls, dir):
        return cls(
            games=np.load(dir / 'games.npy'),
            moves=np.load(dir / 'moves.npy'),
            times=np.load(dir / 'times.npy'),
        )

    def add(s, game):
        start = s._moves.shape[0]

        s._moves = np.concatenate((s._moves, game.moves))
        s._times = np.concatenate((s._times, game.times))

        end = s._moves.shape[0]
        game.set_start_end_idxs(start, end)
        s._games = np.append(
            s._games,
            np.expand_dims(game.meta, axis=0),
            axis=0,
        )

    def n_moves(s):
        return s._moves.shape[0]

    def n_games(s):
        return s._games.shape[0]

    def write(s, dir):
        os.makedirs(dir, exist_ok=True)
        np.save(dir / 'games.npy', s._games)
        np.save(dir / 'moves.npy', s._moves)
        np.save(dir / 'times.npy', s._times)


class GameChunkWriter:
    def __init__(s, output_dir, min_chunk_size):
        output_dir.mkdir(parents=True, exist_ok=True)
        s._output_dir = output_dir
        s._min_chunk_size = min_chunk_size
        s._cur_chunk = None
        s._cur_id = 0

    def add(s, game):
        if s._cur_chunk is None:
            game.set_start_end_idxs(0, game.moves.shape[0])
            s._cur_chunk = GameChunk(
                games=np.expand_dims(game.meta, axis=0),
                moves=np.copy(game.moves),
                times=np.copy(game.times),
            )
        else:
            s._cur_chunk.add(game)

        if s._cur_chunk.n_moves() >= s._min_chunk_size:
            s.flush()

    def flush(s):
        if not s._cur_chunk.n_moves():
            return
        s._cur_chunk.write(s._output_dir / f'{s._cur_id:03d}')
        s._cur_chunk = None
        s._cur_id += 1
