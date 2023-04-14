import numpy as np

from ai.game.chess.action import move_to_action


RESULTS = {
    '1-0': 1,
    '0-1': -1,
    '1/2-1/2': 0,
}

class CompressedGame:
    def __init__(s, moves, times, meta):
        s.moves = moves
        s.times = times
        s.meta = meta

    @classmethod
    def from_chunk(cls, chunk, game_idx):
        meta = chunk.games[game_idx]
        start, end = meta[0], meta[1]
        return cls(
            moves=chunk.moves[start:end],
            times=chunk.times[start:end],
            meta=meta,
        )

    @classmethod
    def from_lichess_game(cls, game, max_len=None):
        winner = RESULTS[game.headers['Result']]
        white_elo, black_elo = map(
            lambda x: int(game.headers[x]),
            ['WhiteElo', 'BlackElo'],
        )
        tc_base, tc_inc = map(
            lambda x: int(x),
            game.headers['TimeControl'].split('+'),
        )

        moves = []
        times = []
        truncated = False
        clocks = [None, None]
        for state in game.mainline():
            if state.move is None:
                continue

            if max_len is not None and state.ply() > max_len:
                truncated = True
                break

            turn = int(state.turn())
            last_clock = clocks[turn]
            clock = state.clock()
            clock = int(clock) if clock is not None else last_clock
            times.append(
                tc_inc + last_clock - clock if last_clock is not None else 0)
            clocks[turn] = clock

            action = move_to_action(state.move, 1 if turn else -1)
            moves.append(action)

        return cls(
            moves=np.asarray(moves, dtype=np.uint16),
            times=np.asarray(times, dtype=np.uint16),
            meta=np.asarray([
                -1, # game start idx (if part of a chunk)
                -1, # game end idx
                winner,
                white_elo,
                black_elo,
                tc_base,
                tc_inc,
                int(truncated),
            ], dtype=np.int32),
        )

    def set_start_end_idxs(s, start, end):
        s.meta[0] = start
        s.meta[1] = end

    def decompress(s):
        # return Game(s.meta, s.moves, s.times)
        raise NotImplementedError('TODO')
