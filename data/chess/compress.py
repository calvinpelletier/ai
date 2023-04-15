import numpy as np
import chess
import chess.pgn

from ai.game.chess.action import move_to_action, action_to_move


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
        game = chunk.games[game_idx]
        start, end, meta = game[0], game[1], game[2:]
        return cls(
            moves=chunk.moves[start:end],
            times=chunk.times[start:end],
            meta=meta,
        )

    @classmethod
    def from_lichess(cls, game, max_len=None):
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

            turn = int(state.parent.turn())
            last_clock = clocks[turn]
            clock = state.clock()
            clock = int(clock) if clock is not None else last_clock
            times.append(
                tc_inc + last_clock - clock if last_clock is not None else 0)
            clocks[turn] = clock

            action = move_to_action(state.move, 1 if turn else -1)
            moves.append(action)

        return cls(
            moves=moves,
            times=times,
            meta=[
                winner,
                white_elo,
                black_elo,
                tc_base,
                tc_inc,
                int(truncated),
            ],
        )

    def decompress(s):
        game = chess.pgn.Game()
        node = game
        for action in s.moves:
            player = 1 if node.turn() else -1
            move = action_to_move(action, player, node.board())
            node = node.add_variation(move)
        return game
