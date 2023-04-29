import chess
import chess.pgn
import io


GAME_SEPARATOR = '[Event "'


def pgn_splitter(f):
    lines = []
    for line in f:
        if line.startswith(GAME_SEPARATOR) and lines:
            yield ''.join(lines)
            lines = []
        lines.append(line)
    yield ''.join(lines)


def pgn_to_game(pgn):
    game = chess.pgn.read_game(io.StringIO(pgn))
    if len(game.errors):
        raise Exception(str(game.errors))
    return game
