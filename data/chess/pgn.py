import chess
import chess.pgn
import io


GAME_SEPARATOR = '[Event "'
READ_SIZE = 4096


def pgn_splitter(f):
    buffer = ''
    bad_chunk = False
    while True:
        chunk = f.read(READ_SIZE)

        if not chunk:
            yield GAME_SEPARATOR + buffer
            break

        try:
            chunk = chunk.decode('utf-8')
        except Exception as e:
            print(f'[WARNING] bad chunk: {e}')
            bad_chunk = True
            buffer = ''
            continue

        buffer += chunk
        while True:
            try:
                part, buffer = buffer.split(GAME_SEPARATOR, 1)
            except ValueError:
                break
            if bad_chunk:
                bad_chunk = False
            elif part:
                yield GAME_SEPARATOR + part


def pgn_to_game(pgn):
    game = chess.pgn.read_game(io.StringIO(pgn))
    if len(game.errors):
        return None, game.errors[0]
    return game, None
