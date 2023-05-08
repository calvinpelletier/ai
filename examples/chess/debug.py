from ai.util import assert_shape
from ai.game.chess import ACTION_SIZE, PIECE_IDX, CASTLE_EP_IDX
from ai.game.chess.action import REL_ACTION_MAP, relative_to_absolute_action


PIECE_MAP = {
    -6: 'k',
    6: 'K',
    -5: 'q',
    5: 'Q',
    -4: 'r',
    4: 'R',
    -3: 'b',
    3: 'B',
    -2: 'n',
    2: 'N',
    -1: 'p',
    1: 'P',
}


def print_board(board):
    assert_shape(board, [2, 8, 8])
    for y in range(8):
        vals = []
        for x in range(8):
            piece = board[PIECE_IDX, y, x].item()
            if piece == 0:
                vals.append('!' if board[CASTLE_EP_IDX, y, x].item() else '.')
            else:
                vals.append(PIECE_MAP[piece])
        print(''.join(vals))
    print('')


def print_legal(legal, x1, y1):
    assert_shape(legal, [ACTION_SIZE])
    for y2 in range(8):
        vals = []
        for x2 in range(8):
            dx, dy = x2 - x1, y2 - y1
            rel = REL_ACTION_MAP.to_action(dx, dy, None, allow_invalid=True)
            if rel is None:
                vals.append('xxx')
            else:
                action = relative_to_absolute_action(x1, y1, rel)
                vals.append(f'{legal[action]:.1f}')
        print(' '.join(vals))
        print('')
