import chess


class Counter:
    def __init__(s):
        s._val = 0

    def get(s):
        return s._val

    def get_and_inc(s):
        ret = s._val
        s._val += 1
        return ret


class RelActionMap:
    def __init__(s, max_delta=7):
        s._max_delta = max_delta
        n = max_delta * 2 + 1
        center = max_delta
        s._to_action = [[None] * n for _ in range(n)]
        s._from_action = {}
        s._count = Counter()

        # horizontal, vertical, and diagonal moves
        for i in range(n):
            if i == center:
                continue
            s._setup(i, center)
            s._setup(center, i)
            s._setup(i, i)
            s._setup((n-1)-i, i)

        # knight moves
        for dx in [-1, 1]:
            for dy in [-2, 2]:
                s._setup(center+dx, center+dy)
        for dy in [-1, 1]:
            for dx in [-2, 2]:
                s._setup(center+dx, center+dy)

        # underpromotions
        s._to_up_action = {}
        s._from_up_action = {}
        for piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
            s._to_up_action[piece] = (
                s._count.get_and_inc(), # promote left
                s._count.get_and_inc(), # promote center
                s._count.get_and_inc(), # promote right
            )
            s._from_up_action[s._to_up_action[piece][0]] = (-1, piece)
            s._from_up_action[s._to_up_action[piece][1]] = (0, piece)
            s._from_up_action[s._to_up_action[piece][2]] = (1, piece)

        s._n_actions = s._count.get()
        del s._count

    def __len__(s):
        return s._n_actions

    def to_action(s, dx, dy, promotion, allow_invalid=False):
        if promotion is None or promotion == chess.QUEEN:
            x, y = dx + s._max_delta, dy + s._max_delta
            action = s._to_action[x][y]
        else: # underpromotion
            assert dx in [-1, 0, 1]
            assert dy == -1
            action = s._to_up_action[promotion][dx + 1]

        if not allow_invalid:
            assert action is not None

        return action

    def from_action(s, action):
        if action in s._from_action:
            dx, dy = s._from_action[action]
            return dx, dy, None
        else:
            dx, underpromo = s._from_up_action[action]
            return dx, -1, underpromo

    def _setup(s, dx, dy):
        action = s._count.get_and_inc()
        s._to_action[dx][dy] = action
        s._from_action[action] = (
            dx - s._max_delta,
            dy - s._max_delta,
        )
