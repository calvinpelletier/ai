from ai.util import no_op


class TrainLog:
    def __init__(s):
        s._fn = no_op

    def setup(s, fn):
        s._fn = fn

    def __call__(s, *a, **kw):
        s._fn(*a, **kw)

log = TrainLog()
