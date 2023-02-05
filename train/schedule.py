from collections import namedtuple


class Logarithmic:
    def __init__(s, start=2**10, end=2**16, mult=2):
        s.start = start
        s.end = end
        s.mult = mult


class Schedule:
    def __init__(s, cfg):
        if isinstance(cfg, int):
            s.freq = cfg
            s.mult = None
        else:
            assert isinstance(cfg, Logarithmic), 'unknown schedule type'
            s.freq = cfg.start
            s.end = cfg.end
            s.mult = cfg.mult

    def __call__(s, step):
        if s.mult is not None and s.freq < s.end and step == s.mult * s.freq:
            s.freq *= s.mult
        return step % s.freq == 0
