from collections import namedtuple


class Logarithmic:
    def __init__(s, start=2**10, end=2**16, mult=2):
        s.start = start
        s.end = end
        s.mult = mult


class Schedule:
    def __init__(s, cfg):
        assert isinstance(cfg, Logarithmic), 'TODO: non-logarithmic schedule'
        s.freq = cfg.start
        s.end = cfg.end
        s.mult = cfg.mult

    def __call__(s, step):
        if s.freq < s.end and step == s.mult * s.freq:
            s.freq *= s.mult
        return step % s.freq == 0
