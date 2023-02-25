from collections import namedtuple
from typing import Union


class ScheduleConfig:
    pass


class Logarithmic(ScheduleConfig):
    '''A schedule config where the interval grows exponentially.'''

    def __init__(s, start: int = 2**10, end: int = 2**16, mult: int = 2):
        s.start = start
        s.end = end
        s.mult = mult


class Schedule:
    '''A callable that returns whether a step is on or off schedule.'''

    def __init__(s, cfg: Union[int, ScheduleConfig]):
        if isinstance(cfg, int):
            s.freq = cfg
            s.mult = None
        else:
            assert isinstance(cfg, Logarithmic), 'unknown schedule type'
            s.freq = cfg.start
            s.end = cfg.end
            s.mult = cfg.mult

    def __call__(s, step: int) -> bool:
        if s.mult is not None and s.freq < s.end and step == s.mult * s.freq:
            s.freq *= s.mult
        return step % s.freq == 0
