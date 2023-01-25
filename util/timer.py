from time import time


class Timer:
    def __init__(s, timelimit):
        s.start = time()
        s.end = s.start + timelimit if timelimit is not None else None

    def __call__(s):
        return s.end is not None and time() >= s.end
