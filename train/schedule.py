

class Exponential:
    def __init__(s, start, end):
        s.freq = start
        s.end = end

    def __call__(s, step):
        if s.freq < s.end and step == 2 * s.freq:
            s.freq *= 2
        return step % s.freq == 0
