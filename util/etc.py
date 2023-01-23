import numpy as np


def log2_diff(a, b):
    return abs(int(np.log2(a)) - int(np.log2(b)))
