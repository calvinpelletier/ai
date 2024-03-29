import numpy as np
from uuid import uuid4
from typing import Optional
from random import choices
import string


def log2_diff(a, b):
    return abs(int(np.log2(a)) - int(np.log2(b)))


def no_op(*a, **kw):
    pass


def gen_id(length=4):
    return ''.join(choices(string.ascii_letters + string.digits, k=length))

def gen_uuid():
    return uuid4().hex


def softmax(x, t=1.):
    x = x.astype(np.float32) / t
    x = np.exp(x - np.max(x))
    return x / x.sum()

def softmax_sample_idx(x, t=1.):
    return np.random.choice(np.arange(len(x)), p=softmax(x, t))


def on_interval(i: int, interval: Optional[int]) -> bool:
    return interval is not None and i % interval == 0
