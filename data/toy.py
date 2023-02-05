'''toy datasets'''

from sklearn.datasets import make_moons
import numpy as np

from ai.data.dataset import Dataset


def moons_dataset(n=128, include_labels=True, mult=1.):
    x, y = make_moons(n, shuffle=True, noise=.03, random_state=0)
    x = x.astype(np.float32) * mult
    if include_labels:
        return Dataset({'x': x, 'y': y.astype(np.uint8)})
    return Dataset(x)
