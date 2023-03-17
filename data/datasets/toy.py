# type: ignore
'''Toy datasets.'''

from sklearn.datasets import make_moons
import numpy as np

from ai.data.dataset import Dataset


def moons(
    n: int = 128,
    include_labels: bool = True,
    mult: float = 1.,
) -> Dataset:
    '''Scikit-learn moons toy dataset.

    DATA
        x : [n, 2] (float32)
            2D coordinates
        y : [n] (uint8)
            0/1 label for the top/bottom moon (if include_labels == True)

    ARGS
        n : int
            number of samples in the dataset
        include_labels : bool
            include binary labels as key 'y'
        mult : float
            scale data by a value
    '''

    x, y = make_moons(n, shuffle=True, noise=.03, random_state=0)
    x = x.astype(np.float32) * mult
    if include_labels:
        return Dataset({'x': x, 'y': y.astype(np.uint8)})
    return Dataset(x)
