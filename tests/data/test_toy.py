import torch

from ai.util.testing import *
from ai.data.toy import moons_dataset
from ai.util import print_info


def test_moons_dataset():
    ds = moons_dataset(n=128)
    assert len(ds) == 128

    batch = ds.sample(16, DEVICE)
    x, y = batch['x'], batch['y']
    print_info(x, 'x')
    print_info(y, 'y')
    assert_shape(x, [16, 2])
    assert_shape(y, [16])
    assert_bounds(y, [0, 1])
