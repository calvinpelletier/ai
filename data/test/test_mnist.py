from ai.testing import DEVICE
from ai.data.mnist import mnist_dataset
from ai.util import assert_shape, assert_bounds


def test_mnist_dataset():
    ds = mnist_dataset()
    assert len(ds) == 70000

    train_ds, val_ds = ds.split(.9, .1)
    assert len(train_ds) == 63000
    assert len(val_ds) == 7000

    batch = val_ds.sample(8, DEVICE)
    assert_shape(batch['x'], [8, 1, 28, 28])
    assert_shape(batch['y'], [8])
    assert_bounds(batch['x'], [-1., 1.])
    assert_bounds(batch['y'], [0, 9])
