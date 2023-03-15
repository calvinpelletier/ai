from ai.util.testing import *
from ai.data import cifar10


def test_cifar10_dataset():
    ds = cifar10()
    assert len(ds) == 60000

    train_ds, val_ds = ds.split(.9, .1)
    assert len(train_ds) == 54000
    assert len(val_ds) == 6000

    train_ds, val_ds = ds.split()
    assert len(train_ds) == 50000
    assert len(val_ds) == 10000

    batch = val_ds.sample(8, DEVICE)
    assert_shape(batch['x'], [8, 3, 32, 32])
    assert_shape(batch['y'], [8])
    assert_bounds(batch['x'], [-1., 1.])
    assert_bounds(batch['y'], [0, 9])
