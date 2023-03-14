import pytest

from ai.util.testing import *
from ai.data import ImgDataset


def test_ffhq_dataset():
    ds = ImgDataset('ffhq', imsize=64)
    _test_dataset(ds, 70_000, [3, 64, 64], [-1., 1.])


def _test_dataset(ds, size, shape, bounds):
    assert len(ds) == size

    val_ds, train_ds = ds.split(.1, .9)
    assert len(val_ds) == int(.1 * size)
    assert len(train_ds) == int(.9 * size)

    with pytest.raises(ValueError):
        ds.split(.1, .2, .3)

    samples = val_ds.sample(8, DEVICE)
    assert_shape(samples, [8] + shape)
    assert_bounds(samples, bounds)

    iterator = train_ds.iterator(8, DEVICE)
    batch = next(iter(iterator))
    assert_shape(batch, [8] + shape)
    assert_bounds(batch, bounds)
