import pytest

from ai.data import img_dataset
from ai.util import assert_shape, assert_bounds


def test_ffhq_dataset():
    ds = img_dataset('ffhq', imsize=64)
    _test_dataset(ds, 70_000, [3, 64, 64], [-1., 1.])


def _test_dataset(ds, size, shape, bounds):
    assert len(ds) == size

    val_ds, train_ds = ds.split(.1, .9)
    assert len(val_ds) == int(.1 * size)
    assert len(train_ds) == int(.9 * size)

    with pytest.raises(ValueError):
        ds.split(.1, .2, .3)

    samples = val_ds.sample(8)
    assert_shape(samples, [8] + shape)
    assert_bounds(samples, bounds)

    loader = train_ds.loader(8)
    batch = next(loader)
    assert_shape(batch, [8] + shape)
    assert_bounds(batch, bounds)