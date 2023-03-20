from ai.util.testing import *
from ai.examples.stylegan2.main import run


def test_stylegan2():
    cfg = {
        'device': DEVICE,
        'task.n_imgs': 100,
    }
    run('/tmp/testing/sg2', 4, **cfg)
