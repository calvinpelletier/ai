from ai.util.testing import *
from ai.examples.diffusion.main import run


def test_diffusion():
    run('/tmp/testing/diffusion', DEVICE, 4)
