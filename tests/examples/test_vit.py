from ai.util.testing import *
from ai.examples.vit.main import vit_cifar10


def test_vit():
    vit_cifar10('/tmp/testing/vit', DEVICE, steps=4)
