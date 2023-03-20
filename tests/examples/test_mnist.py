from ai.util.testing import *
from ai.examples.mnist.main import mnist


def test_mnist():
    mnist('/tmp/testing/mnist', DEVICE, steps=10)
