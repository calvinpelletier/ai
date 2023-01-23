import torch

from ai.examples.stylegan2.model import Generator, Discriminator
from ai.util import assert_shape


DEVICE = 'cuda'


def test_stylegan2_model():
    g = Generator(64).init().to(DEVICE)
    d = Discriminator(64).init().to(DEVICE)
    z = torch.randn(8, g.z_dim).to(DEVICE)

    img = g(z)
    assert_shape(img, [8, 3, 64, 64])

    pred = d(img)
    assert_shape(pred, [8, 1])

    loss = pred.mean()
    loss.backward()

    out2 = g(z, trunc=0.5)
