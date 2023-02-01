from ai.model.ae import ImgAutoencoder
from ai.testing import DEVICE
from ai.util import assert_autoencode


def test_img_autoencoder():
    model = ImgAutoencoder(64, 4)
    assert_autoencode(model, [8, 3, 64, 64], DEVICE)
