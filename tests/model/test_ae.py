from ai.util.testing import *
from ai.model.ae import ImgAutoencoder


def test_img_autoencoder():
    model = ImgAutoencoder(64, 4)
    assert_autoencode(model, [8, 3, 64, 64], DEVICE)
