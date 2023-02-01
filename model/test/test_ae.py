from ai.model.ae import ImgAutoencoder
from ai.util import assert_autoencode


DEVICE = 'cuda'


def test_img_autoencoder():
    model = ImgAutoencoder(64, 4)
    assert_autoencode(model, [8, 3, 64, 64], DEVICE)
