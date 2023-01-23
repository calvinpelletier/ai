import ai.model as m
from ai.util import assert_autoencode


DEVICE = 'cuda'


class ImgAutoencoder(m.Model):
    def __init__(s,
        imsize,
        bottleneck,
        nc_min=32,
        nc_max=512,
        enc_block=lambda _, nc1, nc2: m.resblk(nc1, nc2, stride=2),
        dec_block=lambda _, nc1, nc2: m.resblk(nc1, nc2, stride=.5),
    ):
        super().__init__()

        s.encode = m.seq(
            m.conv(3, nc_min, actv='mish'),
            m.pyramid(imsize, bottleneck, nc_min, nc_max, enc_block),
        )

        s.decode = m.seq(
            m.pyramid(bottleneck, imsize, nc_min, nc_max, dec_block),
            m.conv(nc_min, 3, actv='tanh'),
        )

    def forward(s, x):
        return s.decode(s.encode(x))


def test_img_autoencoder():
    model = ImgAutoencoder(64, 4)
    assert_autoencode(model, [8, 3, 64, 64], DEVICE)
