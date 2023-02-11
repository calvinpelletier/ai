from ai.util.testing import *
import ai.model as m


def test_conv2d():
    model = m.Model(m.seq(
        m.conv(8, 8, k=5),
        m.conv(8, 4, stride=.5),
        m.conv(4, 8, stride=2),
        m.conv(8, 8, actv='relu'),
        m.conv(8, 8, norm='batch'),
        m.conv(8, 8, norm='instance'),
        m.conv(8, 8, clamp=1.),
        m.conv(8, 8, bias=False),
        m.conv(8, 8, padtype='reflect'),
        m.conv(8, 16, scale_w=True),
        m.conv(16, 8, lr_mult=.1),
    ))

    assert_autoencode(model, [16, 8, 32, 32], DEVICE)
