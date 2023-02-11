from ai.util.testing import *
import ai.model as m


class ModulatedAutoencoder(m.Model):
    def __init__(s, modtype):
        super().__init__()

        s.to_z = m.seq(m.global_avg(), m.flatten())

        s.encode = m.modules([
            m.modconv(16, 32, 16, modtype, stride=2),
            m.modconv(32, 64, 16, modtype, stride=2, actv=None),
            m.modconv(64, 64, 16, modtype, stride=2, noise=True),
        ])

        s.decode = m.modules([
            m.modconv(64, 64, 16, modtype, stride=0.5),
            m.modconv(64, 32, 16, modtype, stride=0.5, clamp=1.),
            m.modconv(32, 16, 16, modtype, stride=0.5, actv='tanh'),
        ])

    def forward(s, x):
        z = s.to_z(x)
        for block in s.encode:
            x = block(x, z)
        for block in s.decode:
            x = block(x, z)
        return x


def test_adalin_modulated_conv():
    model = ModulatedAutoencoder('adalin')
    assert_autoencode(model, [8, 16, 64, 64], DEVICE)


def test_weight_modulated_conv():
    model = ModulatedAutoencoder('weight')
    assert_autoencode(model, [8, 16, 64, 64], DEVICE, lr=1e-2)
