import ai.model as m
from ai.testing import DEVICE
from ai.util import assert_autoencode


def test_fully_connected():
    model = m.Model(m.seq(
        m.fc(8, 8, actv='relu'),
        m.fc(8, 16, bias=False),
        m.fc(16, 16, bias_init=1.),
        m.fc(16, 8, scale_w=True),
        m.fc(8, 8, lr_mult=.1),
    ))

    assert_autoencode(model, [16, 8], DEVICE)
