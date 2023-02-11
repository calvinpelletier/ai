import torch

from ai.util.testing import *
import ai.model as m


def test_resblk():
    net = m.seq(m.resblk(8, 8, 2), m.resblk(8, 8, 1), m.resblk(8, 8, 0.5))
    assert_autoencode(m.Model(net), [8, 8, 8, 8], DEVICE)


def test_resblk_group():
    pass # TODO
