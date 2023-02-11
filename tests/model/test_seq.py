import torch

from ai.util.testing import *
import ai.model as m


def test_seq():
    a = m.fc(8, 8, actv='relu')
    b = m.fc(8, 8, actv='relu')
    i = torch.randn(8)

    assert_equal(m.seq()(i), i)
    assert_equal(m.seq(a)(i), a(i))
    assert_equal(m.seq(a, b)(i), b(a(i)))


def test_repeat():
    a = m.fc(8, 8, actv='relu')
    i = torch.randn(8)

    assert_equal(m.repeat(0, a)(i), i)
    assert_equal(m.repeat(1, a)(i), a(i))
    assert_equal(m.repeat(2, a)(i), a(a(i)))


def test_pyramid():
    pass # TODO
