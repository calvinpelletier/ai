from ai.util.testing import *
import ai.model as m


def test_self_attention():
    assert_autoencode(m.Model(m.self_attn(8, 2, 8)), [2, 4, 8])


def test_transformer_enc():
    assert_autoencode(m.Model(m.transformer_enc(2, 8, 2, 8, 8)), [2, 4, 8])
