from math import prod

import ai.model as m


class Model(m.Model):
    def __init__(s, input_shape=[1, 28, 28], n_out=10, dim=128, n_layers=4):
        super().__init__(m.seq(
            m.flatten(),
            m.fc(prod(input_shape), dim, actv='relu'),
            m.repeat(n_layers, m.fc(dim, dim, actv='relu')),
            m.fc(dim, n_out),
        ))


class Model2(m.Model):
    def __init__(s, input_shape=[1, 28, 28], n_out=10, dim=128, layers=4):
        super().__init__()
        s.net = m.seq(
            m.fc(prod(input_shape), dim, actv='relu'),
            m.repeat(n_layers, m.fc(dim, dim, actv='relu')),
            m.fc(dim, n_out),
        )

    def forward(s, x):
        x = m.f.flatten(x)
        return s.net(x)
