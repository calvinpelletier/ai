from math import prod

import ai.model as m


class Model(m.Model):
    def __init__(s, input_shape=[1, 28, 28], n_out=10, dim=128, n_layers=4):
        super().__init__(m.seq(
            m.flatten(),
            m.fc(int(prod(input_shape)), dim, actv='relu'),
            m.repeat(n_layers, m.fc(dim, dim, actv='relu')),
            m.fc(dim, n_out),
        ))
