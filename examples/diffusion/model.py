import ai.model as m


class DiffusionMLP(m.DiffusionModel):
    def __init__(s, dim):
        super().__init__([dim])
        s.encode_x = m.fc(dim, 128)
        s.encode_t = m.pos_emb(128)
        s.main = m.seq(
            m.fc(256, 128, actv='gelu'),
            m.repeat(3, m.fc(128, 128, actv='gelu')),
            m.fc(128, dim),
        )

    def forward(s, x, t):
        return s.main(m.f.cat([s.encode_x(x), s.encode_t(t)], dim=1))
