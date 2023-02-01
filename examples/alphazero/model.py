import math
import ai.model as m


class AlphaZeroMLP(m.Model):
    def __init__(s, game, n_layers=2, layer_dim=128):
        super().__init__()

        s.body = m.seq(
            m.flatten(),
            m.fc(math.prod(game.ob_shape), layer_dim, actv='mish'),
            m.repeat(n_layers, m.fc(layer_dim, layer_dim, actv='mish')),
        )

        s.policy_head = m.fc(layer_dim, game.n_actions, actv=None)
        s.value_head = m.fc(layer_dim, 1, actv='tanh')

    def forward(s, ob):
        enc = s.body(ob.float())
        return {
            'pi': s.policy_head(enc),
            'v': s.value_head(enc).squeeze(1),
        }
