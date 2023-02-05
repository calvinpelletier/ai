import torch
import math

import ai.model as m


class MuZeroMLP(m.Model):
    def __init__(s, game):
        super().__init__()
        s._representation_net = RepresentationNet(game.ob_shape)
        s._prediction_net = PredictionNet(game.n_actions)
        s._dynamics_net = DynamicsNet(game.n_actions)

    def forward(s, ob=None, state=None, action=None):
        is_initial = ob is not None and state is None and action is None
        is_recurrent = ob is None and state is not None and action is not None
        assert is_initial or is_recurrent

        if is_initial:
            pi, v, r, state = s._infer_initial(ob)
        else:
            pi, v, r, state = s._infer_recurrent(state, action)

        return {'pi': pi, 'v': v, 'r': r, 'state': state}

    def _infer_initial(s, ob):
        state = s._representation_net(ob)
        pi, v = s._prediction_net(state)
        return pi, v, torch.zeros_like(v), state

    def _infer_recurrent(s, state, action):
        new_state, r = s._dynamics_net(state, action)
        pi, v = s._prediction_net(new_state)
        return pi, v, r, new_state


class RepresentationNet(m.Module):
    def __init__(s, ob_shape):
        super().__init__()
        s._net = m.seq(
            m.flatten(),
            m.fc(math.prod(ob_shape), 64, actv='sigmoid'),
        )

    def forward(s, ob):
        return s._net(ob)


class PredictionNet(m.Module):
    def __init__(s, n_actions):
        super().__init__()
        s._policy_net = m.fc(64, n_actions)
        s._value_net = m.fc(64, 1, actv='tanh')

    def forward(s, state):
        return s._policy_net(state), s._value_net(state)


class DynamicsNet(m.Module):
    def __init__(s, n_actions):
        super().__init__()
        s._n_actions = n_actions
        s._state_net = m.fc(64 + n_actions, 64, actv='sigmoid')
        s._reward_net = m.fc(64, 1, actv='tanh')

    def forward(s, state, action):
        x = torch.cat((state, m.f.one_hot(action, s._n_actions)), dim=1)
        new_state = s._state_net(x)
        return new_state, s._reward_net(new_state)
