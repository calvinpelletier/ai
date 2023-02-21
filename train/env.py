import torch

from ai.util import no_op
from ai.train.log import log


# base environments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Env:
    def __call__(s, model, batch, step=0):
        raise NotImplementedError()

class MultiEnv:
    def __call__(s, key, models, batch, step=0):
        return getattr(s, key)(models, batch, step)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Classify(Env):
    '''Classification environment.

    Calculates the cross entropy loss between model(data['x']) and data['y'].
    '''

    def __init__(s):
        s.loss_fn = torch.nn.CrossEntropyLoss()

    def __call__(s, model, batch, step=0):
        return s.loss_fn(model(batch['x']), batch['y'])


class Reconstruct(Env):
    '''Reconstruction environment (model tries to recreate its input).

    Calculates the difference (default: MSE) between model(data) and data.
    '''

    def __init__(s, loss_fn=torch.nn.MSELoss()):
        s.loss_fn = loss_fn

    def __call__(s, model, x, step=0):
        return s.loss_fn(model(x), x)


class Diffusion(Env):
    '''Diffusion environment.'''

    def __init__(s):
        s.loss_fn = torch.nn.MSELoss()

    def __call__(s, model, x, step=0):
        # choose random timesteps
        t = torch.randint(0, len(model), [x.shape[0]], device=x.device)

        # generate gaussian noise
        noise = torch.randn_like(x)

        # create noisy data
        noisy = model.noisify(x, t, noise)

        # predict the noise
        pred = model(noisy, t)

        # calc loss
        return s.loss_fn(pred, noise)


class RL(Env):
    '''Reinforcement learning environment.'''

    def __init__(s, v_weight=1.):
        s._pi_loss_fn = torch.nn.CrossEntropyLoss()
        s._v_loss_fn = torch.nn.MSELoss()
        s._v_weight = v_weight

    def __call__(s, model, batch, step=0):
        # predict policy/value from observation
        pred = model(batch['ob'])

        # policy loss
        pi_loss = s._pi_loss_fn(pred['pi'], batch['pi'])
        log('loss.policy', pi_loss)

        # value loss
        v_loss = s._v_loss_fn(pred['v'], batch['v'])
        log('loss.value', v_loss)

        return pi_loss + v_loss * s._v_weight
