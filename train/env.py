import torch

from ai.util import no_op


class Env:
    def __call__(s, model, batch, step=0, log=no_op):
        raise NotImplementedError()

class MultiEnv:
    def __call__(s, key, models, batch, step=0, log=no_op):
        return getattr(s, key)(models, batch, step, log)


class Classify(Env):
    def __init__(s):
        s.loss_fn = torch.nn.CrossEntropyLoss()

    def __call__(s, model, batch, step=0, log=no_op):
        return s.loss_fn(model(batch['x']), batch['y'])


class Reconstruct(Env):
    def __init__(s, loss_fn=torch.nn.MSELoss()):
        s.loss_fn = loss_fn

    def __call__(s, model, x, step=0, log=no_op):
        return s.loss_fn(model(x), x)


class Diffusion(Env):
    def __init__(s):
        s.loss_fn = torch.nn.MSELoss()

    def __call__(s, model, x, step=0, log=no_op):
        t = torch.randint(0, len(model), [x.shape[0]], device=x.device)
        noise = torch.randn_like(x)
        noisy = model.noisify(x, t, noise)
        pred = model(noisy, t)
        return s.loss_fn(pred, noise)
