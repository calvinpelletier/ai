import torch

from ai.util import null_op


class Env:
    def __init__(s):
        s.log = null_op


class Classify(Env):
    def __init__(s):
        super().__init__()
        s.loss_fn = torch.nn.CrossEntropyLoss()

    def __call__(s, model, batch, step=0):
        return s.loss_fn(model(batch['x']), batch['y'])


class Reconstruct(Env):
    def __init__(s, loss_fn=torch.nn.MSELoss()):
        super().__init__()
        s.loss_fn = loss_fn

    def __call__(s, model, x, step=0):
        return s.loss_fn(model(x), x)


class Diffusion(Env):
    def __init__(s):
        super().__init__()
        s.loss_fn = torch.nn.MSELoss()

    def __call__(s, model, x, step=0):
        t = torch.randint(0, len(model), [x.shape[0]], device=x.device)
        noise = torch.randn_like(x)
        noisy = model.noisify(x, t, noise)
        pred = model(noisy, t)
        return s.loss_fn(pred, noise)
