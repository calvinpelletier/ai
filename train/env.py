import torch

from ai.util import null_op


class Env:
    def __init__(s):
        s.log = null_op


class Classify(Env):
    def __init__(s, loss_fn=torch.nn.CrossEntropyLoss()):
        super().__init__()
        s.loss_fn = loss_fn

    def __call__(s, model, batch, step=0):
        x, y = batch
        return s.loss_fn(model(x), y)


class Reconstruct(Env):
    def __init__(s, loss_fn=torch.nn.MSELoss()):
        super().__init__()
        s.loss_fn = loss_fn

    def __call__(s, model, batch, step=0):
        return s.loss_fn(model(batch), batch)
