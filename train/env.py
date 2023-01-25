import torch


class Env:
    def __call__(s, model, batch):
        raise NotImplementedError()


class Classify(Env):
    def __init__(s, loss_fn=torch.nn.CrossEntropyLoss()):
        s.loss_fn = loss_fn

    def __call__(s, model, batch):
        x, y = batch
        return s.loss_fn(model(x), y)


class Reconstruct(Env):
    def __init__(s, loss_fn=torch.nn.MSELoss()):
        s.loss_fn = loss_fn

    def __call__(s, model, batch):
        return s.loss_fn(model(batch), batch)
