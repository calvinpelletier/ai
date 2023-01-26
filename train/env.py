import torch


class Classify:
    def __init__(s, loss_fn=torch.nn.CrossEntropyLoss()):
        s.loss_fn = loss_fn

    def __call__(s, model, batch, step=0):
        x, y = batch
        return s.loss_fn(model(x), y)


class Reconstruct:
    def __init__(s, loss_fn=torch.nn.MSELoss()):
        s.loss_fn = loss_fn

    def __call__(s, model, batch, step=0):
        return s.loss_fn(model(batch), batch)
