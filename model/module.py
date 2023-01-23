import torch

from ai.model.param import param_init


class Model(torch.nn.Module):
    def __init__(s, net=None):
        super().__init__()
        s._net = net

    def forward(s, *a, **kw):
        return s._net(*a, **kw)

    def init(s, path=None):
        if path is None:
            s.apply(param_init)
        else:
            s.load_state_dict(torch.load(path))
        return s

    def save(s, path):
        torch.save(s.state_dict(), path)

    def set_req_grad(s, req_grad):
        for param in s.parameters():
            param.requires_grad = req_grad


class Module(torch.nn.Module):
    pass


def modules(x):
    return torch.nn.ModuleList(x)
