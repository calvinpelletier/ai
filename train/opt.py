import torch
from copy import deepcopy
from typing import Union


# TODO: complete refactor (maybe move gradient clipping to trainer?)
# TODO: comments, docstrings, type hints


class Opt:
    def __init__(s, opt_cls, params, grad_clip=False, **kw):
        s._opt = opt_cls(params, **kw)
        s._opt.zero_grad()

        s._grad_clip = grad_clip
        s._params = params if grad_clip else None

    def step(s):
        if s._grad_clip:
            torch.nn.utils.clip_grad_norm_(s._params, 1.)
        s._opt.step()

    def zero_grad(s):
        s._opt.zero_grad()

    def state_dict(s):
        return s._opt.state_dict()

    def load_state_dict(s, sd):
        s._opt.load_state_dict(sd)


def sgd(model, lr=1e-4, **kw):
    return Opt(torch.optim.SGD, model.parameters(), lr=lr, **kw)

def adam(model, lr=1e-4, **kw):
    return Opt(torch.optim.Adam, model.parameters(), lr=lr, **kw)

def adamw(model, lr=1e-4, **kw):
    return Opt(torch.optim.AdamW, model.parameters(), lr=lr, **kw)


OPTS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}

def build(cfg, model):
    if cfg.type not in OPTS:
        raise ValueError(f'unknown opt: {cfg.type}')
    opt_cls = OPTS[cfg.type]

    cfg_dict = deepcopy(vars(cfg))
    del cfg_dict['type']

    return Opt(opt_cls, model.parameters(), **cfg_dict)


OptLike = Union[Opt, torch.optim.Optimizer]
