import torch
from copy import deepcopy


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


def sgd(model, **kw):
    return Opt(torch.optim.SGD, model.parameters(), **kw)

def adam(model, **kw):
    return Opt(torch.optim.Adam, model.parameters(), **kw)

def adamw(model, **kw):
    return Opt(torch.optim.AdamW, model.parameters(), **kw)


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
