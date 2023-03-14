import torch
from copy import deepcopy


# TODO: complete refactor (maybe move gradient clipping to trainer?)
# TODO: comments, docstrings, type hints


class SGD(torch.optim.SGD):
    def __init__(s, model, lr=1e-4, grad_clip=False, **kw):
        params = model.parameters()
        super().__init__(params, lr=lr, **kw)
        s.__ai__params = params
        s.__ai__grad_clip = grad_clip

    def step(s, *a, **kw):
        if s.__ai__grad_clip:
            torch.nn.utils.clip_grad_norm_(s.__ai__params, 1.)
        super().step(*a, **kw)


class Adam(torch.optim.Adam):
    def __init__(s, model, lr=1e-4, grad_clip=False, **kw):
        params = model.parameters()
        super().__init__(params, lr=lr, **kw)
        s.__ai__params = params
        s.__ai__grad_clip = grad_clip

    def step(s, *a, **kw):
        if s.__ai__grad_clip:
            torch.nn.utils.clip_grad_norm_(s.__ai__params, 1.)
        super().step(*a, **kw)


class AdamW(torch.optim.AdamW):
    def __init__(s, model, lr=1e-4, grad_clip=False, **kw):
        params = model.parameters()
        super().__init__(params, lr=lr, **kw)
        s.__ai__params = params
        s.__ai__grad_clip = grad_clip

    def step(s, *a, **kw):
        if s.__ai__grad_clip:
            torch.nn.utils.clip_grad_norm_(s.__ai__params, 1.)
        super().step(*a, **kw)


OPTS = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
}

def build(cfg, model):
    type_ = cfg.type.lower()
    if type_ not in OPTS:
        raise ValueError(f'unknown opt: {cfg.type}')
    opt_cls = OPTS[type_]

    cfg_dict = deepcopy(vars(cfg))
    del cfg_dict['type']

    return opt_cls(model, **cfg_dict)
