from __future__ import annotations
import torch
from typing import Union, Optional

from ai.model.param import param_init
from ai.path import PathLike


class Model(torch.nn.Module):
    '''Class that wraps the top-level module of a model.

    METHODS
        __init__(net: module or null)
            if net is null, user must implement Model.forward

        init(path: pathlike or null)
            initialize parameters
                if pathlike, load from disk
                if null, initialize randomly (see model/param.py)

        save(path: pathlike)
            save parameters to disk

        set_req_grad(req_grad: bool)
            set "requires_grad" flag of every parameter in model
    '''

    def __init__(s, net: Optional[torch.nn.Module] = None):
        '''
        net : module or null
            the top-level module of the model (if net is null, user must
            implement Model.forward)
        '''

        super().__init__()
        s._net = net

    def forward(s, *a, **kw):
        if s._net is None:
            raise NotImplementedError()
        return s._net(*a, **kw)

    def init(s, path: Optional[PathLike] = None) -> Model:
        '''initialize parameters

        path : pathlike or null
            if pathlike, load from disk
            if null, initialize randomly (see model/param.py)
        '''

        if path is None:
            s.apply(param_init)
        else:
            s.load_state_dict(torch.load(path))
        return s

    def save(s, path: PathLike):
        '''save parameters to disk

        path : pathlike
        '''

        torch.save(s.state_dict(), path)

    def set_req_grad(s, req_grad: bool):
        '''set "requires_grad" flag of every parameter in model

        req_grad : bool
        '''

        for param in s.parameters():
            param.requires_grad = req_grad

    def get_device(s) -> torch.device:
        return next(s.parameters()).device


# aliases
Module = torch.nn.Module
def modules(x):
    return torch.nn.ModuleList(x)
