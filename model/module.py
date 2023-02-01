'''models and modules'''

import torch

from ai.model.param import param_init


class Model(torch.nn.Module):
    '''class that wraps the top-level module of a model

    example 1 (randomly initialized parameters):
        model = Model(module).init().to(device).train()

    example 2 (load parameters from disk):
        model = Model(module).init(path_to_weights).to(device).eval()

    methods
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

    def __init__(s, net=None):
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

    def init(s, path=None):
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

    def save(s, path):
        '''save parameters to disk

        path : pathlike
        '''

        torch.save(s.state_dict(), path)

    def set_req_grad(s, req_grad):
        '''set "requires_grad" flag of every parameter in model

        req_grad : bool
        '''

        for param in s.parameters():
            param.requires_grad = req_grad


# alias
class Module(torch.nn.Module):
    pass


# alias
def modules(x):
    return torch.nn.ModuleList(x)
