'''activation functions'''

from torch import nn


ACTIVATIONS = {
    'relu': nn.ReLU,
    'lrelu': nn.LeakyReLU,
    'prelu': nn.PReLU,
    'sigmoid': nn.Sigmoid,
    'mish': nn.Mish,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
}

def build_actv(actv):
    '''string-to-module for activation functions

    actv : string
        see ACTIVATIONS for possible values
    '''

    if actv is None:
        return None
    if actv not in ACTIVATIONS:
        raise ValueError(f'unknown actv: {actv}')
    return ACTIVATIONS[actv]()
