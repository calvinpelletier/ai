'''Activation functions.'''

from torch import nn
from typing import Optional
from functools import partial


ACTIVATIONS = {
    'relu': nn.ReLU,
    'lrelu': partial(nn.LeakyReLU, 0.2),
    'prelu': nn.PReLU,
    'sigmoid': nn.Sigmoid,
    'mish': nn.Mish,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
}

def build_actv(actv: Optional[str]):
    '''String-to-module for activation functions.

    ARGS
        actv : string or null
            see ACTIVATIONS for possible string values
    '''

    if actv is None:
        return None
    if actv in ACTIVATIONS:
        return ACTIVATIONS[actv]()
    raise ValueError(f'unknown actv: {actv}')
