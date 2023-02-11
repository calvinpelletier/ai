'''position embeddings'''

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(s, size: int):
        super().__init__()
        assert size % 2 == 0
        s._size = size

        half_size = s._size // 2
        c = torch.log(torch.Tensor([10000.])) / (half_size - 1)
        c = torch.exp(-c * torch.arange(half_size))
        s.register_buffer('_c', c.unsqueeze(0))

    def forward(s, x):
        emb = x.unsqueeze(-1) * s._c
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

    def __len__(s):
        return s._size


POSITION_EMBEDDINGS = {
    'sin': SinusoidalPosEmb,
}

def pos_emb(size: int, type_: str = 'sin', **kw):
    '''Position embedding.

    size : int
        number of positions
    type_ : string
        see POSITION_EMBEDDINGS for possible values
    **kw
        passed to module
    '''

    if type_ not in POSITION_EMBEDDINGS:
        raise ValueError(f'unknown position embedding: {type_}')
    return POSITION_EMBEDDINGS[type_](size, **kw)
