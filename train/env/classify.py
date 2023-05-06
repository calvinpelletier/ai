import torch
from typing import Dict

from ai.train.env.base import Env
from ai.model import Model


class Classify(Env):
    '''Classification environment.

    Calculates the cross entropy loss between model(x) and y.
    '''

    def __init__(s, x_key: str = 'x', y_key: str = 'y'):
        s.loss_fn = torch.nn.CrossEntropyLoss()
        s.x = x_key
        s.y = y_key

    def __call__(s,
        model: Model,
        batch: Dict[str, torch.Tensor],
        step: int = 0,
    ) -> torch.Tensor:
        return s.loss_fn(model(batch[s.x]), batch[s.y])
