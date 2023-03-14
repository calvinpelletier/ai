import torch
from typing import Dict

from ai.train.env.base import Env
from ai.model import Model


class Classify(Env):
    '''Classification environment.

    Calculates the cross entropy loss between model(data['x']) and data['y'].
    '''

    def __init__(s):
        s.loss_fn = torch.nn.CrossEntropyLoss()

    def __call__(s,
        model: Model,
        batch: Dict[str, torch.Tensor],
        step: int = 0,
    ) -> torch.Tensor:
        return s.loss_fn(model(batch['x']), batch['y'])
