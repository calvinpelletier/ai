import torch
from torch import Tensor
from typing import Callable, Dict

from ai.train.env.base import Env
from ai.model import Model


class Target(Env):
    '''Target environment (model tries to create y from x).

    Calculates the difference (default: MSE) between model(x) and y.
    '''

    def __init__(s, loss_fn: Callable = torch.nn.MSELoss()):
        s.loss_fn = loss_fn

    def __call__(s,
        model: Model,
        data: Dict[str, Tensor],
        step: int = 0,
    ) -> Tensor:
        return s.loss_fn(model(data['x']), data['y'])
