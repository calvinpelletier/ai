import torch
from torch import Tensor
from typing import Callable

from ai.train.env.base import Env
from ai.model import Model


class Reconstruct(Env):
    '''Reconstruction environment (model tries to recreate its input).

    Calculates the difference (default: MSE) between model(data) and data.
    '''

    def __init__(s, loss_fn: Callable = torch.nn.MSELoss()):
        s.loss_fn = loss_fn

    def __call__(s, model: Model, x: Tensor, step: int = 0) -> Tensor:
        return s.loss_fn(model(x), x)
