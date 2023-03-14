import torch
from typing import Dict

from ai.train.env.base import Env
from ai.model import Model
from ai.train.logger import log


class RL(Env):
    '''Reinforcement learning environment.

    ARGS
        v_weight
            the weight of the value loss relative to the policy loss
    '''

    def __init__(s, v_weight: float = 1.):
        s._pi_loss_fn = torch.nn.CrossEntropyLoss()
        s._v_loss_fn = torch.nn.MSELoss()
        s._v_weight = v_weight

    def __call__(s,
        model: Model,
        batch: Dict[str, torch.Tensor],
        step: int = 0,
    ) -> torch.Tensor:
        # predict policy/value from observation
        pred = model(batch['ob'])

        # policy loss
        pi_loss = s._pi_loss_fn(pred['pi'], batch['pi'])
        log('loss.policy', pi_loss)

        # value loss
        v_loss = s._v_loss_fn(pred['v'], batch['v'])
        log('loss.value', v_loss)

        return pi_loss + v_loss * s._v_weight
