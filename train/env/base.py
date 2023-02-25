from torch import Tensor
from typing import Union, Dict

from ai.util import no_op
from ai.train.logger import log
from ai.model import Model


class Env:
    '''Training environment.'''

    def __call__(s,
        model: Model,
        batch: Union[Tensor, Dict[str, Tensor]],
        step: int = 0,
    ) -> Tensor:
        '''
        ARGS
            model
                The model being trained.
            batch
                A batch of data.
            step
                The current training step (e.g. for regularization that only
                occurs every n steps).
        RETURNS
            A loss value.
        '''

        raise NotImplementedError()


class MultiEnv:
    '''Training environment for a MultiTrainer.'''

    def __call__(s,
        key: str,
        models: Dict[str, Model],
        batch: Union[Tensor, Dict[str, Tensor]],
        step: int = 0,
    ) -> Tensor:
        '''
        ARGS
            key
                A key of the `models` dict indicating which model the returned
                loss value is for.
            models
                A dict of models being trained.
            batch
                A batch of data.
            step
                The current training step (e.g. for regularization that only
                occurs every n steps).
        RETURNS
            A loss value.
        '''

        return getattr(s, key)(models, batch, step)
