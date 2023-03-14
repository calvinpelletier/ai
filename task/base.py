from typing import Union, Dict, Callable

from ai.util import no_op
from ai.model import Model


class Task:
    def __call__(s,
        model: Union[Model, Dict[str, Model]],
        log: Callable = no_op,
    ):
        raise NotImplementedError()
