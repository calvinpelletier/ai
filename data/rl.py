from torch import Tensor
from typing import Optional, Union, Dict

from ai.data.util import transfer_data
from ai.data.buffer import DataBuffer


class RLDataIterator:
    def __init__(s,
        buffer: DataBuffer,
        device: str,
        model_update_interval: Optional[int],
    ):
        s._buffer = buffer
        s._device = device
        s.model_update_interval = model_update_interval

    def __iter__(s):
        return s

    def __next__(s):
        return transfer_data(s._buffer.get_batch(), s._device)

    def model_update(s, params: dict):
        raise NotImplementedError()
