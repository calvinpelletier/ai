import torch
from typing import Optional, Callable


class DataLoader:
    def __init__(s,
        loader: torch.utils.data.DataLoader,
        device: str | torch.device,
        infinite: bool = False,
        postprocessor: Optional[Callable] = None,
    ):
        s._loader = loader
        s._device = device
        s._infinite = infinite
        s._postprocessor = postprocessor

    def __iter__(s):
        if s._infinite:
            return inf_data_iter(s._loader, s._device, s._postprocessor)
        return data_iter(s._loader, s._device, s._postprocessor)


def data_iter(loader, device, postprocessor=None):
    for batch in loader:
        batch = transfer_data(batch, device)
        batch = process(batch, postprocessor)
        yield batch


def inf_data_iter(loader, device, postprocessor=None):
    while 1:
        for batch in loader:
            batch = transfer_data(batch, device)
            batch = process(batch, postprocessor)
            yield batch


def process(data, fn):
    if isinstance(fn, dict):
        assert isinstance(data, dict)
        for k in fn.keys():
            data[k] = fn[k](data[k])
    elif fn is not None:
        data = fn(data)
    return data


def transfer_data(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple) or isinstance(data, list):
        return [x.to(device) for x in data]
    elif isinstance(data, dict):
        return {k: v.to(device) for k, v in data.items()}
    else:
        raise ValueError(f'unexpected data type: {typeof(data)}')
