import torch
from typing import Optional, Callable, Union

from ai.data.util import data_iter, inf_data_iter


def create_data_loader(
    data: torch.utils.data.Dataset,
    batch_size: int,
    device: torch.device = 'cuda',
    shuffle: bool = False,
    infinite: bool = False,
    drop_last: bool = True,
    n_workers: int = 1,
    postprocess: Optional[Callable] = None,
):
    '''Create a data loader and wrap it in an iterator.

    The data iterator around the loader will transfer data to the device,
    optionally run a postprocess function, and optionally loop infinitely.

    ARGS
        data : torch.utils.data.(Dataset or IterableDataset)
        batch_size : int
        device : str
        shuffle : bool
        infinite : bool
            wrap loader in an infinite loop
        drop_last : bool
            drop last batch if incomplete
        n_workers : int
        postprocessor : callable or null
            a function called after the data has been transfered to the device
            args:
                tensor or dict of tensors
            returns:
                tensor or dict of tensors
    '''

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'drop_last': drop_last if batch_size is not None else False,
        'num_workers': n_workers,
    }

    if device == 'cuda' or device.startswith('cuda:'):
        loader_kwargs['pin_memory'] = True
    elif device == 'cpu':
        pass
    else:
        raise ValueError(f'unexpected device: {device}')

    loader = torch.utils.data.DataLoader(data, **loader_kwargs)
    return DataLoader(loader, device, infinite, postprocess)


class DataLoader:
    def __init__(s,
        loader: torch.utils.data.DataLoader,
        device: str,
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
