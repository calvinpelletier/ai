import torch
import torch.utils.data as torch_data
from typing import Optional, Callable, Union

from ai.data.util import data_iter, inf_data_iter


def create_data_iterator(
    data: torch_data.Dataset,
    batch_size: int,
    device: str = 'cuda',
    shuffle: bool = False,
    infinite: bool = False,
    drop_last: bool = True,
    n_workers: int = 1,
    postprocess: Optional[Callable] = None,
    single_batch: bool = False,
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
        single_batch : bool
            (debugging) process a single batch then continuously yield it
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

    loader = torch_data.DataLoader(data, **loader_kwargs)
    return DataIterator(loader, device, infinite, postprocess, single_batch)


class DataIterator:
    def __init__(s,
        loader: torch_data.DataLoader,
        device: Union[torch.device, str],
        infinite: bool = False,
        postprocessor: Optional[Callable] = None,
        single_batch: bool = False,
    ):
        s._loader = loader
        s._device = device
        s._infinite = infinite
        s._postprocessor = postprocessor

        s._single_batch = None
        if single_batch:
            s._single_batch = next(iter(s))

    def __iter__(s):
        if s._single_batch is not None:
            return _single_batch_iter(s._single_batch, s._infinite)

        if s._infinite:
            return inf_data_iter(s._loader, s._device, s._postprocessor)
        return data_iter(s._loader, s._device, s._postprocessor)

def _single_batch_iter(batch, infinite):
    if infinite:
        while 1:
            yield batch
    yield batch
