import torch
import numpy as np

from ai.data.util import create_data_loader


class Dataset:
    def __init__(s, data, preprocess=None, postprocess=None):
        '''
        data : array-like or dict of array-like
            a representation of the underlying dataset that can be held in
            memory all at once (e.g. a list of paths to image files)
        preprocess : callable or null
            a function used by data workers (e.g. read image from disk)
            args
                an item from <data>
            returns
                tensor or ndarray (or list/dict of them)
        postprocess : callable or null
            a function called after the data has been transfered to the device
            (e.g. normalize an image tensor)
            args
                tensor or dict of tensors
            returns
                tensor or dict of tensors
        '''

        if isinstance(data, np.lib.npyio.NpzFile):
            data = {k: data[k] for k in data.keys()}
        s._data = data

        s._preprocess = preprocess
        s._postprocess = postprocess

        s._n = _get_length(s._data)

    def __len__(s):
        return s._n

    def length(s, batch_size):
        '''the length accounting for dropping the last incomplete batch

        args
            batch_size : int
        returns
            int
        '''

        return (s._n // batch_size) * batch_size

    def split(s, *ratio):
        '''split the dataset into subdatasets

        args
            *ratio : list of float
                the relative size of the subdatasets (the sum should be 1)
        returns
            list of Dataset
        '''

        assert len(ratio) > 1
        if not np.isclose(sum(ratio), 1.):
            raise ValueError(f'ratio ({str(ratio)}) != 1.')

        ret = []
        x, y = 0, 0
        for r in ratio[:-1]:
            x = y
            y += int(r * s._n)
            ret.append(s._create_subdataset(x, y))
        ret.append(s._create_subdataset(y, None))
        return ret

    def sample(s, n, device='cuda'):
        '''get n samples

        args
            n : int
                number of samples
            device : str
        returns
            tensor[n, ...] or list/dict of tensor[n, ...]
        '''

        return next(s.loader(n, device))

    def loader(s,
        batch_size,
        device='cuda',
        n_workers=1,
        train=False,
        drop_last=True,
    ):
        '''create a data loader of this dataset

        args
            batch_size : int
            device : str
            n_workers : int
            train : bool
                shuffle and loop infinitely if true
            drop_last: bool
                drop last batch if incomplete
        returns
            iterable
        '''

        return create_data_loader(
            _Dataset(s._data, s._preprocess),
            batch_size=batch_size,
            device=device,
            shuffle=train,
            infinite=train,
            drop_last=drop_last,
            n_workers=n_workers,
            postprocess=s._postprocess,
        )

    def _create_subdataset(s, x, y):
        if isinstance(s._data, dict):
            subdata = {k: _slice(v, x, y) for k, v in s._data.items()}
        else:
            subdata = _slice(s._data, x, y)

        return Dataset(subdata, s._preprocess, s._postprocess)

def _slice(data, x, y):
    return data[x:y] if y is not None else data[x:]


class _Dataset(torch.utils.data.Dataset):
    def __init__(s, data, preprocess=None):
        super().__init__()
        s._data = data
        s._preprocess = preprocess

        s._n = _get_length(data)
        assert s._n > 0

        s._fn = _process_dict if isinstance(data, dict) else _process

    def __len__(s):
        return s._n

    def __getitem__(s, i):
        return s._fn(s._data, s._preprocess, i)

def _process(data, fn, i):
    return data[i] if fn is None else fn(data[i])

def _process_dict(data, fn, i):
    if fn is None:
        return {k: v[i] for k, v in data.items()}

    if isinstance(fn, dict):
        return {
            k: fn[k](v[i]) if k in fn else v[i]
            for k, v in data.items()
        }

    return {k: fn(v[i]) for k, v in data.items()}


def _get_length(data):
    if isinstance(data, dict):
        keys = list(data.keys())
        n = len(data[keys[0]])
        if len(keys) > 1:
            for k in keys[1:]:
                assert len(data[k]) == n
    else:
        n = len(data)
    return n
