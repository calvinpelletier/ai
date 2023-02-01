import torch
import math

from ai.data.util import create_data_loader


class Dataset:
    def __init__(s, data, preprocessor=None, postprocessor=None):
        '''
        data : array-like
            a representation of the underlying dataset that can be held in
            memory all at once (e.g. a list of paths to image files)
        preprocessor : callable or null
            a function used by data workers (e.g. read image from disk)
            args
                an item from <data>
            returns
                tensor or ndarray (or list/dict of them)
        postprocessor : callable or null
            a function called after the data has been transfered to the device
            (e.g. normalize an image tensor)
            args
                tensor or list/dict of tensors
            returns
                tensor or list/dict of tensors
        '''

        s._data = data
        s._preprocessor = preprocessor
        s._postprocessor = postprocessor

    def __len__(s):
        return len(s._data)

    def length(s, batch_size):
        '''the length accounting for dropping the last incomplete batch

        args
            batch_size : int
        returns
            int
        '''

        return (len(s._data) // batch_size) * batch_size

    def split(s, *ratio):
        '''split the dataset into subdatasets

        args
            *ratio : list of float
                the relative size of the subdatasets (the sum should be 1)
        returns
            list of Dataset
        '''

        assert len(ratio) > 1
        if not math.isclose(sum(ratio), 1.):
            raise ValueError(f'ratio ({str(ratio)}) != 1.')

        ret = []
        n, x, y = len(s._data), 0, 0
        for r in ratio[:-1]:
            x = y
            y += int(r * n)
            ret.append(s._create_subdataset(s._data[x:y]))
        ret.append(s._create_subdataset(s._data[y:]))
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
            _Dataset(s._data, s._preprocessor),
            batch_size=batch_size,
            device=device,
            shuffle=train,
            infinite=train,
            drop_last=drop_last,
            n_workers=n_workers,
            postprocess=s._postprocessor,
        )

    def _create_subdataset(s, subdata):
        return Dataset(subdata, s._preprocessor, s._postprocessor)


class _Dataset(torch.utils.data.Dataset):
    def __init__(s, data, preprocessor=None):
        super().__init__()
        assert len(data) > 0
        s._data = data
        s._preprocessor = preprocessor

    def __len__(s):
        return len(s._data)

    def __getitem__(s, i):
        x = s._data[i]
        if s._preprocessor is not None:
            x = s._preprocessor(x)
        return x
