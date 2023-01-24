import torch
import math

from ai.data.util import create_data_loader


class Dataset:
    def __init__(s, data, preprocessor=None, postprocessor=None):
        super().__init__()
        s._data = data
        s._preprocessor = preprocessor
        s._postprocessor = postprocessor

    def __len__(s):
        return len(s._data)

    def split(s, *ratio):
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
        return next(s.loader(n, device))

    def loader(s,
        batch_size,
        device='cuda',
        shuffle=False,
        infinite=False,
        n_workers=1,
    ):
        return create_data_loader(_Dataset(s._data, s._preprocessor),
            batch_size, device, shuffle, infinite, n_workers, s._postprocessor)

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
