import torch
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np

from ai.data.dataset import Dataset
from ai.path import dataset_path
from ai.util.img import normalize


def mnist_dataset(path='mnist'):
    path = dataset_path(path)
    path.mkdir(parents=True, exist_ok=True)

    path = path / 'data.npz'
    if not path.exists():
        _download(path)
    data = np.load(path)

    return Dataset(data, postprocess={'x': normalize})


def _download(path):
    tmp_path = Path('/tmp/mnist_dataset')
    train = _load_torchvision_mnist(tmp_path, True)
    test = _load_torchvision_mnist(tmp_path, False)

    print('\nconverting mnist data...')

    n = len(train) + len(test)
    ds = {
        'x': np.empty([n, 1, 28, 28], dtype=np.uint8),
        'y': np.empty([n], dtype=np.uint8),
    }

    i = 0
    for data in _create_loader(train):
        i = _store(ds, i, data)
    for data in _create_loader(test):
        i = _store(ds, i, data)
    assert i == n

    np.savez(path, **ds)

    print('done\n')


def _store(ds, i, data):
    x, y = data
    ds['x'][i] = (x * 255).clamp(0, 255).to(torch.uint8).numpy()
    ds['y'][i] = y
    return i + 1


def _load_torchvision_mnist(path, train):
    return datasets.MNIST(
        path,
        train=train,
        download=not path.exists(),
        transform=transforms.ToTensor(),
    )


def _create_loader(ds):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=None,
        shuffle=False,
        drop_last=False,
        num_workers=1,
    )
