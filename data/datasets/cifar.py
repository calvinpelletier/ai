import torch
import torch.utils.data as torch_data
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
from typing import Callable, Optional

from ai.data.dataset import Dataset
from ai.util.path import dataset_path, PathLike
from ai.util.img import normalize
from ai.util import print_info


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck')


def cifar10(
    path: PathLike = 'cifar10',
    include_labels: bool = True,
    img_postprocessor: Callable = normalize,
    label_postprocessor: Optional[Callable] = None,
):
    '''CIFAR10 object classification dataset.

    DATA (n=60_000)
        x : [n, 3, 32, 32] (uint8)
            RGB image
        y : [n] (uint8)
            label (0-9), see CLASSES const above

    ARGS
        path : str or Path
            if path starts with "/":
                exact path to dataset root
            else:
                relative path from AI_DATASETS_PATH environment variable
            NOTE: if the data doesn't exist, it will be downloaded and
            converted.
        include_labels
            include labels (key: 'y') in dataset
        img_postprocessor
            postprocess function called on images
        label_postprocessor
            postprocess function called on labels
    '''

    path = dataset_path(path)
    path.mkdir(parents=True, exist_ok=True)

    path = path / 'data.npz'
    if not path.exists():
        _download_and_convert(path)

    data = np.load(path)
    if include_labels:
        postprocess = {'x': img_postprocessor}
        if label_postprocessor is not None:
            postprocess['y'] = label_postprocessor
    else:
        postprocess = img_postprocessor
        data = data['x']

    return Dataset(
        data,
        postprocess=postprocess, # type: ignore
        default_split=[50_000, 10_000],
    )


# TODO: refactor to remove duplicate code with other torchvision datasets


def _download_and_convert(path):
    tmp_path = Path('/tmp/cifar10_dataset')
    train = _load_torchvision_cifar10(tmp_path, True)
    test = _load_torchvision_cifar10(tmp_path, False)

    print('\nconverting cifar10 data...')

    n = len(train) + len(test)
    ds = {
        'x': np.empty([n, 3, 32, 32], dtype=np.uint8),
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


def _load_torchvision_cifar10(path, train):
    return datasets.CIFAR10(
        root=path,
        train=train,
        download=not path.exists(),
        transform=transforms.ToTensor(),
    )


def _create_loader(ds):
    return torch_data.DataLoader(
        ds,
        batch_size=None,
        shuffle=False,
        drop_last=False,
        num_workers=1,
    )
