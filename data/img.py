from torchvision import datasets, transforms

from ai.path import dataset as dataset_path
from ai.util import img as img_util
from ai.data.dataset import Dataset


class ImgDataset(Dataset):
    def __init__(s, path, imsize):
        # parse path to dataset
        path = dataset_path(path)
        if not path.exists():
            raise ValueError(f'missing dataset: {path}')

        # resize imgs if needed
        dir = path / str(imsize)
        if not dir.exists():
            src_imsize = _choose_src_imsize(path, imsize)
            img_util.resize_dir(
                path / str(src_imsize) / 'data',
                dir / 'data',
                imsize,
            )

        # create dataset
        super().__init__(
            sorted(list((dir / 'data').iterdir())),
            img_util.read,
            img_util.normalize,
        )

        s.metadata_path = dir / 'metadata' # e.g. FID stats


def _choose_src_imsize(path, target_imsize):
    imsizes = sorted([int(x.stem) for x in path.iterdir()])
    assert imsizes
    i = 0
    while i < len(imsizes) and imsizes[i] < target_imsize:
        i += 1
    if i < len(imsizes):
        return imsizes[i]
    return imsizes[-1]
