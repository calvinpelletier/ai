from typing import Optional, Callable

from ai.util.path import dataset_path, PathLike
from ai.util import img as img_util
from ai.data.dataset import Dataset


class ImgDataset(Dataset):
    '''Dataset of images.

    File structure:
        <path>/<imsize>/data/*.png (image files)
        <path>/<imsize>/metadata/* (e.g. for caching FID info)
    '''

    def __init__(s,
        path: PathLike,
        imsize: int,
        preprocess: Optional[Callable] = img_util.read,
        postprocess: Optional[Callable] = img_util.normalize,
    ):
        '''
        path : str or Path
            if path starts with "/":
                exact path to dataset root
            else:
                relative path from AI_DATASETS_PATH environment variable
        imsize : int
            target image size (if <path>/<imsize> doesnt exist, it will be
            created by resizing an existing image size)
        preprocess : callable or null
            a function used by data workers (e.g. read image from disk)
            args:
                an item from <data>
            returns:
                tensor/ndarray or dict of tensor/ndarray
        postprocess : callable or null
            a function called after the data has been transfered to the device
            (e.g. normalize an image tensor)
            args:
                tensor or dict of tensors
            returns:
                tensor or dict of tensors
        '''

        # parse path to dataset
        ds_path = dataset_path(path)
        if not ds_path.exists():
            raise ValueError(f'missing dataset: {ds_path}')

        # resize imgs if needed
        dir = ds_path / str(imsize)
        if not dir.exists():
            src_imsize = _choose_src_imsize(ds_path, imsize)
            print(f'\nresizing images from {src_imsize} to {imsize}...')
            img_util.resize_dir(
                ds_path / str(src_imsize) / 'data',
                dir / 'data',
                imsize,
            )
            print('done\n')

        # create dataset
        super().__init__(
            sorted(list((dir / 'data').iterdir())),
            preprocess,
            postprocess,
        )

        s.metadata_path = dir / 'metadata' # e.g. FID stats
        s.metadata_path.mkdir(exist_ok=True)


def _choose_src_imsize(path, target_imsize, allow_upsampling=False):
    imsizes = sorted([int(x.stem) for x in path.iterdir()])
    assert imsizes
    i = 0
    while i < len(imsizes) and imsizes[i] < target_imsize:
        i += 1
    if i < len(imsizes):
        return imsizes[i]
    if allow_upsampling:
        return imsizes[-1]
    raise Exception('no valid source imsizes ({}) for target imsize: {}'.format(
        ','.join([str(x) for x in imsizes]),
        target_imsize,
    ))
