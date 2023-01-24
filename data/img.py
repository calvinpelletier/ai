from torchvision import datasets, transforms

import ai


def img_dataset(path, imsize):
    # parse path to dataset
    path = ai.path.dataset(path)
    if not path.exists():
        raise ValueError(f'missing dataset: {path}')

    # resize imgs if needed
    dir = path / str(imsize)
    if not dir.exists():
        src_imsize = _choose_src_imsize(path, imsize)
        print(src_imsize)
        ai.util.img.resize_dir(path / str(src_imsize), dir, imsize)

    # create dataset
    paths = sorted(list(dir.iterdir()))
    return ai.data.Dataset(paths, ai.util.img.read, ai.util.img.normalize)


def _choose_src_imsize(path, target_imsize):
    imsizes = sorted([int(x.stem) for x in path.iterdir()])
    assert imsizes
    i = 0
    while i < len(imsizes) and imsizes[i] < target_imsize:
        i += 1
    if i < len(imsizes):
        return imsizes[i]
    return imsizes[-1]
