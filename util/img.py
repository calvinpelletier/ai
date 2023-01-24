import numpy as np
import pyspng
import torch
from tqdm import tqdm
from PIL import Image


def read(path):
    with open(path, 'rb') as f:
        img = pyspng.load(f.read())
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    return img.transpose(2, 0, 1) # hwc -> chw


def normalize(tensor):
    return tensor.to(torch.float32) / 127.5 - 1


def resize_dir(src, dest, imsize):
    dest.mkdir()
    for path in tqdm(list(src.iterdir())):
        im = Image.open(path)
        assert im.size[0] == im.size[1], 'TODO: handle non-square images'
        im = im.resize((imsize, imsize), Image.LANCZOS)
        im.save(dest / path.name, 'PNG')
