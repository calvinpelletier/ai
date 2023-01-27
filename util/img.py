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

def unnormalize(tensor):
    return ((tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8)


def to_pil(tensor):
    if len(tensor.shape) == 4:
        return [_to_pil(x) for x in tensor]
    assert len(tensor.shape) == 3
    return _to_pil(tensor)

def _to_pil(tensor):
    assert tensor.shape[0] == 3, 'TODO: non-RGB imgs'
    tensor = unnormalize(tensor).cpu().numpy()
    return Image.fromarray(np.transpose(tensor, (1, 2, 0)), 'RGB')


def resize_dir(src, dest, imsize):
    dest.mkdir()
    for path in tqdm(list(src.iterdir())):
        im = Image.open(path)
        assert im.size[0] == im.size[1], 'TODO: handle non-square images'
        im = im.resize((imsize, imsize), Image.LANCZOS)
        im.save(dest / path.name, 'PNG')


def create_img_grid(imgs):
    ny, nx, c, h, w = imgs.shape
    assert c == 3
    canvas = Image.new(
        'RGB',
        (w * nx, h * ny),
        'black',
    )
    for y, row in enumerate(imgs):
        for x, img in enumerate(row):
            canvas.paste(to_pil(img), (w * x, h * y))
    return canvas

def save_img_grid(path, imgs):
    create_img_grid(imgs).save(path)
