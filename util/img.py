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


def resize(tensor, size, mode='bilinear', align_corners=True):
    assert tensor.shape[-1] == tensor.shape[-2], 'TODO: resize non-square img'
    if tensor.shape[-1] == size:
        return tensor
    return torch.nn.functional.interpolate(
        tensor,
        size=(size, size),
        mode='bilinear',
        align_corners=align_corners,
    )


def resize_dir(src, dest, imsize):
    dest.mkdir(parents=True, exist_ok=True)
    for path in tqdm(list(src.iterdir())):
        im = Image.open(path)
        assert im.size[0] == im.size[1], 'TODO: handle non-square images'
        im = im.resize((imsize, imsize), Image.LANCZOS)
        im.save(dest / path.name, 'PNG')


def create_img_grid(tensors):
    nx = len(tensors)
    ny = len(tensors[0])
    c, h, w = tensors[0][0].shape

    assert c == 3
    canvas = Image.new(
        'RGB',
        (w * nx, h * ny),
        'black',
    )

    for x, col in enumerate(tensors):
        for y, tensor in enumerate(col):
            canvas.paste(to_pil(tensor), (w * x, h * y))

    return canvas

def save_img_grid(path, tensors):
    create_img_grid(tensors).save(path)
