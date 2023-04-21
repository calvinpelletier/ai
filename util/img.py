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


def to_pil(tensor, normalized=True):
    if len(tensor.shape) == 4:
        return [_to_pil(x, normalized) for x in tensor]
    assert len(tensor.shape) == 3
    return _to_pil(tensor, normalized)

def _to_pil(tensor, normalized):
    if normalized:
        tensor = unnormalize(tensor)
    else:
        tensor = tensor.clamp(0, 255).to(torch.uint8)

    # RGB
    if tensor.shape[0] == 3:
        return Image.fromarray(
            np.transpose(tensor.cpu().numpy(), (1, 2, 0)),
            'RGB',
        )

    # greyscale
    if tensor.shape[0] == 1:
        return Image.fromarray(
            tensor.squeeze().cpu().numpy(),
            'L',
        ).convert('RGB')

    raise ValueError(f'channel size not handled: {tensor.shape[0]}')


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


def create_img_grid(tensors, normalized=True):
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
            canvas.paste(to_pil(tensor, normalized), (w * x, h * y)) # type: ignore

    return canvas

def save_img_grid(path, tensors, normalized=True):
    create_img_grid(tensors, normalized).save(path)
