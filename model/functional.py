import torch
import torch.nn.functional as F


ALIGN_CORNERS_MODES = ['linear', 'bilinear', 'trilinear', 'bicubic']

def resample(x, stride, mode='bilinear'):
    return F.interpolate(
        x,
        scale_factor=1 / stride,
        mode=mode,
        align_corners=True if mode in ALIGN_CORNERS_MODES else None,
    )


def flatten(x, keep_batch_dim=True):
    if keep_batch_dim:
        return torch.flatten(x, 1)
    return torch.flatten(x)


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()
