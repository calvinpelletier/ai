import torch
from torch import nn

from ai.util import assert_shape
from ai.util.img import resize
from ai.model.pretrained import ArcFace


class FaceIdentityLoss(nn.Module):
    def __init__(s, device='cuda'):
        super().__init__()
        s._model = ArcFace().eval()
        s.to(device)

    def _resize_and_crop(s, x):
        x = resize(x, 256)
        x = x[:, :, 35:223, 32:220]
        return resize(x, 112)

    def forward(s, x, y):
        bs = x.shape[0]
        assert_shape(x, [bs, 3, -1, -1])
        assert_shape(y, [bs, 3, -1, -1])

        x_out = s._model(s._resize_and_crop(x))
        y_out = s._model(s._resize_and_crop(y))

        loss = (1. - torch.bmm(
            x_out.view(bs, 1, -1),
            y_out.view(bs, -1, 1),
        ))
        return loss.mean()
