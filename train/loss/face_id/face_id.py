import torch
from torch import nn

from ai.util import assert_shape
from ai.util.img import resize
from ai.train.loss.face_id.arcface import ArcFace
from ai.train.loss.loss import Loss


class FaceIdentityLoss(Loss):
    def __init__(s, detach_y=True):
        super().__init__()
        s._detach_y = detach_y
        s._model = ArcFace().eval()

    def _resize_and_crop(s, x):
        x = resize(x, 256)
        x = x[:, :, 35:223, 32:220]
        return resize(x, 112)

    def forward(s, x, y):
        s.to_device(x.device)

        bs = x.shape[0]
        size = x.shape[-1]
        assert_shape(x, [bs, 3, size, size])
        assert_shape(y, [bs, 3, size, size])

        x_out = s._model(s._resize_and_crop(x))
        y_out = s._model(s._resize_and_crop(y))

        if s._detach_y:
            y_out = y_out.detach()

        loss = (1. - torch.bmm(
            x_out.view(bs, 1, -1),
            y_out.view(bs, -1, 1),
        ))
        return loss.mean()
