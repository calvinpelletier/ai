import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lpips import LPIPS

from ai.loss.loss import Loss
from ai.util.img import resize


class PerceptualLoss(Loss):
    def __init__(s, type_='trad-vgg', detach_y=True):
        '''
        type_ : str
            "lpips-alex", "lpips-vgg", or "trad-vgg"
        detach_y : bool
            detach second argument in forward pass before calculating loss
        '''

        super().__init__()
        s._detach_y = detach_y

        type_, net = type_.split('-')
        if type_ == 'lpips':
            s._loss_fn = LPIPS(net=net)
        elif type_ == 'trad':
            s._loss_fn = TradPerceptualLoss(net, detach_y)
        else:
            raise ValueError(type_)

    def forward(s, x, y):
        s.to_device(x.device)

        if s._detach_y:
            y = y.detach()

        return s._loss_fn(x, y).mean()


class TradPerceptualLoss(nn.Module):
    def __init__(s, type_='vgg', detach_y=True):
        super().__init__()
        s._detach_y = detach_y

        assert type_ == 'vgg', f'unsupported perceptual model: {type}'
        s._model = _VGG19().eval()

        s._loss_fn = nn.L1Loss()
        s._weights = [1. / 32, 1. / 16, 1. / 8, 1. / 4, 1.]

        # imagenet normalization
        s.register_buffer('_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        s.register_buffer('_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(s, x, y):
        x_out = s._model(resize(s._normalize(x), 224))
        y_out = s._model(resize(s._normalize(y), 224))

        if s._detach_y:
            y_out = [slice.detach() for slice in y_out]

        loss = 0.
        for i in range(len(x_out)):
            loss += s._weights[i] * s._loss_fn(x_out[i], y_out[i])
        return loss

    def _normalize(s, x):
        # [-1, 1] to [0, 1]
        x = (x + 1) / 2

        # [0, 1] to imagenet normalized
        return (x - s._mean) / s._std


class _VGG19(nn.Module):
    def __init__(s, requires_grad=False):
        super().__init__()

        vgg = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.DEFAULT,
        ).features

        s._slice1 = nn.Sequential(*[vgg[x] for x in range(2)])
        s._slice2 = nn.Sequential(*[vgg[x] for x in range(2, 7)])
        s._slice3 = nn.Sequential(*[vgg[x] for x in range(7, 12)])
        s._slice4 = nn.Sequential(*[vgg[x] for x in range(12, 21)])
        s._slice5 = nn.Sequential(*[vgg[x] for x in range(21, 30)])

        if not requires_grad:
            for param in s.parameters():
                param.requires_grad = False

    def forward(s, x):
        feat1 = s._slice1(x)
        feat2 = s._slice2(feat1)
        feat3 = s._slice3(feat2)
        feat4 = s._slice4(feat3)
        feat5 = s._slice5(feat4)
        return [feat1, feat2, feat3, feat4, feat5]
