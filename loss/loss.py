import torch
from ai.train.log import log


class Loss(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s._device = torch.device('cpu')

    def to_device(s, device):
        if s._device != device:
            # print(f'[INFO] transfering loss to device {device}')
            s._device = device
            s.to(device)


class ComboLoss(torch.nn.Module):
    def __init__(s, **kwargs):
        super().__init__()
        s._losses = []
        for k, v in kwargs.items():
            if isinstance(v, tuple) or isinstance(v, list):
                assert len(v) == 2
                s._losses.append((k, v[0], v[1]))
            else:
                s._losses.append((k, v, 1.))

    def forward(s, *a, **kw):
        total_loss = 0.
        for name, fn, weight in s._losses:
            loss = fn(*a, **kw) * weight
            log(f'loss.{name}', loss)
            total_loss += loss
        return total_loss
