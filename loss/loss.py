import torch


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
    def __init__(s, *args):
        super().__init__()
        s._losses = []
        for a in args:
            if isinstance(a, tuple) or isinstance(a, list):
                assert len(a) == 2
                s._losses.append(a)
            else:
                s._losses.append((a, 1.))

    def forward(s, *a, **kw):
        loss = 0.
        for fn, weight in s._losses:
            loss += fn(*a, **kw) * weight
        return loss
