import torch

from ai.util.testing import *
from ai.train import Trainer
from ai.train.env import Reconstruct
from ai.data.img import ImgDataset
from ai.model.ae import ImgAutoencoder


def test_reconstruction():
    _test_trainer(
        Reconstruct(),
        ImgDataset('ffhq', 64),
        ImgAutoencoder(64, 4).init(),
        DEVICE,
    )


def _test_trainer(env, data, model, device='cuda', bs=8, lr=1e-3):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    data = data.iterator(bs, device, train=True)
    sample = next(iter(data))

    with torch.no_grad():
        loss1 = env(model, sample)

    Trainer(env, data).train(model, opt, steplimit=8)

    with torch.no_grad():
        loss2 = env(model, sample)

    assert loss2 < loss1
