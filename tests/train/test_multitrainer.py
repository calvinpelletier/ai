import torch

from ai.util.testing import *
from ai.train import MultiTrainer
from ai.train.env.gan import Gan
from ai.data.img import ImgDataset
from ai.examples.stylegan2.model import Generator, Discriminator
from ai.examples.stylegan2.train import StyleGan


def test_gan():
    _test_multitrainer(
        Gan(),
        ImgDataset('ffhq', 64),
        {
            'G': Generator(64).init(),
            'D': Discriminator(64).init(),
        },
        DEVICE,
    )


def test_stylegan():
    _test_multitrainer(
        StyleGan(),
        ImgDataset('ffhq', 64),
        {
            'G': Generator(64).init(),
            'D': Discriminator(64).init(),
        },
        DEVICE,
    )


def _test_multitrainer(env, data, models, device='cuda', bs=8, lr=1e-3):
    data = data.iterator(bs, device, train=True)

    opts = {}
    for k, model in models.items():
        model.to(device)
        opts[k] = torch.optim.SGD(model.parameters(), lr=lr)

    trainer = MultiTrainer(env, data)
    trainer.train(models, opts, steplimit=2)
