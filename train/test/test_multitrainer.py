import torch

from ai.train import MultiTrainer
from ai.train.gan import Gan
from ai.data.img import img_dataset
from ai.examples.stylegan2.model import Generator, Discriminator
from ai.examples.stylegan2.train import StyleGan


DEVICE = 'cuda'


def test_gan():
    _test_multitrainer(
        Gan(),
        img_dataset('ffhq', 64),
        {
            'G': Generator(64).init(),
            'D': Discriminator(64),
        },
        DEVICE,
    )


def test_stylegan():
    _test_multitrainer(
        StyleGan(),
        img_dataset('ffhq', 64),
        {
            'G': Generator(64).init(),
            'D': Discriminator(64),
        },
        DEVICE,
    )


def _test_multitrainer(env, data, models, device='cuda', bs=8, lr=1e-3):
    data = data.loader(bs, device, train=True)

    opts = {}
    for k, model in models.items():
        model.to(device)
        opts[k] = torch.optim.SGD(model.parameters(), lr=lr)

    trainer = MultiTrainer(env, data)
    i = trainer.train(models, opts, steplimit=2)
    assert i == 2
