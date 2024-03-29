import torch
from shutil import rmtree
from pathlib import Path

from ai.util.testing import *
from ai.train import Trainer, MultiTrainer
from ai.train.env import Reconstruct
from ai.data.img import ImgDataset
from ai.model.ae import ImgAutoencoder
from ai.util.logger import Tensorboard
from ai.train.hook import Hook
from ai.train.env.gan import Gan
from ai.examples.stylegan2.model import Generator, Discriminator
from ai.examples.stylegan2.train import StyleGan


TENSORBOARD_PATH = Path('/tmp/ai/tensorboard')


def test_tensorboard_hook():
    _test_tensorboard_hook(TENSORBOARD_PATH / 'ae', _train)
    _test_tensorboard_hook(TENSORBOARD_PATH / 'sg2', _multitrain)


def _test_tensorboard_hook(path, train_fn):
    if path.exists():
        rmtree(path)
    hook = Hook(log={
        'fn': Tensorboard(path),
        'interval': 10,
    })
    train_fn(hook)
    assert path.exists()


def _train(hook):
    env = Reconstruct()
    data = ImgDataset('ffhq', 64).iterator(8, DEVICE, train=True)

    model = ImgAutoencoder(64, 4).init().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    Trainer(env, data).train(model, opt, hook, steplimit=16)


def _multitrain(hook):
    env = StyleGan()
    data = ImgDataset('ffhq', 64).iterator(8, DEVICE, train=True)

    models = {
        'G': Generator(64).init(),
        'D': Discriminator(64),
    }
    opts = {}
    for k, model in models.items():
        model.to(DEVICE)
        opts[k] = torch.optim.SGD(model.parameters(), lr=1e-3)

    MultiTrainer(env, data).train(models, opts, hook, steplimit=16)
