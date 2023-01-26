import torch

from ai.train import Trainer
from ai.train.env import Reconstruct
from ai.data.img import img_dataset
from ai.model.ae import ImgAutoencoder


DEVICE = 'cuda'


def test_reconstruction():
    _test_trainer(
        Reconstruct(),
        img_dataset('ffhq', 64),
        ImgAutoencoder(64, 4).init(),
        DEVICE,
    )


def _test_trainer(env, data, model, device='cuda', bs=8, lr=1e-3):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    data = data.loader(bs, device, train=True)
    sample = next(data)

    model.eval()
    with torch.no_grad():
        loss1 = env(model, sample)

    Trainer(env, data).train(model, opt, steplimit=8)

    model.eval()
    with torch.no_grad():
        loss2 = env(model, sample)
    assert loss2 < loss1