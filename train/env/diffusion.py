import torch

from ai.train.env.base import Env
from ai.model import DiffusionModel


class Diffusion(Env):
    '''Diffusion environment.'''

    def __init__(s):
        s.loss_fn = torch.nn.MSELoss()

    def __call__(s,
        model: DiffusionModel,
        x: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        # choose random timesteps
        t = torch.randint(0, len(model), [x.shape[0]], device=x.device)

        # generate gaussian noise
        noise = torch.randn_like(x)

        # create noisy data
        noisy = model.noisify(x, t, noise)

        # predict the noise
        pred = model(noisy, t)

        # calc loss
        return s.loss_fn(pred, noise)
