import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, List, Union

from ai.model.module import Model
from ai.util.noise import Noisify


class DiffusionModel(Model):
    def __init__(s, shape: List[int], n_timesteps: int = 50, **kw):
        super().__init__()
        s._shape = shape
        s._n_timesteps = n_timesteps
        s.noisify = Noisify(shape, n_timesteps, **kw)

    def __len__(s) -> int:
        return s._n_timesteps

    def sample(s, n: int, frame_rate: Optional[int] = None) -> np.ndarray:
        sample = s.gen_noise(n)

        if frame_rate is None:
            for t in s.denoise_iter():
                sample = s.denoise(sample, t)
            return sample.cpu().numpy()

        frames = []
        for i, t in enumerate(s.denoise_iter()):
            if i % frame_rate == 0:
                frames.append(sample.cpu().numpy())
            sample = s.denoise(sample, t)
        frames.append(sample.cpu().numpy())
        return np.stack(frames)

    def denoise(s, x: Tensor, step: int) -> Tensor:
        t = torch.full([x.shape[0]], step, dtype=torch.long, device=x.device)
        noise_pred = s(x, t)
        return s.noisify.denoise(x, step, noise_pred)

    def denoise_iter(s) -> List[int]:
        return list(range(s._n_timesteps))[::-1]

    def gen_noise(s, bs: int) -> Tensor:
        return torch.randn(bs, *s._shape, device=s.get_device())


