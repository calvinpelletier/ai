import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, List

from ai.model.module import Model


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
            return sample

        frames = []
        for i, t in enumerate(s.denoise_iter()):
            if i % frame_rate == 0:
                frames.append(sample.cpu().numpy())
            sample = s.denoise(sample, t)
        frames.append(sample.cpu().numpy())
        return np.stack(frames)

    def denoise(s, x: Tensor, t: int) -> Tensor:
        t = torch.full([x.shape[0]], t, dtype=torch.long, device=x.device)
        noise_pred = s(x, t)
        return s.noisify.denoise(x, t[0], noise_pred)

    def denoise_iter(s) -> List[int]:
        return list(range(s._n_timesteps))[::-1]

    def gen_noise(s, bs: int) -> Tensor:
        return torch.randn(bs, *s._shape, device=s.get_device())


class Noisify(torch.nn.Module):
    def __init__(s,
        shape: List[int],
        n_timesteps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
    ):
        super().__init__()
        assert len(shape) == 1, 'TODO'

        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, n_timesteps,
                dtype=torch.float32)
        elif beta_schedule == 'quadratic':
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
                n_timesteps, dtype=torch.float32) ** 2
        else:
            raise ValueError(beta_schedule)
        s.register_buffer('_betas', betas)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        s.register_buffer('_alphas_cumprod', alphas_cumprod)
        alphas_cumprod_prev = F.pad(
            alphas_cumprod[:-1], (1, 0), value=1.)
        s.register_buffer('_alphas_cumprod_prev', alphas_cumprod_prev)

        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        s.register_buffer('_sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5
        s.register_buffer('_sqrt_one_minus_alphas_cumprod',
            sqrt_one_minus_alphas_cumprod)

        sqrt_inv_alphas_cumprod = torch.sqrt(1 / alphas_cumprod)
        s.register_buffer('_sqrt_inv_alphas_cumprod', sqrt_inv_alphas_cumprod)
        sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / alphas_cumprod - 1)
        s.register_buffer('_sqrt_inv_alphas_cumprod_minus_one',
            sqrt_inv_alphas_cumprod_minus_one)

        posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) /
            (1. - alphas_cumprod))
        s.register_buffer('_posterior_mean_coef1', posterior_mean_coef1)
        posterior_mean_coef2 = ((1. - alphas_cumprod_prev) *
            torch.sqrt(alphas) / (1. - alphas_cumprod))
        s.register_buffer('_posterior_mean_coef2', posterior_mean_coef2)

    def __call__(s, x: Tensor, t: int, noise: Tensor) -> Tensor:
        s1 = s._sqrt_alphas_cumprod[t].reshape(-1, 1)
        s2 = s._sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        return s1 * x + s2 * noise

    def denoise(s, x: Tensor, t: int, pred: Tensor) -> Tensor:
        pred_original_sample = s._reconstruct_x0(x, t, pred)
        pred_prev_sample = s._q_posterior(pred_original_sample, x, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(pred)
            variance = (s._get_variance(t) ** 0.5) * noise

        return pred_prev_sample + variance

    def _reconstruct_x0(s, x_t, t, noise):
        s1 = s._sqrt_inv_alphas_cumprod[t]
        s2 = s._sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def _q_posterior(s, x_0, x_t, t):
        s1 = s._posterior_mean_coef1[t]
        s2 = s._posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def _get_variance(s, t):
        if t == 0:
            return 0
        variance = (s._betas[t] * (1. - s._alphas_cumprod_prev[t]) /
            (1. - s._alphas_cumprod[t]))
        variance = variance.clip(1e-20)
        return variance
