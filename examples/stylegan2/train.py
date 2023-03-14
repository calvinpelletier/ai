import torch
import numpy as np

from ai.train import Gan, MultiTrainer
from ai import Config


# TODO: adaptive discriminator augmentation
# TODO: comments, docstrings, type hints


class StyleGan(Gan):
    @staticmethod
    def from_cfg(cfg: Config, aug=None):
        return StyleGan(aug, cfg.style_mix_prob, cfg.G.reg.interval,
            cfg.G.reg.weight, cfg.G.reg.batch_shrink, cfg.G.reg.decay,
            cfg.D.reg.interval, cfg.D.reg.weight)

    def __init__(s,
        aug=None,
        style_mix_prob=.9,
        g_reg_interval=4,
        g_reg_weight=2.,
        g_reg_batch_shrink=2,
        g_reg_decay=.01,
        d_reg_interval=16,
        d_reg_weight=.1,
    ):
        super().__init__(aug, g_reg_interval, g_reg_weight, d_reg_interval,
            d_reg_weight)
        s._style_mix_prob = style_mix_prob
        s._pl_batch_shrink = g_reg_batch_shrink
        s._pl_decay = g_reg_decay
        s._pl_mean = None

    def _g_reg(s, models, batch):
        bs = batch.shape[0] // s._pl_batch_shrink
        img, w = s._generate(models['G'], batch, True, bs)
        return s._path_length_reg(img, w)

    def _generate(s, generator, batch, return_w=False, bs=None):
        if bs is None:
            bs = batch.shape[0]
        z = s._random_z(generator.z_dim, bs, batch.device)

        w = generator.f(z)
        w = s._style_mix(generator, z, w)
        img = generator.g(w)

        if return_w:
            return img, w
        return img

    def _style_mix(s, generator, z, w):
        prob = s._style_mix_prob
        if prob is None or prob == 0.:
            return w

        cutoff = torch.empty(
            [],
            dtype=torch.int64,
            device=w.device,
        ).random_(1, w.shape[1])
        cutoff = torch.where(
            torch.rand([], device=w.device) < prob,
            cutoff,
            torch.full_like(cutoff, w.shape[1]),
        )
        w[:, cutoff:] = generator.f(
            torch.randn_like(z),
            update_w_ema=False,
        )[:, cutoff:]
        return w

    def _path_length_reg(s, img, w):
        if s._pl_mean is None:
            s._pl_mean = torch.zeros([], device=img.device)

        noise = torch.randn_like(img) / np.sqrt(img.shape[2] * img.shape[3])
        grads = torch.autograd.grad(
            outputs=[(img * noise).sum()],
            inputs=[w],
            create_graph=True,
            only_inputs=True,
        )[0]
        lengths = grads.square().sum(2).mean(1).sqrt()
        pl_mean = s._pl_mean.lerp(lengths.mean(), s._pl_decay)
        s._pl_mean.copy_(pl_mean.detach())
        loss = (lengths - pl_mean).square()
        return loss.mean()
