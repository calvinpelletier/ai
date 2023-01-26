import torch
import torch.nn.functional as F

from ai.train.env import Env
from ai.train.util import on_interval


class Gan(Env):
    def __init__(s,
        aug=None,
        g_reg_interval=None,
        g_reg_weight=1.,
        d_reg_interval=16,
        d_reg_weight=1.,
    ):
        super().__init__()
        s._aug = aug
        s._g_reg_interval, s._g_reg_weight = g_reg_interval, g_reg_weight
        s._d_reg_interval, s._d_reg_weight = d_reg_interval, d_reg_weight

    # generator step
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def G(s, models, batch, step=0):
        # main
        loss = s._g_main(models, batch)
        s.log('loss.G.main', loss)

        # regularize
        if on_interval(step, s._g_reg_interval):
            reg_loss = s._g_reg(models, batch)
            s.log('loss.G.reg', reg_loss)
            loss += reg_loss * s._g_reg_weight

        return loss

    def _g_main(s, models, batch):
        g_out = s._generate(models['G'], batch)
        d_out = s._discriminate(models['D'], g_out, detach=False)
        return s._g_loss_fn(d_out)

    def _g_reg(s, models, batch):
        raise NotImplementedError(
            'set g_reg_interval without implementing _g_reg')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # discriminator step
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def D(s, models, batch, step=0):
        # main
        loss = s._d_main(models, batch)
        s.log('loss.D.main', loss)

        # regularize
        if on_interval(step, s._d_reg_interval):
            reg_loss = s._d_reg(models, batch)
            s.log('loss.D.reg', reg_loss)
            loss += reg_loss * s._d_reg_weight

        return loss

    def _d_main(s, models, batch):
        G, D = models['G'], models['D']
        g_out = s._generate(G, batch)
        loss_fake = s._d_loss_fn(s._discriminate(D, g_out, True), False)
        loss_real = s._d_loss_fn(s._discriminate(D, batch, True), True)
        return loss_fake + loss_real

    def _d_reg(s, models, batch):
        return s._gradient_penalty(models['D'], batch)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # helpers
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _generate(s, generator, batch):
        z = s._random_z(generator.z_dim, batch.shape[0], batch.device)
        return generator(z)

    def _discriminate(s, discriminator, x, detach):
        if s._aug is not None:
            x = s._aug(x)
        if detach:
            x = x.detach()
        return discriminator(x)

    def _g_loss_fn(s, logits):
        return F.softplus(-logits).mean()

    def _d_loss_fn(s, logits, is_real):
        sign = -1 if is_real else 1
        return F.softplus(sign * logits).mean()

    def _random_z(s, z_dim, bs, device):
        return torch.randn([bs, z_dim], device=device)

    def _gradient_penalty(s, discriminator, batch):
        input_real = batch.detach().requires_grad_(True)
        logits_real = s._discriminate(discriminator, input_real, False)
        grads = torch.autograd.grad(
            outputs=[logits_real.sum()],
            inputs=[input_real],
            create_graph=True,
            only_inputs=True,
        )[0]
        loss = grads.square().sum([1, 2, 3])
        return loss.mean()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
