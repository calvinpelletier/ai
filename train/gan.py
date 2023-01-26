import torch
import torch.nn.functional as F

from ai.train.util import on_interval


class Gan:
    def __init__(s, aug=None, d_reg_interval=16, d_reg_weight=1.):
        s._aug = aug
        s._d_reg_interval = d_reg_interval
        s._d_reg_weight = d_reg_weight

    # generator step
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def G(s, models, batch, step=0):
        g_out = s._generate(models['G'], batch)
        d_out = s._discriminate(models['D'], g_out, detach=False)
        return s._g_loss_fn(d_out)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # discriminator step
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def D(s, models, batch, step=0):
        loss = s._d_main(models, batch)
        if on_interval(step, s._d_reg_interval):
            loss += s._d_reg(models, batch) * s._d_reg_weight
        return loss

    def _d_main(s, models, batch):
        G, D = models['G'], models['D']
        g_out = s._generate(G, batch)
        loss_fake = s._d_loss_fn(s._discriminate(D, g_out, detach=True), False)
        loss_real = s._d_loss_fn(s._discriminate(D, batch, detach=True), True)
        return loss_fake + loss_real

    def _d_reg(s, models, batch):
        return s._gradient_penalty(models['D'], batch)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # helpers
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _generate(s, generator, batch):
        z = s._random_z(generator, batch)
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

    def _random_z(s, generator, x):
        return torch.randn([x.shape[0], generator.z_dim], device=x.device)

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
