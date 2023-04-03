import torch
from typing import Optional, Callable

from ai.train.env.gan import Gan
from ai.train.logger import log


class AdversarialAE(Gan):
    '''Training environment for Adversarial Autoencoders.

    ARGS
        rec_loss_fn : callable
            reconstruction loss function (default: MSE)
        rec_loss_weight : float
            weight of the reconstruction loss relative to adversarial loss
        aug : callable or null
            optional augmentation before calling the discriminator
            TODO: adaptive discriminator augmentation
        d_reg_interval : int or null
            perform regularization for the discrim every d_reg_interval steps.
            default is gradient penalty (https://arxiv.org/pdf/1704.00028.pdf)
        d_reg_weight : float
            weight of the discriminator's regularization loss
    '''

    def __init__(s,
        rec_loss_fn: Callable = torch.nn.MSELoss(),
        rec_loss_weight: float = 1.,
        aug: Optional[Callable] = None,
        d_reg_interval: Optional[int] = 16,
        d_reg_weight: float = 1.,
    ):
        super().__init__(
            aug=aug, 
            d_reg_interval=d_reg_interval, 
            d_reg_weight=d_reg_weight,
        )
        s._rec_loss_fn = rec_loss_fn
        s._rec_loss_weight = rec_loss_weight

    def _g_main(s, models, batch):
        g_out = s._generate(models['G'], batch)
        rec_loss = s._rec_loss_fn(g_out, batch.detach())
        log('G.loss.main.rec', rec_loss)

        d_out = s._discriminate(models['D'], g_out, detach=False)
        adv_loss = s._g_loss_fn(d_out)
        log('G.loss.main.adv', adv_loss)

        return rec_loss * s._rec_loss_weight + adv_loss 
    
    def _generate(s, autoencoder, batch):
        return autoencoder(batch)

