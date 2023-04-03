import ai
from ai.model.ae import ImgAutoencoder
from ai.data.datasets.ffhq import MISSING_FFHQ_MSG
from ai.examples.stylegan2.model import Discriminator
from ai.util.testing import DEVICE


def test_aae():
    outpath = '/tmp/test/aae'
    imsize = 64
    bottleneck = 4
    nc_min = 16
    nc_max = 256
    batch_size = 32
    n_samples = 8
    lr = 0.0025
    rec_loss_weight = 1.
    steplimit = 4
    
    try:
        ds = ai.data.ImgDataset('ffhq', imsize)
    except ValueError as e:
        print(e)
        print(MISSING_FFHQ_MSG.format(imsize=imsize))
        return
    val_ds, train_ds = ds.split(.01, .99)
    train_iter = train_ds.iterator(batch_size, DEVICE, train=True)
    samples = val_ds.sample(n_samples, DEVICE)

    trial = ai.Trial(
        outpath, 
        clean=True,
        sampler=lambda path, step, models: ai.util.save_img_grid(
            path / f'{step}.png',
            [samples, models['G'](samples)],
        ),
    )

    models = {
        'G': ImgAutoencoder(imsize, bottleneck, nc_min, nc_max),
        'D': Discriminator(imsize, nc_min, nc_max),
    }
    for model in models.values():
        model.init().to(DEVICE)

    opts = {
        'G': ai.opt.Adam(models['G'], lr=lr, betas=[0, .99]),
        'D': ai.opt.Adam(models['D'], lr=lr, betas=[0, .99]),
    }

    ai.train.MultiTrainer(
        ai.train.AdversarialAE(
            rec_loss_fn=ai.loss.L2Loss(), 
            rec_loss_weight=rec_loss_weight,
        ),
        train_iter,
    ).train(models, opts, trial.hook(), steplimit=steplimit) # type: ignore
