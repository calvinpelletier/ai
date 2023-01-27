import torch

import ai
from ai.examples.stylegan2.model import Generator, Discriminator
from ai.examples.stylegan2.train import StyleGan


def config(override={}):
    return ai.Config({
        'device': 'cuda',

        ## data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        'data.dataset': 'ffhq',
        'data.imsize': 256,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ## model
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # generator
        'model.G.z_dim': 512,
        'model.G.g.nc_min': 32,
        'model.G.g.nc_max': 512,
        'model.G.f.n_layers': 8,

        # discriminator
        'model.D.nc_min': 32,
        'model.D.nc_max': 512,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ## opt
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # generator
        'opt.G.type': 'adam',
        'opt.G.lr': 0.0025,

        # discriminator
        'opt.D.type': 'adam',
        'opt.D.lr': 0.0025,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ## train
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        'train.bs': 16,
        'train.n_data_workers': 2,
        'train.timelimit': None,
        'train.steplimit': None,
        'train.style_mix_prob': .9,

        # generator
        'train.G.reg.interval': 4,
        'train.G.reg.weight': 2.,
        'train.G.reg.batch_shrink': 2,
        'train.G.reg.decay': 0.01,

        # discriminator
        'train.D.reg.interval': 16,
        'train.D.reg.weight': 1.,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ## task
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        'task.bs': 64,
        'task.n_workers': 8,
        'task.n_imgs': 10_000,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    }, override)


class Trial(ai.lab.Trial):
    def __init__(s, path, cfg, **kw):
        super().__init__(path, **kw)
        s.cfg = cfg
        (s.path / 'samples').mkdir(exist_ok=True)

    def save_samples(s, step, models, opts):
        G = models['G'].eval()
        with ai.no_grad():
            out = G(torch.randn(64, G.z_dim, device=s.cfg.device))
        ai.util.img.save_img_grid(
            s.path / f'samples/{step}.png',
            out.reshape(8, 8, 3, s.cfg.data.imsize, s.cfg.data.imsize),
        )


def run(path, **kw):
    cfg = config(kw)
    print(cfg)

    trial = Trial(path, cfg, clean=True)
    print(f'path: {trial.path}')

    ds = ai.data.ImgDataset(cfg.data.dataset, cfg.data.imsize)

    task = _task(ds, trial.log, cfg.device, cfg.task)

    models = {
        'G': _G(cfg.data.imsize, cfg.model.G).init().to(cfg.device),
        'D': _D(cfg.data.imsize, cfg.model.D).init().to(cfg.device),
    }

    opts = {
        'G': ai.build_opt(cfg.opt.G, models['G']),
        'D': ai.build_opt(cfg.opt.D, models['D']),
    }

    hook = ai.train.Hook(trial.log)
    hook.add(trial.save_snapshot)
    hook.add(trial.save_samples)
    hook.add(lambda step, models, opts: task(models['G'], step))

    _train(cfg, ds, models, opts, hook)


def _train(cfg, ds, models, opts, hook, step=0):
    ai.train.MultiTrainer(
        StyleGan(
            aug=None, # TODO: ada
            g_reg_interval=cfg.train.G.reg.interval,
            g_reg_weight=cfg.train.G.reg.weight,
            d_reg_interval=cfg.train.D.reg.interval,
            d_reg_weight=cfg.train.D.reg.weight,
            style_mix_prob=cfg.train.style_mix_prob,
            pl_batch_shrink=cfg.train.G.reg.batch_shrink,
            pl_decay=cfg.train.G.reg.decay,
        ),
        ds.loader(
            cfg.train.bs,
            cfg.device,
            cfg.train.n_data_workers,
            train=True,
        ),
    ).train(
        models,
        opts,
        hook=hook,
        step=step,
        timelimit=cfg.train.timelimit,
        steplimit=cfg.train.steplimit,
    )


def _task(ds, log, device, cfg):
    return ai.task.ImgGenTask(
        ds, log, device, cfg.bs, cfg.n_workers, cfg.n_imgs)


def _G(imsize, cfg):
    return Generator(
        imsize, cfg.z_dim, cfg.g.nc_min, cfg.g.nc_max, cfg.f.n_layers)


def _D(imsize, cfg):
    return Discriminator(imsize, cfg.nc_min, cfg.nc_max)


if __name__ == '__main__':
    ai.fire(run)
