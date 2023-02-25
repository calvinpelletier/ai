import torch
from pathlib import Path

import ai
from ai.examples.stylegan2.model import Generator, Discriminator
from ai.examples.stylegan2.train import StyleGan


BASE_CONFIG_PATH = Path(__file__).parent / 'config.yaml'


def run(output_path, **kw):
    cfg = ai.Config(BASE_CONFIG_PATH, override=kw)
    print(cfg)

    trial = ai.Trial(output_path, clean=True, sampler=_save_samples)
    print(f'path: {trial.path}')
    hook = trial.hook()

    ds = ai.data.ImgDataset(cfg.data.dataset, cfg.data.imsize)

    task = _task(ds, hook.val_log, cfg.device, cfg.task)

    models = {
        'G': _G(cfg.data.imsize, cfg.model.G).init().to(cfg.device),
        'D': _D(cfg.data.imsize, cfg.model.D).init().to(cfg.device),
    }

    opts = {
        'G': ai.opt.build(cfg.opt.G, models['G']),
        'D': ai.opt.build(cfg.opt.D, models['D']),
    }

    ai.train.MultiTrainer(
        StyleGan.from_cfg(cfg.train),
        ds.iterator(
            cfg.train.bs,
            cfg.device,
            cfg.train.n_data_workers,
            train=True,
        ),
    ).train(models, opts, hook, steplimit=6250)

    print(task(models['G'], 6250))


def _save_samples(dir, step, models):
    G = models['G'].eval()
    with ai.no_grad():
        out = G(torch.randn(G.imsize, G.z_dim, device=G.get_device()))
    ai.util.img.save_img_grid(
        dir / f'{step}.png',
        out.reshape(8, 8, 3, G.imsize, G.imsize),
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
    ai.run(run)
