import torch
from pathlib import Path

import ai
from ai.examples.stylegan2.model import Generator, Discriminator
from ai.examples.stylegan2.train import StyleGan
from ai.data.datasets.ffhq import MISSING_FFHQ_MSG


BASE_CONFIG_PATH = Path(__file__).parent / 'config.yaml'


def run(outpath, steplimit=5000, **kw):
    cfg = ai.Config(BASE_CONFIG_PATH, override=kw)
    print(cfg)

    try:
        ds = ai.data.ImgDataset(cfg.data.dataset, cfg.data.imsize)
    except ValueError as e:
        print(e)
        print(MISSING_FFHQ_MSG.format(imsize=cfg.data.imsize))
        return

    task = ai.task.ImgGenTask(ds, cfg.device, cfg.task.bs, cfg.task.n_workers,
        cfg.task.n_imgs)

    trial = ai.Trial(outpath, clean=True, sampler=_save_samples)
    print(f'output path: {trial.path}')

    models = {
        'G': _build_G(cfg.data.imsize, cfg.model.G).init().to(cfg.device),
        'D': _build_D(cfg.data.imsize, cfg.model.D).init().to(cfg.device),
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
    ).train(models, opts, trial.hook(), steplimit=steplimit)

    fid = task(models)
    print(f'fid: {fid}')


def _save_samples(dir, step, models):
    G = models['G']
    with ai.no_grad():
        out = G(torch.randn(64, G.z_dim, device=G.get_device()))
    ai.util.img.save_img_grid(
        dir / f'{step}.png',
        out.reshape(8, 8, 3, G.imsize, G.imsize),
    )


def _build_G(imsize, cfg):
    return Generator(imsize, cfg.z_dim, cfg.g.nc_min, cfg.g.nc_max,
        cfg.f.n_layers)

def _build_D(imsize, cfg):
    return Discriminator(imsize, cfg.nc_min, cfg.nc_max)


if __name__ == '__main__':
    ai.run(run)
