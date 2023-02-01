import torch
from pathlib import Path

import ai
from ai.examples.stylegan2.model import Generator, Discriminator
from ai.examples.stylegan2.train import StyleGan


BASE_CONFIG_PATH = Path(__file__).parent / 'config.yaml'


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


def run(output_path, **kw):
    cfg = ai.Config(BASE_CONFIG_PATH, override=kw)
    print(cfg)

    trial = Trial(output_path, cfg, clean=True)
    print(f'path: {trial.path}')

    ds = ai.data.ImgDataset(cfg.data.dataset, cfg.data.imsize)

    task = _task(ds, trial.log, cfg.device, cfg.task)

    models = {
        'G': _G(cfg.data.imsize, cfg.model.G).init().to(cfg.device),
        'D': _D(cfg.data.imsize, cfg.model.D).init().to(cfg.device),
    }

    opts = {
        'G': ai.opt.build(cfg.opt.G, models['G']),
        'D': ai.opt.build(cfg.opt.D, models['D']),
    }

    hook = trial.hook(snapshot=True)
    hook.add(trial.save_samples)
    hook.add(lambda step, models, opts: task(models['G'], step))

    _train(cfg, ds, models, opts, hook)


def _train(cfg, ds, models, opts, hook, step=0):
    ai.train.MultiTrainer(
        StyleGan(cfg.train),
        ds.loader(
            cfg.train.bs,
            cfg.device,
            cfg.train.n_data_workers,
            train=True,
        ),
    ).train(models, opts, hook, step=step)


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
