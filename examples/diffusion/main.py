import torch
import matplotlib.pyplot as plt

import ai
from ai.examples.diffusion.model import DiffusionMLP


TRAIN_BS = 32
EVAL_BS = 1000
DEVICE = 'cpu'
SAMPLE_INTERVAL = 500
N_STEPS = 5000


def run(outpath):
    ds = ai.data.toy.moons(n=8000, include_labels=False, mult=2.)

    model = DiffusionMLP(2).init().to(DEVICE)

    opt = ai.opt.AdamW(model, lr=1e-3, grad_clip=True)

    trial = ai.Trial(
        outpath,
        clean=True,
        sampler=_save_samples,
        sample_interval=SAMPLE_INTERVAL,
    )

    ai.Trainer(
        ai.train.Diffusion(),
        ds.iterator(TRAIN_BS, DEVICE, train=True),
    ).train(model, opt, trial.hook(), steplimit=N_STEPS)


def _save_samples(dir, step, model):
    dir = dir / str(step)
    dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(model.sample(EVAL_BS, frame_rate=10)):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.savefig(dir / f'{i:04}.png')
        plt.close()


if __name__ == '__main__':
    ai.run(run)
