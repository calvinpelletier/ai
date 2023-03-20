import matplotlib.pyplot as plt

import ai
from ai.examples.diffusion.model import DiffusionMLP


EVAL_BS = 1000


def run(outpath, device='cpu', n_steps=5000, train_bs=32, sample_interval=500):
    ds = ai.data.toy.moons(n=8000, include_labels=False, mult=2.)

    model = DiffusionMLP(2).init().to(device)

    opt = ai.opt.AdamW(model, lr=1e-3, grad_clip=True)

    trial = ai.Trial(
        outpath,
        clean=True,
        sampler=_save_samples,
        sample_interval=sample_interval,
    )

    ai.Trainer(
        ai.train.Diffusion(),
        ds.iterator(train_bs, device, train=True),
    ).train(model, opt, trial.hook(), steplimit=n_steps)


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
