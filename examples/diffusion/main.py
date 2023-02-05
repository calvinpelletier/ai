import torch
import matplotlib.pyplot as plt

import ai
from ai.examples.diffusion.model import DiffusionMLP


TRAIN_BS = 32
EVAL_BS = 1000
DEVICE = 'cpu'
INSPECT_INTERVAL = 500
N_STEPS = 5000


def run(output_path):
    ds = ai.data.toy.moons_dataset(n=8000, include_labels=False, mult=2.)

    model = DiffusionMLP(2).init().to(DEVICE)

    opt = ai.opt.adamw(model, lr=1e-3, grad_clip=True)

    trial = Trial(output_path, clean=True)

    hook = trial.hook(snapshot=False)
    hook.add(trial.inspect, INSPECT_INTERVAL)

    ai.Trainer(
        ai.train.Diffusion(),
        ds.loader(TRAIN_BS, DEVICE, train=True),
    ).train(model, opt, hook, steplimit=N_STEPS)

    trial.finalize()


class Trial(ai.lab.Trial):
    def __init__(s, path, **kw):
        super().__init__(path, **kw)
        s.frames = []

    def inspect(s, step, model, opt):
        model.eval()
        s.frames.append((step, model.sample(EVAL_BS, frame_rate=10)))

    def finalize(s):
        for step, frames in s.frames:
            imgdir = s.path / 'imgs' / str(step)
            imgdir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                plt.figure(figsize=(10, 10))
                plt.scatter(frame[:, 0], frame[:, 1])
                plt.xlim(-6, 6)
                plt.ylim(-6, 6)
                plt.savefig(imgdir / f'{i:04}.png')
                plt.close()


if __name__ == '__main__':
    ai.fire(run)
